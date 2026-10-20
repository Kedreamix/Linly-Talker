"""Actual WebUI dispatch and MCP transport tests with a scripted model.

Only model/audio/GPU dependencies are stubbed. The production LLM registry,
WebUI component wiring, Gradio SessionState and MCP HTTP client execute normally.
"""

import asyncio
import importlib
import json
import os
import sys
import types
import unittest
from unittest.mock import patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import httpx
from gradio.blocks import SessionState

from src import web_search


class ScriptedModel:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def chat(self, system, message, history):
        self.calls.append((system, message))
        answer = "Grounded" if "example.org" in system else "Local"
        return answer, history + [(message, answer)]

    def clear_history(self):
        pass


def load_webui():
    modules = {name: types.ModuleType(name) for name in ("torch", "torchaudio", "zhconv", "TTS")}
    modules["zhconv"].convert = lambda text, _: text
    modules["TTS"].EdgeTTS = lambda: types.SimpleNamespace(SUPPORTED_VOICE=["zh-CN-XiaoxiaoNeural"])
    for name in ("Linly", "Qwen", "Qwen2", "Gemini", "ChatGPT", "ChatGLM", "Llama2Chinese", "GPT4Free", "QAnything"):
        module = types.ModuleType("LLM." + name)
        setattr(module, "GPT4FREE" if name == "GPT4Free" else name, ScriptedModel)
        modules[module.__name__] = module
    with patch.dict(sys.modules, modules):
        return importlib.import_module("webui")


class MCPServer:
    def __init__(self):
        self.requests = []
        self.error = False
        self.malformed = False
        self.missing_tool = False
        self.slow = False
        self.cancelled = False
        self.http_error = False
        self.sse = False
        self.empty = False

    async def handle(self, request):
        self.requests.append(request)
        if self.http_error:
            return httpx.Response(503, text="Unavailable")
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "test", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": [] if self.missing_tool else [
                {"name": name, "inputSchema": {"type": "object"}}
                for name in ("web_search", "web_fetch")]}
        elif method == "tools/call":
            if self.slow:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            result = {"isError": self.error, "content": [
                {"type": "text", "text": "Quota exceeded" if self.error else "Evidence: https://example.org/\nCurrent fact"}]}
            if self.empty:
                result["content"] = []
        else:
            raise AssertionError(method)
        if self.malformed:
            return httpx.Response(200, text="invalid", headers={"content-type": "application/json"})
        body = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if self.sse:
            return httpx.Response(200, text="event: message\ndata: " + json.dumps(body) + "\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=body)

    def tool_calls(self):
        return [json.loads(r.content)["params"] for r in self.requests
                if json.loads(r.content)["method"] == "tools/call"]


class WebLookupTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.webui = load_webui()
        cls.webui.GPT_SoVITS_ckpt = "GPT_SoVITS/pretrained_models"
        cls.app = cls.webui.app_multi()
        functions = list(cls.app.fns.items()) if isinstance(cls.app.fns, dict) else list(enumerate(cls.app.fns))
        cls.fn_index, cls.fn = next((index, fn) for index, fn in functions if fn.fn is cls.webui.chat_response)
        cls.clear_index = next(index for index, fn in functions if fn.fn is cls.webui.clear_session)
        cls.system_index = next(index for index, fn in functions if fn.fn is cls.webui.modify_system_session)
        cls.state_id = cls.fn.inputs[-1]._id

    def setUp(self):
        self.environment = patch.dict(os.environ, {"LINLY_ALLOW_WEB_SEARCH": "1",
                                                 "PARALLEL_API_KEY": "ambient-key-must-not-be-used"})
        self.environment.start()
        self.server = MCPServer()
        original_client = httpx.AsyncClient
        self.http = patch("httpx.AsyncClient", side_effect=lambda **kwargs: original_client(
            **kwargs, transport=httpx.MockTransport(self.server.handle)))
        self.http.start()
        self.webui.llm_class = self.webui.LLM(mode="offline")
        # Exercise the real model registry; the backend is deliberately scripted.
        self.webui.llm = self.webui.llm_class.init_model("Qwen2")
        self.state = SessionState(self.app)

    def tearDown(self):
        self.http.stop()
        self.environment.stop()

    async def dispatch(self, message="current facts", history=None, enabled=None, state=None):
        state = self.state if state is None else state
        enabled = self.fn.inputs[3].value if enabled is None else enabled
        inputs = ["System", message, history or [], enabled, None]
        result = await self.app.process_api(self.fn_index, inputs, state=state)
        while result["is_generating"]:
            result = await self.app.process_api(self.fn_index, inputs, state=state, iterator=result["iterator"])
            if result["data"] and result["data"][0] is not None:
                last = result["data"]
        return last

    async def test_default_anonymous_dispatch_consumes_evidence(self):
        self.assertTrue(self.fn.inputs[3].value)
        result = await self.dispatch()
        self.assertEqual(result[1], [["current facts", "Grounded"]])
        self.assertIn("https://example.org", result[2])
        system, message = self.webui.llm.calls[-1]
        self.assertEqual(message, "current facts")
        self.assertIn("untrusted source material", system)
        self.assertIn("Current fact", system)
        for request in self.server.requests:
            self.assertEqual(str(request.url), web_search.MCP_URL)
            self.assertEqual(request.headers["user-agent"], web_search.USER_AGENT)
            self.assertNotIn("authorization", request.headers)
        self.assertEqual(self.server.tool_calls()[0]["name"], "web_search")

    async def test_search_fetch_followup_reset_and_isolation(self):
        await self.dispatch()
        first = self.state[self.state_id]
        await self.dispatch("read https://example.org/page", history=[("current facts", "Grounded")])
        await self.dispatch("what changed?", history=[("read https://example.org/page", "Grounded")])
        calls = self.server.tool_calls()
        self.assertEqual([c["name"] for c in calls], ["web_search", "web_fetch", "web_search"])
        self.assertEqual(calls[1]["arguments"]["urls"], ["https://example.org/page"])
        self.assertIn("Previous question:", calls[2]["arguments"]["objective"])
        self.assertEqual({c["arguments"]["session_id"] for c in calls}, {first})
        other = SessionState(self.app)
        await asyncio.gather(self.dispatch("second chat", state=other), self.dispatch("first chat again"))
        self.assertNotEqual(first, other[self.state_id])
        self.assertEqual(self.state[self.state_id], first)
        interleaved = {c["arguments"]["objective"]: c["arguments"]["session_id"]
                       for c in self.server.tool_calls()[-2:]}
        self.assertEqual(interleaved, {"second chat": other[self.state_id], "first chat again": first})
        await self.app.process_api(self.clear_index, [], state=self.state)
        self.assertNotEqual(first, self.state[self.state_id])
        await self.dispatch("after reset")
        self.assertEqual(self.server.tool_calls()[-1]["arguments"]["session_id"], self.state[self.state_id])
        before = self.state[self.state_id]
        await self.app.process_api(self.system_index, ["new system"], state=self.state)
        self.assertNotEqual(before, self.state[self.state_id])

    async def test_off_admin_permission_and_direct_reply_keep_choices(self):
        model = self.webui.llm
        result = await self.dispatch(enabled=False)
        self.assertEqual(result[1][-1][1], "Local")
        self.assertEqual(result[2], "")
        identity = self.state[self.state_id]
        with patch.dict(os.environ, {"LINLY_ALLOW_WEB_SEARCH": "off"}):
            await self.dispatch(enabled=True)  # even a forged client setting cannot override policy
        self.assertIs(self.webui.llm, model)
        self.webui.llm = self.webui.llm_class.init_model("直接回复 Direct Reply")
        result = await self.dispatch("echo me")
        self.assertEqual(result[1][-1], ["echo me", "echo me"])
        self.assertEqual(self.server.requests, [])
        self.assertEqual(identity, self.state[self.state_id])

    async def test_off_then_on_preserves_session_and_model(self):
        model = self.webui.llm
        await self.dispatch(enabled=False)
        identity = self.state[self.state_id]
        await self.dispatch(enabled=True)
        self.assertIs(self.webui.llm, model)
        self.assertEqual(identity, self.server.tool_calls()[-1]["arguments"]["session_id"])

    async def test_quota_error_retry_keeps_identity_and_no_model_fallback(self):
        self.server.error = True
        with self.assertRaises(gr.Error):
            await self.dispatch()
        identity = self.state[self.state_id]
        self.assertEqual(self.webui.llm.calls, [])
        self.server.error = False
        await self.dispatch()
        self.assertEqual({c["arguments"]["session_id"] for c in self.server.tool_calls()}, {identity})

    async def test_http_failure_and_missing_tool_do_not_call_model(self):
        self.server.http_error = True
        with self.assertRaises(gr.Error):
            await self.dispatch()
        self.server.http_error = False
        self.server.missing_tool = True
        with self.assertRaises(gr.Error):
            await self.dispatch()
        self.assertEqual(self.webui.llm.calls, [])

    async def test_invalid_or_empty_response_does_not_call_model(self):
        self.server.empty = True
        with self.assertRaises(gr.Error):
            await self.dispatch()
        self.server.empty = False
        self.server.malformed = True
        with patch.object(web_search, "LOOKUP_TIMEOUT", 0.1):
            with self.assertRaises(gr.Error):
                await self.dispatch()
        self.assertEqual(self.webui.llm.calls, [])

    async def test_timeout_and_cancellation_close_transport(self):
        self.server.slow = True
        with patch.object(web_search, "LOOKUP_TIMEOUT", 0.1):
            with self.assertRaises(gr.Error):
                await self.dispatch()
        self.assertTrue(self.server.cancelled)
        self.assertEqual(self.webui.llm.calls, [])
        self.server.cancelled = False
        iterator = self.webui.chat_response("System", "question", [], True, self.state[self.state_id])
        await anext(iterator)  # identity saved before network access
        task = asyncio.create_task(anext(iterator))
        while len(self.server.tool_calls()) < 2:
            await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.server.cancelled)
        self.assertEqual(self.webui.llm.calls, [])
        await iterator.aclose()

    async def test_sse_transport_and_multiple_urls(self):
        self.server.sse = True
        await self.dispatch("read https://example.org/a and https://example.org/b.")
        self.assertEqual(self.server.tool_calls()[0]["arguments"]["urls"],
                         ["https://example.org/a", "https://example.org/b"])

    async def test_credentials_in_urls_rejected(self):
        with self.assertRaises(gr.Error):
            await self.dispatch("https://user:password@example.org")
        self.assertEqual(self.server.requests, [])
        self.assertEqual(self.webui.llm.calls, [])

    async def test_invalid_urls_and_url_limit(self):
        for message in ("https://[invalid", "https://example.org:abc", "https://", " ".join(["https://example.org"] * 6)):
            with self.subTest(message=message):
                with self.assertRaises(gr.Error):
                    await self.dispatch(message)
        self.assertEqual(self.server.requests, [])

    async def test_parentheses_in_url_path(self):
        await self.dispatch("Read https://example.org/wiki/Thing_(name)")
        self.assertEqual(self.server.tool_calls()[0]["arguments"]["urls"],
                         ["https://example.org/wiki/Thing_(name)"])


if __name__ == "__main__":
    unittest.main()
