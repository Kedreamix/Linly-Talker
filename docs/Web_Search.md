# Web lookup in multi-turn chat

The **数字人多轮智能对话** tab in `webui.py` can use current web evidence with
the selected LLM. **联网查询 / Web lookup (Parallel)** is enabled by default.
It uses [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
at `https://search.parallel.ai/mcp`, with Fast mode search and no account or API
key. The free service has usage limits.

1. Install the WebUI dependencies as usual. For an existing installation on
   Python 3.10 or newer, add the small lookup dependency:

   ```bash
   python -m pip install -r requirements_search.txt
   ```

2. Start `python webui.py`, open **数字人多轮智能对话**, and select a chat-capable
   model such as Qwen or Qwen2. Direct Reply stays a text echo and never performs
   a web lookup. LLM model downloads and credentials remain as before.
3. Submit a question to search the web, or include an HTTP/HTTPS URL to read that
   page. The lookup runs before the selected model answers. Open **网页参考 / Web
   evidence** to inspect the excerpts and source URLs used for the answer.

When enabled, the question is sent to Parallel. For search follow-ups, the previous
question is also included to provide context. A message containing URLs fetches
those pages instead of running a search, with a maximum of five URLs per message.
Use public URLs; embedded login credentials are rejected. Web content is
provided to the model as untrusted evidence; check the sources for important facts.

Uncheck **联网查询 / Web lookup (Parallel)** to chat without sending questions
to the search service. The setting stays with the current browser session.
Administrators can disable lookup for every session before starting the WebUI:

```bash
# Linux/macOS
LINLY_ALLOW_WEB_SEARCH=0 python webui.py
```

Clearing history or changing the system prompt starts a new lookup conversation.
Switching lookup off and back on retains the current conversation. A failed
lookup leaves the question in place and displays an error; retry it or turn
lookup off to use the model alone. There is no automatic fallback to another
search provider. Lookup has a 20-second deadline; **停止 / Stop** can cancel it.
Stopping during model inference does not interrupt a backend's running generation.

This feature applies to multi-turn WebUI chat. The API and other WebUI tabs retain
their existing behavior.

## Development checks

The lookup tests use the real MCP transport and Gradio dispatcher with an in-memory
HTTP server response and a scripted LLM backend. They do not download model weights
or require a GPU. In a fresh Python 3.10+ environment:

```bash
python -m pip install -r requirements_search.txt 'gradio==4.44.1' 'huggingface_hub==0.25.2'
python -m unittest discover -s tests -p 'test_web_search.py' -v
```
