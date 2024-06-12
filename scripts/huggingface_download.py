# pip install huggingface_hub
# https://huggingface.co/Kedreamix/Linly-Talker
from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="Kedreamix/Linly-Talker",
  resume_download=True,
  local_dir="Kedreamix/Linly-Talker",
  local_dir_use_symlinks=False,
#   proxies={"https": "http://localhost:7890"}
)