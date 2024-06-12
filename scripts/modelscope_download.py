# pip install modelscope
# https://www.modelscope.cn/models/Kedreamix/Linly-Talker/summary
from modelscope import snapshot_download
model_dir = snapshot_download('Kedreamix/Linly-Talker', cache_dir='./', revision='master')