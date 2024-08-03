import sys
import torch
import subprocess

pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
version_str="".join([
    f"py3{sys.version_info.minor}_cu",
    torch.version.cuda.replace(".",""),
    f"_pyt{pyt_version_str}"
])
subprocess.run(["pip", "install", "fvcore", "iopath"])
subprocess.run(["pip", "install", f"--no-index", f"--no-cache-dir", f"pytorch3d", f"-f", f"https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html"])