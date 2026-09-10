"""Launch the checked-out Studio app and its own built frontend."""
import os
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
home = Path(sys.argv[2]).resolve()
port = int(sys.argv[3])
home.mkdir(parents=True, exist_ok=True)
os.environ.update({
    "UNSLOTH_STUDIO_HOME": str(home),
    "HF_HOME": str(home / "hf"),
    "HF_HUB_CACHE": str(home / "hf" / "hub"),
    "HF_XET_CACHE": str(home / "hf" / "xet"),
    "XDG_CACHE_HOME": str(home / "cache"),
    "UNSLOTH_ALLOW_CPU": "1",
    "UNSLOTH_IS_PRESENT": "1",
    "UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE": "1",
    "UNSLOTH_DISABLE_MLX_AUTOREPAIR": "1",
    "UNSLOTH_COMPILE_LOCATION": str(home / "compiled_cache"),
})
sys.path.insert(0, str(repo / "studio" / "backend"))
from main import app, setup_frontend
import uvicorn

assert setup_frontend(app, repo / "studio" / "frontend" / "dist")
uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
