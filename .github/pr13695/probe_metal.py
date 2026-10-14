import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))

out = {"label": sys.argv[1], "mac_ver": platform.mac_ver()[0], "machine": platform.machine()}
try:
    import mlx.core as mx
    out["mlx_ws"] = int(mx.device_info().get("max_recommended_working_set_size") or 0)
except Exception as e:
    out["mlx_ws"] = f"unavailable: {type(e).__name__}: {e}"
try:
    import torch
    out["torch"] = torch.__version__
    out["mps_available"] = bool(torch.backends.mps.is_available())
    out["torch_ws"] = int(torch.mps.recommended_max_memory()) if out["mps_available"] else 0
except Exception as e:
    out["torch_ws"] = f"unavailable: {type(e).__name__}: {e}"
import psutil
vm = psutil.virtual_memory()
out["total"] = vm.total
out["available"] = vm.available
from core.inference.llama_cpp import LlamaCppBackend
from utils.hardware import hardware
out["helper_present"] = hasattr(hardware, "_metal_recommended_working_set_bytes")
if out["helper_present"]:
    out["helper_ws"] = hardware._metal_recommended_working_set_bytes()
out["gpu_stats"] = hardware._read_apple_gpu_stats()
out["budget"] = LlamaCppBackend._apple_metal_memory_budget_bytes()
out["wired_ceiling"] = LlamaCppBackend._apple_metal_wired_ceiling_bytes()
out["fit_floor_mib"] = LlamaCppBackend._metal_fit_target_floor_mib()
print("PROBE " + json.dumps(out, default=str))
