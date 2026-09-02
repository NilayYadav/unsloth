"""PR 10224 repro probe: is Apple unified "free VRAM" memory a new allocation can get?

Identical on both branches. Only studio/backend/utils/hardware/hardware.py differs.
"""

import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "studio", "backend"))

import psutil
import mlx.core as mx

import utils.hardware.hardware as H
from utils.hardware.hardware import DeviceType

# Device detection wants the whole mlx/mlx-lm/mlx-vlm stack, which is orthogonal to
# the memory arithmetic under test. Everything below is the real machine.
H.get_device = lambda: DeviceType.MLX

GB = 1024 ** 3
MODEL_SIZES_GB = (3, 4, 5, 6, 7, 8, 10, 13)
failures = []

dev = mx.device_info()
WORKING_SET = int(dev.get("max_recommended_working_set_size") or 0)


def check(phase, label, ok, detail):
    print("%s  [%s] %s -- %s" % ("PASS" if ok else "FAIL", phase, label, detail))
    if not ok:
        failures.append("[%s] %s" % (phase, label))


def honest_free(vm):
    """What a single new allocation can actually get."""
    free = vm.available / GB
    return min(free, WORKING_SET / GB) if WORKING_SET > 0 else free


def battery(phase, info, vm):
    free = info["free_gb"]
    avail = vm.available / GB
    print("  psutil total=%.2f GB  available=%.2f GB  |  reported free_gb=%.2f GB"
          % (vm.total / GB, avail, free))
    check(phase, "free_gb <= what the OS says is available", free <= avail + 0.25,
          "free=%.2f GB vs available=%.2f GB (overstated by %.2f GB)" % (free, avail, free - avail))
    check(phase, "free_gb <= the Metal working set one allocation can get",
          WORKING_SET <= 0 or free <= WORKING_SET / GB + 0.25,
          "free=%.2f GB vs working set=%.2f GB" % (free, WORKING_SET / GB))

    # studio/frontend/src/features/training/stores/training-method-hardware-policy.ts:
    # lora when modelSizeGb * 1.5 (ctx 4096) <= vram_free_gb, else qlora.
    truth_free = honest_free(vm)
    wrong = []
    print("  training method the picker offers (estimate = size * 1.5, ctx 4096):")
    for size in MODEL_SIZES_GB:
        need = size * 1.5
        got = "lora" if need <= free else "qlora"
        truth = "lora" if need <= truth_free else "qlora"
        if got != truth:
            wrong.append(size)
        print("    %2d GB model needs %5.1f GB -> picker says %-5s / fits: %-5s%s"
              % (size, need, got, truth, "   <-- offers LoRA the machine cannot fit" if got != truth else ""))
    check(phase, "no model size is offered LoRA on memory the machine does not have",
          not wrong, "sizes mis-offered LoRA: %s" % (wrong if wrong else "none"))


print("machine : %s %s macOS %s" % (platform.system(), platform.machine(), platform.mac_ver()[0]))
print("gpu     : %s" % dev.get("device_name"))
print("metal max_recommended_working_set_size: %.2f GB" % (WORKING_SET / GB))
print("")

vm_idle = psutil.virtual_memory()
idle = H.get_gpu_memory_info()
print("=== get_gpu_memory_info() at rest ===")
print(json.dumps(idle, indent=2, sort_keys=True))
battery("idle", idle, vm_idle)

budget = min(2.0 * GB, max(0.0, vm_idle.available * 0.35))
if budget >= 0.5 * GB:
    hog = bytearray(int(budget))
    for off in range(0, len(hog), 4096):
        hog[off] = 1
    vm_load = psutil.virtual_memory()
    load = H.get_gpu_memory_info()
    print("")
    print("=== the same machine after another process takes %.2f GB of host RAM ==="
          % (len(hog) / GB))
    moved_free = idle["free_gb"] - load["free_gb"]
    moved_avail = (vm_idle.available - vm_load.available) / GB
    battery("under load", load, vm_load)
    check("under load", "free_gb tracks the memory the host actually gave away",
          abs(moved_free - moved_avail) <= 0.25,
          "available moved %.2f GB, reported free moved %.2f GB" % (moved_avail, moved_free))
    del hog
else:
    print("SKIP  host-allocation phase -- only %.2f GB available" % (vm_idle.available / GB))

print("")
if failures:
    print("REPRO RESULT: DEFECT PRESENT -- %d assertion(s) failed" % len(failures))
    for f in failures:
        print("  - %s" % f)
    sys.exit(1)
print("REPRO RESULT: CLEAN -- reported free memory is what a new allocation can actually get")
