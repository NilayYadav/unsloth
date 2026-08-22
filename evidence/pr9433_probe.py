"""Focused repro probe for unslothai/unsloth#9433.

Prints the real classifications the picker and the loader resolve for three GGUF folders, then
asserts them. Runs identically on the PR head and on a branch with the fix reverted; only the
implementation under test differs.
"""

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "studio/backend")


def arch_gguf(path: Path, architecture: str) -> None:
    """A minimal GGUF carrying only general.architecture."""

    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
        + string("general.architecture")
        + struct.pack("<I", 8)
        + string(architecture)
    )


def folder(name: str, files):
    root = Path(tempfile.mkdtemp()) / name
    for filename, architecture in files:
        arch_gguf(root / filename, architecture)
    return root


def main() -> int:
    import routes.models as models_route
    from core.inference.diffusion_families import detect_family_for_pick

    failures = []

    def check(label, got, want):
        ok = got == want
        print(f"{'PASS' if ok else 'FAIL'} {label}: got={got!r} want={want!r}", flush=True)
        if not ok:
            failures.append(label)

    # The regression: a speech GGUF beside a runnable chat one must not tag the folder speech,
    # or the frontend arch-task gate hides the row and the runnable qwen3 with it.
    mixed_chat = folder("mixed-chat", [("csm-1b-Q4_0.gguf", "llama-csm"),
                                       ("qwen3-8b-Q4_K_M.gguf", "llama")])
    check("mixed csm+qwen3 folder task",
          models_route._gguf_folder_task(mixed_chat, ("someone/mixed-GGUF",)), "text-generation")

    # Controls: the two rankings the fix must not disturb.
    speech_only = folder("speech-only", [("csm-1b-Q4_0.gguf", "llama-csm")])
    check("csm-only folder task",
          models_route._gguf_folder_task(speech_only, ("someone/csm-GGUF",)), "text-to-speech")

    mixed_media = folder("mixed-media", [("csm-1b-Q4_0.gguf", "llama-csm"),
                                         ("flux1-dev-Q4_K_M.gguf", "flux")])
    check("mixed csm+flux folder task",
          models_route._gguf_folder_task(mixed_media, ("someone/flux-GGUF",)), "text-to-image")

    # Why the pick is dangerous at all: the family is resolved from the FOLDER name, so a csm
    # file beside a denoiser answers flux.1 and reaches the image loader.
    fam = detect_family_for_pick("/models/flux1-dev-GGUF", "csm-1b-Q4_0.gguf")
    fam_name = getattr(fam, "name", None)
    print(f"INFO  detect_family_for_pick(csm file in flux folder) = {fam_name!r}", flush=True)

    # The preflight that refuses that pick before the download and the eviction. Absent entirely
    # before the fix, so report that as a failed check rather than crashing: the negative run has
    # to fail at the assertion, not during setup.
    from core.inference import diffusion_compat

    if not hasattr(diffusion_compat, "speech_pick_refusal"):
        print("FAIL speech pick refused before download: "
              "diffusion_compat.speech_pick_refusal is absent", flush=True)
        print("FAIL runnable sibling still allowed: preflight absent", flush=True)
        failures.extend(["speech pick refused before download", "runnable sibling still allowed"])
        print(f"\n{'REPRO FAILED' if failures else 'REPRO PASSED'}: "
              f"{len(failures)} failing check(s)", flush=True)
        print("failing: " + ", ".join(failures), flush=True)
        return 1

    # Resolve each pick to its OWN file in the mixed folder, as the real cache lookup would;
    # a stub that returns one path for every filename would judge the denoiser by csm bytes.
    diffusion_compat._local_gguf_path = lambda repo_id, filename, *a, **k: str(
        mixed_media / filename
    )
    diffusion_compat._reset_inner_dim_cache()
    refusal = diffusion_compat.speech_pick_refusal("someone/flux-GGUF", "csm-1b-Q4_0.gguf")
    print(f"INFO  speech_pick_refusal = {refusal!r}", flush=True)
    check("speech pick refused before download", refusal is not None, True)

    diffusion_compat._reset_inner_dim_cache()
    allowed = diffusion_compat.speech_pick_refusal("someone/flux-GGUF", "flux1-dev-Q4_K_M.gguf")
    check("runnable sibling still allowed", allowed, None)

    print(f"\n{'REPRO FAILED' if failures else 'REPRO PASSED'}: {len(failures)} failing check(s)",
          flush=True)
    if failures:
        print("failing: " + ", ".join(failures), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
