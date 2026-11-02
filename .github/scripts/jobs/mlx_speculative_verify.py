# SPDX-License-Identifier: AGPL-3.0-only
"""Validate the MTP runtime candidate on Metal before profiling it."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required = True)
    parser.add_argument("--revision", default = "9b1a6e9b062ad42e69e0aedb7afa6cf4c6c01c45")
    args = parser.parse_args()
    output = Path(args.out).resolve()
    result = {"finished": False, "passed": False, "revision": args.revision}
    scratch = Path(os.environ.get("RUNNER_TEMP", "temp"))
    scratch.mkdir(parents = True, exist_ok = True)
    try:
        os.environ["UNSLOTH_IS_PRESENT"] = "1"
        os.environ.pop("UNSLOTH_FORCE_GPU_PATH", None)
        import mlx.core as mx

        assert mx.metal.is_available(), "Metal unavailable"
        mx.set_default_device(mx.gpu)
        result.update(
            device = str(mx.default_device()),
            versions = {
                name: importlib.metadata.version(name)
                for name in ("mlx", "mlx-lm", "mlx-vlm", "unsloth_zoo")
            },
        )
        with tempfile.TemporaryDirectory(prefix = "mtp-verify-", dir = scratch) as folder:
            root = Path(folder)
            repo = root / "zoo"
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "--branch",
                    "fix-mlx-draft-cost",
                    "https://github.com/danielhanchen/unsloth-zoo-staging.git",
                    str(repo),
                ],
                check = True,
            )
            head = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text = True
            ).strip()
            assert head == args.revision, (head, args.revision)
            report = root / "tests.xml"
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                f"--junitxml={report}",
                "tests/test_mlx_speculative.py",
                "tests/test_mlx_speculative_metal.py::test_mtp_rounds_match_solo_decoding",
                "tests/test_mlx_speculative_metal.py::test_mtp_catches_up_after_plain_windows_and_copies",
                "tests/test_mlx_speculative_metal.py::test_an_mtp_heads_cache_holds_only_target_hidden_pairs_however_its_drafts_fared",
            ]
            env = dict(
                os.environ,
                PYTHONPATH = str(repo),
                HF_DOWNLOAD_OK = "1",
                UNSLOTH_DISABLE_AUTO_UPDATES = "1",
            )
            proc = subprocess.run(command, cwd = repo, env = env, check = False)
            cases = ET.parse(report).findall(".//testcase")
            result["tests"] = [
                {
                    "id": f"{case.get('classname')}::{case.get('name')}",
                    "outcome": "failed"
                    if case.find("failure") is not None or case.find("error") is not None
                    else "skipped"
                    if case.find("skipped") is not None
                    else "passed",
                }
                for case in cases
            ]
            result["passed"] = (
                proc.returncode == 0
                and bool(cases)
                and all(case["outcome"] == "passed" for case in result["tests"])
            )
            result["pytest_exit"] = proc.returncode
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["finished"] = True
        output.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
