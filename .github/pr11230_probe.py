# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

SEED_TOKEN = "hf_PROBE_SEED_TOKEN_0123456789"

uploaded = {}


class _UploadError(Exception):
    pass


class _Client:
    def __init__(self, token):
        pass

    def __getattr__(self, name):
        return lambda **kwargs: None

    def _upload_config_files(self, *, repo_id, metadata_path, builder_config_path):
        uploaded["builder_config"] = (
            builder_config_path.read_text(encoding = "utf-8")
            if builder_config_path.exists()
            else None
        )
        uploaded["metadata"] = metadata_path.read_text(encoding = "utf-8")


class _Card:
    @classmethod
    def from_metadata(cls, *, builder_config, **kwargs):
        uploaded["card"] = json.dumps(builder_config)
        return SimpleNamespace(text = "", push_to_hub = lambda *a, **k: None)


def _install_stubs():
    modules = {
        "data_designer.engine.storage.artifact_storage": {
            "FINAL_DATASET_FOLDER_NAME": "parquet-files",
            "METADATA_FILENAME": "metadata.json",
            "PROCESSORS_OUTPUTS_FOLDER_NAME": "processors-files",
            "SDG_CONFIG_FILENAME": "builder_config.json",
        },
        "data_designer.integrations.huggingface.client": {
            "HuggingFaceHubClient": _Client,
            "HuggingFaceHubClientUploadError": _UploadError,
        },
        "data_designer.integrations.huggingface.dataset_card": {
            "DataDesignerDatasetCard": _Card,
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


def main() -> int:
    _install_stubs()
    from core.data_recipe import huggingface as recipe_hf

    builder_config = {
        "data_designer": {
            "columns": [{"name": "answer", "column_type": "llm-text"}],
            "seed_config": {
                "source": {
                    "seed_type": "hf",
                    "path": "datasets/acme/private-seed/data/*.parquet",
                    "token": SEED_TOKEN,
                },
                "sampling_strategy": "ordered",
            },
        }
    }

    with tempfile.TemporaryDirectory() as raw:
        artifact = Path(raw)
        (artifact / "metadata.json").write_text("{}", encoding = "utf-8")
        on_disk = artifact / "builder_config.json"
        on_disk.write_text(json.dumps(builder_config), encoding = "utf-8")

        recipe_hf._resolve_recipe_artifact_path = lambda _: artifact
        recipe_hf.publish_recipe_dataset(
            artifact_path = str(artifact),
            repo_id = "acme/published-recipe",
            description = "probe",
            hf_token = "hf_publish_credential",
        )

        disk_after = on_disk.read_text(encoding = "utf-8")

    print("=== what the Hub repo would receive ===")
    print(uploaded["builder_config"])
    print("=== dataset card builder_config ===")
    print(uploaded["card"])

    leaked_file = SEED_TOKEN in (uploaded["builder_config"] or "")
    leaked_card = SEED_TOKEN in (uploaded["card"] or "")
    disk_intact = SEED_TOKEN in disk_after

    print(f"seed token in uploaded builder_config.json : {leaked_file}")
    print(f"seed token in uploaded dataset card        : {leaked_card}")
    print(f"seed token still on local disk (must be True): {disk_intact}")

    if leaked_file or leaked_card:
        print("REPRO: FAIL - the seed token reaches the Hugging Face dataset repo")
        return 1
    if not disk_intact:
        print("REGRESSION: the local builder_config.json was mutated")
        return 1
    print("REPRO: PASS - no seed token leaves the machine, local file untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
