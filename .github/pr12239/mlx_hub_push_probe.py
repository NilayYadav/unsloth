import json
import os
import shutil
import sys
from pathlib import Path

WORK = Path(os.environ["PROBE_WORK"]).resolve()
os.environ["UNSLOTH_STUDIO_HOME"] = str(WORK / "studio")
sys.path.insert(0, str(Path.cwd()))

import huggingface_hub
from huggingface_hub import HfApi
from huggingface_hub.utils import filter_repo_objects

hub = {"pr12239/vis": {"private": True, "files": set()}}
log = []


def _files(folder, allow = None, ignore = None):
    root = Path(folder)
    rel = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
    return sorted(filter_repo_objects(rel, allow_patterns = allow, ignore_patterns = ignore))


def create_repo(self, repo_id, *a, private = None, exist_ok = False, **k):
    repo_id = repo_id if "/" in repo_id else f"pr12239/{repo_id}"
    log.append(("create_repo", repo_id, private))
    hub.setdefault(repo_id, {"private": bool(private), "files": set()})
    return huggingface_hub.RepoUrl(f"https://huggingface.co/{repo_id}")


def update_repo_settings(self, repo_id, *a, private = None, **k):
    log.append(("update_repo_settings", repo_id, private))
    if private is not None:
        hub[repo_id]["private"] = bool(private)


def repo_info(self, repo_id, *a, **k):
    return type("Info", (), {"private": hub[repo_id]["private"]})()


def file_exists(self, repo_id, filename, *a, **k):
    return filename in hub.get(repo_id, {}).get("files", set())


def upload_folder(self, *, folder_path, repo_id, allow_patterns = None, ignore_patterns = None, **k):
    sent = _files(folder_path, allow_patterns, ignore_patterns)
    log.append(("upload_folder", repo_id, sent))
    hub[repo_id]["files"].update(sent)


def upload_large_folder(self, repo_id, folder_path, *a, allow_patterns = None, ignore_patterns = None, **k):
    sent = _files(folder_path, allow_patterns, ignore_patterns)
    log.append(("upload_large_folder", repo_id, sent))
    hub[repo_id]["files"].update(sent)


from unsloth_zoo.mlx.loader import FastMLXModel

ckpt = WORK / "studio" / "outputs" / "pr12239-lora"
if not (ckpt / "adapter_config.json").exists():
    model, tokenizer = FastMLXModel.from_pretrained(
        "unsloth/gemma-3-270m-it", load_in_4bit = False, dtype = "float16", text_only = True, max_seq_length = 128
    )
    model = FastMLXModel.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = ["q_proj", "v_proj"])
    model.save_pretrained_merged(str(ckpt), tokenizer = tokenizer, save_method = "lora")
    del model

from core.export.export import ExportBackend

backend = ExportBackend()
ok, msg = backend.load_checkpoint(str(ckpt), load_in_4bit = False)
assert ok, msg
print("loaded:", msg, "is_peft:", backend.is_peft)

def install_fake_hub():
  for name, fn in dict(
    create_repo = create_repo,
    update_repo_settings = update_repo_settings,
    repo_info = repo_info,
    file_exists = file_exists,
    upload_folder = upload_folder,
    upload_large_folder = upload_large_folder,
).items():
    setattr(HfApi, name, fn)
  huggingface_hub.ModelCard.push_to_hub = lambda self, repo_id, **k: log.append(("card", repo_id))


install_fake_hub()
results = {}


def run(name, save_dir, repo, private, prefill = None):
    target = WORK / "studio" / "exports" / save_dir
    if prefill:
        target.mkdir(parents = True, exist_ok = True)
        for f in prefill:
            shutil.copy(ckpt / f, target / f)
    start = len(log)
    ok, msg, out = backend.export_merged_model(save_dir, push_to_hub = True, repo_id = repo, hf_token = "hf_probe", private = private)
    calls = log[start:]
    uploads = [c for c in calls if c[0].startswith("upload")]
    sent = sorted({f for c in uploads for f in c[2]})
    repo_full = repo if "/" in repo else f"pr12239/{repo}"
    results[name] = {
        "ok": ok,
        "msg": msg,
        "local_files": sorted(p.name for p in Path(out).iterdir()) if out else None,
        "calls": [c[:2] + ((c[2],) if c[0] == "update_repo_settings" else ()) for c in calls],
        "upload_kind": [c[0] for c in uploads],
        "uploaded": sent,
        "repo_private_after": hub[repo_full]["private"],
    }
    print(f"== {name}")
    print(json.dumps(results[name], indent = 1))
    return results[name]


fails = []
r = run("fresh_folder", "pr12239-fresh", "pr12239/fresh", True)
if not r["ok"] or "export_metadata.json" in r["uploaded"]:
    fails.append("fresh_folder: export_metadata.json uploaded" if r["ok"] else f"fresh_folder: {r['msg']}")
if r["ok"] and not any(f.endswith(".safetensors") for f in r["uploaded"]):
    fails.append("fresh_folder: no weights uploaded")

r = run("reused_folder", "pr12239-reused", "pr12239/reused", True, prefill = ["adapter_config.json", "adapters.safetensors"])
leaked = [f for f in ("adapter_config.json", "adapters.safetensors", "export_metadata.json") if f in r["uploaded"]]
if not r["ok"] or leaked:
    fails.append(f"reused_folder: leftovers uploaded {leaked}" if r["ok"] else f"reused_folder: {r['msg']}")

r = run("existing_private_repo_unticked", "pr12239-vis", "pr12239/vis", False)
if not r["ok"] or not r["repo_private_after"]:
    fails.append("existing_private_repo_unticked: repo made public" if r["ok"] else f"vis: {r['msg']}")

print("SUMMARY", json.dumps({k: {"uploaded": v["uploaded"], "private_after": v["repo_private_after"]} for k, v in results.items()}))
for f in fails:
    print("FAIL", f)
if fails:
    sys.exit(1)
print("PASS all three scenarios")
