import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request

IMAGE = os.environ["REPRO_IMAGE"]
MODE = os.environ.get("REPRO_MODE", "fresh")
FIRST_IMAGE = os.environ["REPRO_PUBLISHED"] if MODE == "upgrade" else IMAGE
ARM = os.environ.get("REPRO_ARM", "?")
OUT = os.environ.get("REPRO_OUT", "repro-out")
HF = os.environ.get("REPRO_HF", "/mnt/hf")
PORT = 8000
BASE = f"http://127.0.0.1:{PORT}"
VOLUME = "unsloth-studio"
PW1 = "Init-" + secrets.token_urlsafe(10)
PW2 = "Chg-" + secrets.token_urlsafe(10)
PROJECT_ID = "p12792-dockervol"
CONTENT = "kept-by-the-volume-12792"
os.makedirs(OUT, exist_ok = True)
results = {"arm": ARM, "mode": MODE, "first_image": FIRST_IMAGE, "second_image": IMAGE}


def log(*a):
    print(f"[{ARM}/{MODE}]", *a, flush = True)


def sh(*cmd, check = True):
    r = subprocess.run(cmd, capture_output = True, text = True)
    if check and r.returncode:
        raise SystemExit(f"{cmd}: {r.stderr.strip()[-800:]}")
    return r.stdout.strip()


def http(method, path, body = None, token = None, timeout = 60, raw = False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data = data, method = method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout = timeout) as r:
            payload = r.read()
            return r.status, payload if raw else json.loads(payload or b"null")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:500]


def run_container(name, image):
    sh("docker", "run", "-d", "--name", name, "-p", f"{PORT}:8000",
       "-e", f"UNSLOTH_STUDIO_PASSWORD={PW1}",
       "-v", f"{HF}:/workspace/.cache/huggingface",
       "-v", f"{VOLUME}:/opt/unsloth-studio", image)
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            status, body = http("GET", "/api/health", timeout = 5)
            if status == 200 and isinstance(body, dict) and body.get("status") == "healthy":
                log(f"{name} healthy")
                return
        except Exception:
            pass
        if sh("docker", "inspect", "-f", "{{.State.Running}}", name, check = False) != "true":
            break
        time.sleep(3)
    print(sh("docker", "logs", "--tail", "200", name, check = False))
    raise SystemExit(f"{name} never became healthy")


def login():
    for pw in (PW2, PW1):
        status, body = http("POST", "/api/auth/login", {"username": "unsloth", "password": pw})
        if status != 200:
            continue
        token = body["access_token"]
        if body.get("must_change_password"):
            s, b = http("POST", "/api/auth/change-password",
                        {"current_password": pw, "new_password": PW2}, token = token)
            assert s == 200, (s, b)
            return login()
        return token, pw
    raise SystemExit("login failed with both passwords")


def tool_write(token, session_id, filename):
    prompt = (f"Use the python tool to run exactly this code: "
              f"open('{filename}', 'w').write('{CONTENT}') . Then reply with the word done.")
    for attempt in range(6):
        body = {"messages": [{"role": "user", "content": prompt}], "enable_tools": True,
                "enabled_tools": ["python"], "session_id": session_id, "temperature": 0.2,
                "seed": 3407 + attempt, "max_tokens": 600, "stream": True}
        if attempt >= 3:
            body["permission_mode"] = "full"
        req = urllib.request.Request(BASE + "/v1/chat/completions", data = json.dumps(body).encode(),
                                     method = "POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("X-Unsloth-Events", "1")
        try:
            with urllib.request.urlopen(req, timeout = 300) as r:
                for _ in r:
                    pass
        except Exception as e:
            log("chat attempt", attempt, "error", repr(e))
        files = list_sandbox(token, session_id)
        log(f"tool write attempt {attempt} (permission_mode={body.get('permission_mode', 'default')}): files={files}")
        if filename in files:
            return attempt, body.get("permission_mode", "default")
    raise SystemExit(f"HARNESS: the model never wrote {filename} into {session_id}")


def list_sandbox(token, session_id):
    s, b = http("GET", f"/api/inference/sandbox/{session_id}", token = token)
    if s != 200:
        return []
    return [f if isinstance(f, str) else f.get("name") or f.get("path") for f in b.get("files", [])]


def fetch_file(token, session_id, filename):
    s, b = http("GET", f"/api/inference/sandbox/{session_id}/{filename}", token = token, raw = True)
    return s, (b.decode("utf-8", "replace") if isinstance(b, bytes) else b)


def library_names(token):
    s, b = http("GET", "/api/library", token = token, timeout = 120)
    if s != 200:
        return f"HTTP {s}"
    names = []
    for item in b.get("items", []):
        blob = json.dumps(item)
        if "notes-" in blob:
            names.append(item)
    return names


def snapshot(label, token):
    s, project = http("GET", f"/api/chat/projects/{PROJECT_ID}", token = token)
    root = project.get("rootPath") if isinstance(project, dict) else None
    psid = f"project-{PROJECT_ID}"
    snap = {
        "project_http": s,
        "rootPath": root,
        "sandboxPath": project.get("sandboxPath") if isinstance(project, dict) else None,
        "project_files": list_sandbox(token, psid),
        "project_file_get": fetch_file(token, psid, "notes-project.txt"),
        "chat_files": list_sandbox(token, "chat-12792-adjacent"),
        "chat_file_get": fetch_file(token, "chat-12792-adjacent", "notes-chat.txt"),
        "library_hits": library_names(token),
        "mount_of_rootPath": sh("docker", "exec", label, "sh", "-c",
                                f"df -P '{root}' | tail -1 | awk '{{print $1\" \"$6}}'", check = False) if root else None,
        "ls_rootPath_sandbox": sh("docker", "exec", label, "sh", "-c", f"ls -la '{root}/sandbox' 2>&1", check = False) if root else None,
        "env_projects_home": sh("docker", "exec", label, "sh", "-c", "echo ${UNSLOTH_STUDIO_PROJECTS_HOME:-<unset>}", check = False),
    }
    results[label] = snap
    log(label, json.dumps(snap, indent = 2))
    return snap


def ui(label, token_pw):
    if MODE != "fresh":
        return
    env = dict(os.environ, UI_URL = BASE, UI_PW = token_pw, UI_OUT = os.path.join(OUT, f"{label}"),
               UI_PROJECT = "Docker volume demo")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "pr12792_ui.py")],
                       env = env, capture_output = True, text = True, timeout = 600)
    log("ui", label, r.returncode, r.stdout[-2000:], r.stderr[-2000:])


def main():
    sh("docker", "volume", "create", VOLUME)
    run_container("studio-1", FIRST_IMAGE)
    token, _ = login()
    now = int(time.time() * 1000)
    s, project = http("POST", "/api/chat/projects",
                      {"id": PROJECT_ID, "name": "Docker volume demo", "createdAt": now, "updatedAt": now},
                      token = token)
    assert s == 200, (s, project)
    log("created project", project)
    s, b = http("POST", "/api/inference/load",
                {"model_path": "unsloth/Qwen3.5-2B-GGUF", "gguf_variant": "UD-Q4_K_XL", "is_lora": False,
                 "max_seq_length": 4096, "speculative_type": "off", "n_parallel": 1}, token = token, timeout = 1200)
    log("load", s, str(b)[:300])
    assert s == 200, (s, b)
    results["project_write"] = tool_write(token, f"project-{PROJECT_ID}", "notes-project.txt")
    results["chat_write"] = tool_write(token, "chat-12792-adjacent", "notes-chat.txt")
    before = snapshot("studio-1", token)
    ui("studio-1", PW2)
    log("docker rm -f studio-1 (what an image update does)")
    sh("docker", "rm", "-f", "studio-1")
    run_container("studio-2", IMAGE)
    token, used = login()
    if MODE == "upgrade":
        s, b = http("POST", "/api/inference/load",
                    {"model_path": "unsloth/Qwen3.5-2B-GGUF", "gguf_variant": "UD-Q4_K_XL", "is_lora": False,
                     "max_seq_length": 4096, "speculative_type": "off", "n_parallel": 1}, token = token, timeout = 1200)
        assert s == 200, (s, b)
        results["project_write_after_upgrade"] = tool_write(token, f"project-{PROJECT_ID}", "notes-project-2.txt")
    results["login_after_recreate_with"] = "changed password (stored on volume)" if used == PW2 else "initial password"
    after = snapshot("studio-2", token)
    ui("studio-2", used)
    sh("docker", "rm", "-f", "studio-2", check = False)

    project_kept = "notes-project.txt" in after["project_files"] and after["project_file_get"][0] == 200 \
        and after["project_file_get"][1] == CONTENT
    chat_kept = "notes-chat.txt" in after["chat_files"] and after["chat_file_get"][0] == 200
    results["verdict"] = {
        "project_file_before_rm": "notes-project.txt" in before["project_files"],
        "project_file_after_recreate": project_kept,
        "chat_file_after_recreate": chat_kept,
        "password_kept": used == PW2,
    }
    if MODE == "upgrade":
        results["verdict"]["post_upgrade_file_listed_in_project"] = "notes-project-2.txt" in after["project_files"]
        results["verdict"]["post_upgrade_file_in_library"] = any("notes-project-2" in json.dumps(h) for h in after["library_hits"]) \
            if isinstance(after["library_hits"], list) else after["library_hits"]
        results["verdict"]["rootPath_after_upgrade"] = after["rootPath"]
        results["verdict"]["rootPath_on_volume"] = str(after["mount_of_rootPath"])
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent = 2, default = str)
    print("RESULT", ARM, MODE, json.dumps(results["verdict"]))
    if MODE == "upgrade":
        sys.exit(0)
    if not (chat_kept and used == PW2):
        print("FAIL adjacent: non-project chat file or password lost")
        sys.exit(3)
    if project_kept:
        print(f"PASS project file survived docker rm: {after['rootPath']}")
        sys.exit(0)
    print(f"FAIL project file lost after docker rm: rootPath={after['rootPath']} files={after['project_files']} "
          f"GET={after['project_file_get'][0]}")
    sys.exit(1)


if __name__ == "__main__":
    main()
