import http.cookiejar
import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE = os.environ.get("JL_URL", "http://127.0.0.1:8888")
PW = os.environ["JUPYTER_PASSWORD"]
MARK = "PR13660 my edit"
NB_DIR = "unsloth-notebooks/nb"

jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def xsrf():
    return next((c.value for c in jar if c.name == "_xsrf"), "")


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method)
    r.add_header("X-XSRFToken", xsrf())
    if data:
        r.add_header("Content-Type", "application/json")
    with opener.open(r, timeout=30) as resp:
        return resp.status, json.loads(resp.read() or b"null")


def login():
    opener.open(BASE + "/login", timeout=30).read()
    form = urllib.parse.urlencode({"password": PW, "_xsrf": xsrf()}).encode()
    opener.open(urllib.request.Request(BASE + "/login", data=form, method="POST"), timeout=30).read()


def contents(path):
    return req("GET", "/api/contents/" + urllib.parse.quote(path))[1]


def wait(limit=900):
    end = time.time() + limit
    last = ""
    while time.time() < end:
        try:
            login()
            items = [i for i in contents(NB_DIR)["content"] if i["name"].endswith(".ipynb")]
            if items:
                print(f"jupyter ready, {len(items)} notebooks in {NB_DIR}")
                return
        except Exception as e:
            last = repr(e)
        time.sleep(5)
    sys.exit(f"jupyter never became ready: {last}")


def edit():
    login()
    items = sorted(i["path"] for i in contents(NB_DIR)["content"] if i["name"].endswith(".ipynb"))
    path = items[0]
    nb = contents(path)["content"]
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": MARK})
    status, _ = req("PUT", "/api/contents/" + urllib.parse.quote(path),
                    {"type": "notebook", "format": "json", "content": nb})
    print(f"saved edit to {path}: HTTP {status}")
    print(path)


def check(path):
    login()
    out = {}
    try:
        nb = contents(path)["content"]
        out["edit"] = any(MARK in "".join(c["source"]) for c in nb["cells"])
    except Exception as e:
        out["edit"] = False
        out["edit_err"] = repr(e)
    folder = path.rsplit("/", 1)[0]
    try:
        out["lora_model"] = sorted(i["name"] for i in contents(folder + "/lora_model")["content"])
    except Exception as e:
        out["lora_model"] = []
        out["lora_err"] = repr(e)
    print(json.dumps(out))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "wait":
        wait()
    elif cmd == "edit":
        edit()
    elif cmd == "check":
        check(sys.argv[2])
