import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

DDG = "https://html.duckduckgo.com/html/?q="
SEARCH = ["llama.cpp", "node.js", "package.json", "index.html", "model.safetensors",
          "Qwen3-8B.Q4_K_M.gguf", "LLAMA.CPP", "config.yaml", "notes.txt"]
HOST = {"unsloth.ai": "https://unsloth.ai", "docs.rs/serde": "https://docs.rs/serde",
        "docs.python.org/3": "https://docs.python.org/3",
        "github.com/ggml-org/llama.cpp": "https://github.com/ggml-org/llama.cpp",
        "huggingface.co/unsloth/x.gguf": "https://huggingface.co/unsloth/x.gguf",
        "example.com/index.html": "https://example.com/index.html", "main.py": "https://main.py",
        "readme.md": "https://readme.md", "build.ts:8443": "https://build.ts:8443",
        "localhost:8888": "http://localhost:8888",
        "192.168.1.10:8080": "https://192.168.1.10:8080",
        "пример.рф": "https://пример.рф"}

script = Path(sys.argv[1]).read_text(encoding="utf-8")
targets = sys.argv[2].split(",")
failed = 0
with sync_playwright() as p:
    for target in targets:
        kind, _, channel = target.partition(":")
        browser = getattr(p, kind).launch(**({"channel": channel} if channel else {}))
        page = browser.new_page()
        page.add_script_tag(content=script)
        inputs = SEARCH + list(HOST)
        got = page.evaluate("xs => xs.map(x => globalThis.__resolveAddress(x, 'duckduckgo'))", inputs)
        ua = page.evaluate("navigator.userAgent")
        browser.close()
        bad = []
        for text, out in zip(inputs, got):
            want = HOST.get(text) or DDG + __import__("urllib.parse").parse.quote(text, safe="-_.!~*'()")
            if out != want:
                bad.append(f"{text!r} -> {out!r} (want {want!r})")
        print(f"{target}: {len(inputs) - len(bad)}/{len(inputs)} as expected  [{ua}]")
        for line in bad:
            print(f"  FAIL {line}")
        failed += len(bad)
print("PASS" if not failed else f"FAIL {failed} mismatches")
sys.exit(1 if failed else 0)
