"""A/B the GLM-5.3 chat template against llama.cpp's own capability probe.

BEFORE is the template exactly as unsloth/GLM-5.3-GGUF embeds it, which is what
llama-server saw before this PR. AFTER is the same template through Studio's
repair. Only the template changes; probe, source build and runner are identical.
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "studio" / "backend"))

from core.inference.chat_template_helpers import repair_numeric_member_access

PROBE = sys.argv[1]
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

GATED = ("supports_tools", "supports_tool_calls", "supports_parallel_tool_calls",
         "supports_object_arguments")


def embedded_template(repo):
    url = f"https://huggingface.co/api/models/{repo}?expand%5B%5D=gguf"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)["gguf"]["chat_template"]
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"  retrying {repo} after {exc}")
            time.sleep(5 * (attempt + 1))


def caps(name, text):
    path = OUT / f"{name}.jinja"
    path.write_text(text, encoding="utf-8")
    proc = subprocess.run([PROBE, str(path)], capture_output=True, text=True)
    print(f"--- {name} ({len(text)} chars) ---")
    print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr.rstrip())
    out = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("supports_") and "=" in line:
            key, value = line.split("=", 1)
            out[key] = value == "true"
    return out


failures = []


def expect(label, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {got} (expected {want})")
    if not ok:
        failures.append(label)


print("== BEFORE: unsloth/GLM-5.3-GGUF template as shipped ==")
raw = embedded_template("unsloth/GLM-5.3-GGUF")
before = caps("glm53-before", raw)
sites = raw.count(".0.")
print(f"  numeric member-access sites in the shipped template: {sites}")
expect("shipped template has numeric member access", sites > 0, True)
for cap in GATED:
    expect(f"BEFORE {cap}", before.get(cap), False)

print()
print("== AFTER: same template through repair_numeric_member_access ==")
fixed = repair_numeric_member_access(raw)
expect("repair rewrote the template", fixed is not None, True)
if fixed:
    expect("no numeric member access left", ".0." in fixed, False)
    after = caps("glm53-after", fixed)
    for cap in GATED:
        expect(f"AFTER {cap}", after.get(cap), True)
    expect("repair is idempotent", repair_numeric_member_access(fixed), None)

print()
print("== CONTROL: unsloth/GLM-4.7-GGUF, a template with no numeric member access ==")
control = embedded_template("unsloth/GLM-4.7-GGUF")
expect("control template left alone", repair_numeric_member_access(control), None)
control_caps = caps("glm47-control", control)
for cap in GATED:
    expect(f"CONTROL {cap}", control_caps.get(cap), True)

print()
print("== FALSE-POSITIVE GUARD: templates the repair must not touch ==")
expect("raw block emitted verbatim",
       repair_numeric_member_access("{% raw %}{{ example.0 }}{% endraw %}"), None)
expect("raw block, whitespace-control spelling",
       repair_numeric_member_access("{%- raw -%}{{ example.0 }}{%- endraw -%}"), None)
expect("a real site beside a raw block still repaired",
       repair_numeric_member_access("{% raw %}{{ example.0 }}{% endraw %}{{ m.content.0.output }}"),
       "{% raw %}{{ example.0 }}{% endraw %}{{ m.content[0].output }}")
expect("quoted prompt text untouched",
       repair_numeric_member_access('{{ "see step.0 below" }}'), None)
expect("float literal untouched",
       repair_numeric_member_access("{% if t > 0.5 %}{{ 1.0 }}{% endif %}"), None)
expect("whole numeric chain rewritten",
       repair_numeric_member_access("{{ matrix.0.1 }}"), "{{ matrix[0][1] }}")
expect("spaced dot rewritten",
       repair_numeric_member_access("{{ messages[i] . 0.role }}"), "{{ messages[i][0].role }}")
expect("printed Jinja example untouched",
       repair_numeric_member_access('{{ "{{ example.0 }}" }}'), None)
expect("a literal brace does not hide a later real site",
       repair_numeric_member_access('{{ "a }} b" }}{{ m.content.0.type }}'),
       '{{ "a }} b" }}{{ m.content[0].type }}')
expect("comment not worth a relaunch",
       repair_numeric_member_access("{# {{ a.0 }} #}"), None)

print()
for repo in ("unsloth/gpt-oss-120b-GGUF", "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF",
             "unsloth/DeepSeek-V3.2-GGUF", "unsloth/gemma-4-31b-it-GGUF"):
    expect(f"{repo} left alone", repair_numeric_member_access(embedded_template(repo)), None)

print()
if failures:
    print(f"FAILED {len(failures)} assertion(s): {failures}")
    sys.exit(1)
print("ALL ASSERTIONS PASSED")
