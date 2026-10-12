import importlib.util
import sys

spec = importlib.util.spec_from_file_location("h", "studio/backend/core/inference/_html_to_md.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

failures = 0
for skipped in ("nav", "footer", "svg", "button"):
    for main_content in (False, True):
        html = (
            f"<body><div><{skipped}><div><a href=/x>Menu</a></div></div>"
            "<main><h1>Title</h1><p>Real text of the page that must reach the model.</p></main></body>"
        )
        out = m.html_to_markdown(html, main_content=main_content)
        ok = "Menu" not in out and "Real text of the page" in out
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {skipped} main_content={main_content} out={out!r}")

go_like = (
    '<aside class="NavigationDrawer"><nav class="NavigationDrawer-nav"><ul><li>'
    '<div class="NavigationDrawer-submenuItem"><nav class="NavigationDrawer-nav"><a href="#">Why Go</a>'
    "</div></li></ul></nav></aside>"
    '<main id="main-content"><article><h1>Download and install</h1>'
    "<p>Download and install Go quickly with the steps described here.</p></article></main>"
)
out = m.html_to_markdown(go_like, main_content=True)
ok = "Download and install Go quickly" in out and "Why Go" not in out
failures += not ok
print(f"{'PASS' if ok else 'FAIL'} go.dev drawer out={out!r}")

normal = "<body><nav><a>Home</a></nav><main><h1>T</h1><p>Body text.</p></main><footer>F</footer></body>"
out = m.html_to_markdown(normal, main_content=False)
ok = out == "# T\n\nBody text."
failures += not ok
print(f"{'PASS' if ok else 'FAIL'} normal nav/footer out={out!r}")
sys.exit(1 if failures else 0)
