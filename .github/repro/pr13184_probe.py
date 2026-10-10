import importlib.util
import sys
import urllib.request

spec = importlib.util.spec_from_file_location("h", "studio/backend/core/inference/_html_to_md.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

kept = {
    "radix": ('<h3 data-state="closed"><button type="button" aria-expanded="false">What is the right plan for me?'
              '<svg><path/></svg></button></h3><div role="region"><p>Answer text.</p></div>', "### What is the right plan for me?"),
    "bootstrap": ('<h2 class="accordion-header">\n  <button class="accordion-button" type="button">\n    Accordion Item #1\n  </button>\n</h2>'
                  '<div class="accordion-body">Answer text.</div>', "Accordion Item #1"),
    "uswds": ('<h4 class="usa-accordion__heading"><button type="button" class="usa-accordion__button" aria-expanded="true">'
              'First Amendment</button></h4><div class="usa-accordion__content"><p>Answer text.</p></div>', "#### First Amendment"),
    "role_heading": ('<div role="heading" aria-level="3"><button aria-expanded="false">Is it accessible?</button></div><p>Answer text.</p>',
                     "Is it accessible?"),
}
dropped = {
    "tooltip_after_text": "<h4><span>Create Artifacts</span><button><span class='sr-only'>More information</span></button></h4><p>Answer text.</p>",
    "body_button": "<p>Answer text.</p><button>More information</button>",
    "entity_text": "<h4>&#67;&#114;&#101;&#97;&#116;&#101;<button>More information</button></h4><p>Answer text.</p>",
    "hgroup": "<hgroup><h1>Create</h1><h2></h2><button>More information</button></hgroup><p>Answer text.</p>",
}
failed = 0
for name, (html, want) in kept.items():
    out = m.html_to_markdown(html)
    ok = want in out and "Answer text." in out
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} kept/{name}: {out!r}")
for name, html in dropped.items():
    out = m.html_to_markdown(html)
    ok = "More information" not in out and "Answer text." in out
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'} dropped/{name}: {out!r}")
try:
    req = urllib.request.Request("https://cursor.com/pricing", headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"})
    body = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    md = m.html_to_markdown(body, main_content=True, site_links=m.SiteLinks("https://cursor.com/pricing"))
    faq = md.split("Questions & Answers", 1)[-1]
    heads = [l for l in faq.splitlines() if l.startswith("###")]
    print(f"INFO live cursor.com/pricing FAQ headings ({len(heads)}): {heads}")
except Exception as exc:
    print(f"INFO live fetch skipped: {exc!r}")
print(f"RESULT {'FAIL' if failed else 'PASS'} ({failed} failing) on {sys.platform} python {sys.version.split()[0]}")
sys.exit(1 if failed else 0)
