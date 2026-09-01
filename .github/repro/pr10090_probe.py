"""PR 10090 probe: what the Anthropic passthrough tells the client for two upstream bodies.

Identical across variants. Only studio/backend/routes/inference.py is swapped, so any
difference in the printed values is caused by that file alone.
"""
import json
import sys

sys.path.insert(0, "studio/backend")

import routes.inference as ri  # noqa: E402

OVERSIZE = "the request (214331 tokens) exceeds the available context size (131072 tokens)"
STARVATION = "Context size has been exceeded."

# The PR introduces _anthropic_upstream_error; before it, the Anthropic passthrough
# used _friendly_upstream_error. Ask for whichever this checkout actually ships.
render = getattr(ri, "_anthropic_upstream_error", None)
rendered_by = "_anthropic_upstream_error"
if render is None:
    render = ri._friendly_upstream_error
    rendered_by = "_friendly_upstream_error"

out = {"rendered_by": rendered_by, "cases": {}}
for name, body in (("oversize", OVERSIZE), ("starvation", STARVATION)):
    cls = ri._classify_llama_generation_error(Exception(body))
    out["cases"][name] = {
        "upstream_body": body,
        "message": render(body),
        "classify": cls,
        # The formula both changed call sites use. Pre-PR checkouts sent the
        # upstream status instead, so this column only binds on the head variants.
        "status_under_pr_formula": 400 if bool(cls) else 500,
    }

print(json.dumps(out, indent=2))
with open("probe_out.json", "w") as fh:
    json.dump(out, fh)
