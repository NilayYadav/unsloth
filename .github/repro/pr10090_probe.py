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

# The in-band surface: a 200 stream that later emits data: {"error": ...}. Before the
# in-band fix the route wrapped the body in RuntimeError and let _friendly_error render it.
in_band_fn = getattr(ri, "_anthropic_upstream_stream_error_event", None)
in_band_by = "_anthropic_upstream_stream_error_event"
if in_band_fn is None:
    in_band_by = "_anthropic_stream_error_event(RuntimeError(...))"

    def in_band_fn(text):
        return ri._anthropic_stream_error_event(RuntimeError(text), force=True)


def in_band(text):
    event = in_band_fn(text)
    payload = json.loads(event.split("data: ", 1)[1].strip().splitlines()[0])["error"]
    return {"message": payload["message"], "type": payload["type"]}


out = {"rendered_by": rendered_by, "in_band_by": in_band_by, "cases": {}, "in_band": {}}
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

NOCOUNT = "the request exceeds the available context size. try increasing the context size"
for name, body in (("oversize_no_counts", NOCOUNT), ("oversize", OVERSIZE), ("starvation", STARVATION)):
    out["in_band"][name] = dict(in_band(body), upstream_body=body)

print(json.dumps(out, indent=2))
with open("probe_out.json", "w") as fh:
    json.dump(out, fh)
