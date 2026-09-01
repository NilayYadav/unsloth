"""Turn what the frontend sends into what the backend gives the document.

`_whole_doc_budget` is the real function the RAG path calls; nothing here
reimplements it.
"""

import json
import sys

sys.path.insert(0, "studio/backend")
from core.inference import tools  # noqa: E402

CONVERSATION = [
    {"role": "user", "content": "Summarise the attached document."},
]

rev = sys.argv[1]
probe = json.load(open(sys.argv[2]))

print(f"--- {rev} ---")
out = {}
for row in probe["results"]:
    context = row["context_length"]
    scope = {"context_length": context} if context is not None else {}
    budget = tools._whole_doc_budget(scope, CONVERSATION)
    out[row["name"]] = {"context_length": context, "whole_doc_budget": budget}
    sent = "no window" if context is None else f"{context} tokens"
    print(f"  {row['name']}: sends {sent} -> {budget} tokens of document")

json.dump(out, open(sys.argv[3], "w"), indent=2)
