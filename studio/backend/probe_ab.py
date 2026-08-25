"""A/B probe: does the RAG grounding nudge name the documents retrieval can see?

Identical on both branches. Only studio/backend/routes/inference.py differs.
Builds a real rag.db with two ingested, lexically retrievable documents, then asks
_apply_rag_nudge for the system-prompt nudge the model would receive.
"""
import asyncio
import inspect
import os
import sys
import tempfile

HOME = tempfile.mkdtemp(prefix = "pr9718-probe-")
os.environ["UNSLOTH_STUDIO_HOME"] = HOME
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.rag import store  # noqa: E402
from routes import inference  # noqa: E402
from storage import rag_db  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "search_knowledge_base"}}]
SCOPE = store.thread_scope("t-probe")
DOCS = {
    "d1": ("syllabus.pdf", "The course syllabus lists the grading policy and lecture plan."),
    "d2": ("allotment.pdf", "The hostel allotment letter confirms room 214 in block C."),
}


class _Chunk:
    def __init__(self, text):
        self.chunk_index = 0
        self.text = text
        self.page_number = 1
        self.source_page_index = 0
        self.token_count = len(text.split())
        self.kind = "text"


def _vec(text):
    # Deterministic 8-dim stand-in: this probe asserts on lexical retrieval, and a real
    # embedder would only add a model download to both branches equally.
    acc = [0.0] * 8
    for i, ch in enumerate(text):
        acc[i % 8] += (ord(ch) % 17) / 100.0
    return acc


def main():
    conn = rag_db.get_connection()
    for doc_id, (name, text) in DOCS.items():
        store.create_document(
            conn, scope = SCOPE, filename = name, sha256 = doc_id,
            document_id = doc_id, status = "pending",
        )
        store.add_chunks(conn, SCOPE, doc_id, [_Chunk(text)], [_vec(text)])
        store.set_document_status(conn, doc_id, "completed", num_chunks = 1)
    conn.commit()

    hits = store.search_lexical(conn, SCOPE, "allotment syllabus", 10)
    hydrated = store.chunks_by_id(conn, [cid for cid, _ in hits])
    retrievable = sorted({row["filename"] for row in hydrated.values()})
    conn.close()

    print(f"RETRIEVABLE: {retrievable}")
    if retrievable != sorted(n for n, _ in DOCS.values()):
        print("FAIL: setup did not make both documents retrievable; probe is inconclusive")
        return 2

    out = inference._apply_rag_nudge(
        "", TOOLS, rag_scope = {"thread_id": "t-probe"},
    )
    if inspect.isawaitable(out):
        out = asyncio.run(out)
    print(f"NUDGE: {out}")

    named = [n for n, _ in DOCS.values() if f'"{n}"' in out]
    print(f"NAMED_IN_NUDGE: {sorted(named)}")
    if sorted(named) == sorted(n for n, _ in DOCS.values()):
        print("PASS: the system prompt names every retrievable attached document")
        return 0
    print(
        "FAIL: retrieval can see "
        f"{retrievable} but the system prompt names {sorted(named)}; "
        "the model is told documents are attached without being told which"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
