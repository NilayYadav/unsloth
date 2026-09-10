# PR 10762: native Anthropic tool-image evidence

Compared merge base `d0dbe9059efa443c6ad8bd1d51af7e2d9276a2bc` with approved head `dbf0d7aacf8f733939f2969fe004742f53dfe890`.

The real Studio API monitor shows `NO_IMAGE` before and `742` after for the same tool-result screenshot. No mocked HTTP routes, monitor data, model responses, or tokenizer were used in the live evidence. The screenshot source is a synthetic public fixture (`tool-capture.png`).

| Live case | Before | After |
| --- | --- | --- |
| Base64, mixed text/image/text | NO_IMAGE; 99 input tokens | 742; 231 input tokens |
| Image-only result | NO_IMAGE; 92 | 742; 224 |
| Data URL image source | NO_IMAGE; 99 | 742; 231 |
| Two images in one result | NO_IMAGE; 99 | 742; 363 |
| Image in earlier tool history | NO_IMAGE; 144 | 742; 276 |
| Streaming result | NO_IMAGE | 742 |
| Top-level image control | 742; 178 | 742; 178 |
| Image count endpoint | 200, incomplete count 99 | 503, explicit refusal |
| Text count control | 200, 32 tokens | 200, 32 tokens |

The generation token figures are actual model usage, not estimated fixture values. Timings are single local observations, with prompt caching, and are not performance claims.

## Coverage

1,339 selected existing/head tests passed, 6 skipped. Seven supplemental evidence cases passed. The six skips are five redundant matrix cases and one optional MLX import. The initial instrumented suite exposed NumPy's extension reload restriction; preloading NumPy and Transformers before pytest made the complete rerun pass. Both frontend builds and all six UI-driver offline guards also passed.

Coverage.py records 50/50 changed executable lines and 38/38 branch destinations originating at those changed lines. This is **not 100% coverage of all backend code**. `changed-coverage.json` retains whole-file statement percentages for context. Seven supplemental tests are evidence-only; the PR code/head was not modified.

## Environment and reproduction

Local Apple Silicon macOS; Python 3.12.13. Exact Python package versions are in `requirements.txt`. Real Gemma `unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL`, revision `739965d73654c0ead8020786aa998fc813070087`, with its F16 projector. Pinned llama-server build 10840, commit `58670d128`. Context 8192, one slot, temperature 0, thinking disabled. See `meta.json` for browser/viewport and exact facts.

Create `before` and `wt_r10762` worktrees at the refs above, with these scripts in a sibling `full-evidence` directory. Build each frontend independently using `npm ci --ignore-scripts && npm run build`. Install the Studio requirements and listed test/browser dependencies into a Python 3.12 environment. These runs launch the source app directly; they do not test the installer. Homes, auth credentials, ports, and HF/XDG caches are separate; Python dependencies, cached read-only weights, and the pinned server binary are shared. The model path is configured in `load_model.py`.

Launch each side with `LLAMA_SERVER_PATH=<pinned-binary> python launch.py <worktree> <isolated-home> <port>` using ports 18861/18862. Run `load_model.py <side> <port>`, `live_request.py <side> <port>`, `live_matrix.py <side> <port>`, and `capture_ui.py <side> <port>`. The tools log in and rotate the isolated home's bootstrap credential without publishing it. The skill helper modules are provided by `pr-ui-evidence` and `pr-repro-ci`. Run models sequentially, unloading between sides. Run the final `live_request.py` immediately before UI capture to clear the monitor to one comparable request. `package_evidence.py` checks the expected answers, visible image-marker delta, zero page errors, and nonidentical screenshots before composing labelled originals.

Coverage suite: `test_anthropic_messages.py`, `test_anthropic_native_tool_images.py`, `test_anthropic_admission.py`, `test_llama_admission*.py`, and `test_openai_auto_switch.py` under `studio/backend/tests`. Use `python -c 'import numpy, transformers, pytest; raise SystemExit(pytest.main())' <tests> --cov=core.inference.anthropic_compat --cov=routes.inference --cov-branch`. On macOS use a case-sensitive pytest temporary directory for the cache case-variant test. Run `test_extended_tool_images.py` with `--cov-append`, then `finalize_coverage.py` against the JSON report. Test logs are included.

## Limits

This is local automated evidence, not a GitHub Actions run or independent human testing. Those original merge gates remain pending. The screenshots are unaltered UI pixels composed with before/after labels and were visually inspected. No auth state, credentials, model weights, or full server logs are published.
