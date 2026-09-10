# PR 10753 — API-monitor UI evidence

Upstream: https://github.com/unslothai/unsloth/pull/10753

Before: merge base `d0dbe9059efa443c6ad8bd1d51af7e2d9276a2bc`.
After: PR head `1f9e49059f074982129064c3476c5687e1bc5e48`.

Both checkouts were separately installed with `npm ci` and built with `npm run build`.
Production `main.app`, `setup_frontend`, auth, Responses routes and API-monitor
routes ran in separate uvicorn servers, separate Python environments with identical
dependency versions, separate Studio homes, caches, credentials and ports.
No browser API responses, production routes, monitor rows or inference calls were mocked.
MLX auto-repair was disabled before the final run to keep dependencies identical.

Browser: Chromium, Playwright 1.62.0, macOS ARM64, Python 3.12.13.
Viewport: 1440 × 1000. Both screenshots use the same browser clip:
`x=310, y=35, width=1100, height=915`. The composite adds labels and a gap without
rescaling or altering UI pixels. Both originals and the composite were visually inspected.

## Scenario

No model is loaded on either side. API recording is enabled on both.

1. Clear the API monitor.
2. Send the control request without `previous_response_id`. Both return HTTP 400
   with `No model loaded...` and create exactly one failed `/v1/responses` row.
3. Clear the API monitor again.
4. Send the same non-streaming request with `previous_response_id=resp_previous`.
5. Open the real `/api-monitor` page and compare its rendered rows to the live API.

```json
{
  "input": "What was my project code?",
  "stream": false,
  "previous_response_id": "resp_previous"
}
```

| Observed value | Before | After |
| --- | --- | --- |
| HTTP status | 400 | 400 |
| API error | No model loaded | previous_response_id unsupported; send full input history |
| Error code | null | unsupported_parameter |
| Monitor request rows | 1 | 0 |
| Rendered Requests / Errors | 1 / 1 | 0 / 0 |
| Browser console / page errors | 0 / 0 | 0 / 0 |
| Loaded model | None | None |

The before monitor displays its generic `An internal error occurred` text; the HTTP
response itself says no model is loaded. The after UI displays its existing empty
state. It does **not** display the new validation error; that error goes to the API caller.

The control proves that the empty after-state is not disabled logging. The scene
proves the indirect API-monitor effect and early validation on an unloaded server.
It does not prove successful inference, model switching, or a changed normal chat
flow. This is local browser evidence, not a GitHub Actions result or another human's test.

See `meta.json` for measured facts and `before-api.json` / `after-api.json` for
the exact HTTP response bodies. Secrets, auth storage and server logs are excluded.

The capture uses the installed `pr-ui-evidence` and `pr-repro-ci` skill helpers.
The included source scripts document the real requests and assertions.
