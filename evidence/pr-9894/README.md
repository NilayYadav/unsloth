# PR 9894 UI evidence

Canvas blocked-resource notice, merge base vs head.

- base `e745043b3d6dd3e8d4e69ff806480c555f511439` (the PR's merge base)
- head `acd820d4ee2b646c9ea943de493756f3ac1b0e4b`
- scene `canvas_network_blocked_alert`
- Chromium, 1280x900, headless, one isolated Studio install per side
- `UNSLOTH_DISABLE_MLX_AUTOREPAIR=1` on both sides, so neither Studio races a
  background MLX reinstall while the page boots

Both sides run the same seeded thread. The canvas' only external resource is a
Chart.js script from cdn.jsdelivr.net, and "Allow canvas network access" is left
off (its default) on both sides, so the preview frame's CSP refuses that script
and the frame reports exactly one block.

`scene-facts.json` holds the full per-side facts and the pre-declared expectation.
