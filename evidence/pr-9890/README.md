# PR #9890 UI evidence

`pr9890-update-banner-before-after.png` compares the PR merge base
(`32ba55b57`) with the PR head, both rendering the real `UpdateBanner` /
`UpdateScreen` through the real `useTauriUpdate` hook.

Each side runs the same scripted Tauri bridge (`window.__TAURI_INTERNALS__`),
the same click on **Update**, and the same waits. Only the application code
differs. `scene-facts.json` holds the DOM facts read at each capture.
