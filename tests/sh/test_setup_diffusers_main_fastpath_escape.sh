#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Test setup.sh's handling of the pinned Diffusers main probe's exit status.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

awk '/^_fast_path_escapes\(\) \{/ {on=1} on {print} on && /^}/ {exit}' "$SETUP_SH" > "$WORK/escapes.sh"
grep -q -- "--diffusers-main-needs-dependency-pass" "$WORK/escapes.sh" || {
    echo "FATAL: the diffusers main escape is not in _fast_path_escapes in $SETUP_SH" >&2; exit 1; }

VENV_DIR="$WORK/venv"
mkdir -p "$VENV_DIR/bin"
: > "$WORK/install_python_stack.py"

# Record the probe and return $PROBE_RC; other checks report no repair needed.
cat > "$VENV_DIR/bin/python" <<'STUB'
#!/bin/sh
case "$*" in
    *--diffusers-main-needs-dependency-pass*) printf '%s\n' "$*" >> "$PROBE_LOG"; exit "$PROBE_RC" ;;
esac
exit 1
STUB
chmod +x "$VENV_DIR/bin/python"

PROBE_LOG="$WORK/calls.txt"
export PROBE_LOG

run_escapes() {
    (
        PROBE_RC="$1"
        _SKIP_PYTHON_DEPS="${2:-true}"
        _OFFLINE_FAST_PATH="${3:-false}"
        _UV_OFFLINE="${4:-false}"
        export PROBE_RC
        SCRIPT_DIR="$WORK"
        unset UNSLOTH_STUDIO_FULL_DEPS UNSLOTH_DESKTOP_BACKEND_VERSION \
            UNSLOTH_TORCH_INDEX_URL UNSLOTH_TORCH_INDEX_FAMILY
        SUBSTEPS=""
        substep() { SUBSTEPS="$SUBSTEPS|$1"; }
        _uv_offline_requested() { [ "$_UV_OFFLINE" = true ]; }
        # shellcheck disable=SC1090
        . "$WORK/escapes.sh"
        _fast_path_escapes
        echo "$_SKIP_PYTHON_DEPS$SUBSTEPS"
    )
}

_forced="false|pinned Diffusers main build is missing -- forcing dependency pass to install it..."

echo "=== only a conclusive answer forces the pass ==="
assert_eq "exit 0 forces the dependency pass, and says why" "$_forced" "$(run_escapes 0)"
assert_eq "exit 1 keeps the fast path silently" "true" "$(run_escapes 1)"

echo "=== a probe that cannot answer keeps the fast path ==="
# 2 is an unknown flag: an older install_python_stack.py beside a newer setup.sh.
for rc in 2 124 126 127 137; do
    assert_eq "exit $rc keeps the fast path" "true" "$(run_escapes "$rc")"
done

echo "=== the probe is asked once, with the module and the flag ==="
: > "$PROBE_LOG"
run_escapes 1 >/dev/null
assert_eq "one call" "1" "$(wc -l < "$PROBE_LOG" | tr -d ' ')"
assert_eq "exactly that call" \
    "$WORK/install_python_stack.py --diffusers-main-needs-dependency-pass" "$(cat "$PROBE_LOG")"

echo "=== a pass another escape already forced is not probed again ==="
: > "$PROBE_LOG"
assert_eq "stays forced" "false" "$(run_escapes 0 false)"
assert_eq "no probe" "0" "$(wc -l < "$PROBE_LOG" | tr -d ' ')"

echo "=== offline keeps the fast path, because the source build can only fail ==="
: > "$PROBE_LOG"
assert_eq "the offline keep is not overridden" "true" "$(run_escapes 0 true true false)"
assert_eq "UV_OFFLINE alone keeps it" "true" "$(run_escapes 0 true false true)"
assert_eq "offline never probes" "0" "$(wc -l < "$PROBE_LOG" | tr -d ' ')"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
