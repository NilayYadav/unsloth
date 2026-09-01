#!/usr/bin/env bash
# One reverted behaviour at a time, so the assertions that fail name the defect Codex found.
set -euo pipefail
cd "$(dirname "$0")/.."
TESTS="tests/per-model-config-delete-aliases.test.ts tests/api-monitor-forget-model-override.test.ts tests/api-load-settings-refetch-sequencing.test.ts"

expect_fail() {
  local name="$1"; shift
  local edits="$1"; shift
  python3 .mimir/pr10160_revert.py $edits
  ( cd studio/frontend && set +e
    node --experimental-strip-types --test $TESTS > /tmp/neg.tap 2>&1
    echo $? > /tmp/neg.code )
  local code; code=$(cat /tmp/neg.code)
  git checkout -- studio/frontend/src
  if [ "$code" -eq 0 ]; then
    echo "FAIL[$name]: the tests passed with the behaviour reverted"; sed -n '1,40p' /tmp/neg.tap; exit 1
  fi
  for want in "$@"; do
    if ! grep -q "^not ok .* - $want\$" /tmp/neg.tap; then
      echo "FAIL[$name]: expected '$want' to fail"; grep '^not ok' /tmp/neg.tap || true; exit 1
    fi
  done
  echo "NEGATIVE OK [$name]: $(grep -c '^not ok' /tmp/neg.tap) assertion(s) failed, including the ones named"
}

expect_fail cached-repo-alias "cached-repo-alias" \
  "forgetting the snapshot-path row also drops the repo-id record" \
  "forgetting the repo-id row also drops a record still keyed by the path"
expect_fail loose-gguf-alias "loose-gguf-alias" \
  "a bare-path forget also drops the label a loose .gguf used to be keyed by"
expect_fail bare-fallback "bare-fallback" \
  "forgetting a repo's last quant also drops the bare record a load falls back to" \
  "the same forget from the legacy side drops the bare-path record too"
expect_fail qualified-split "qualified-split" \
  "a path-qualified variant splits off the repo it belongs to" \
  "a filename-stem variant splits off the repo too"
expect_fail stored-record-wins "stored-record-wins" \
  "the stored record wins over the key's own split"
expect_fail local-failure-reported "local-failure-reported" \
  "a browser copy that could not be deleted is reported, not swallowed"
expect_fail newest-refetch-wins "newest-refetch-wins newest-refetch-wins2" \
  "every refetch takes a sequence number" \
  "a superseded refetch paints no rows" \
  "a superseded refetch does not report its failure either"

git diff --quiet && echo "tree restored to head"
( cd studio/frontend && node --experimental-strip-types --test $TESTS > /tmp/pos.tap 2>&1 )
grep -E '^# (tests|pass|fail)' /tmp/pos.tap
grep -q '^# fail 0' /tmp/pos.tap
echo "POSITIVE OK: every one of them passes on the PR head"
