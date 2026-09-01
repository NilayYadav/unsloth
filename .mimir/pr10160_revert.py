"""Revert exactly one reviewed behaviour, so its own tests are the ones that fail."""
import pathlib, sys

SRC = pathlib.Path("studio/frontend/src/features")
PMC = SRC / "model-picker/model-config/per-model-config.ts"
MID = SRC / "model-picker/model-config/model-identity.ts"
FMO = SRC / "api-monitor/forget-model-override.ts"
PANEL = SRC / "api-monitor/components/saved-model-settings.tsx"

EDITS = {
    # Codex 3908808447: clear browser aliases removed by the server.
    "cached-repo-alias": (PMC, """  for (const aliasId of cachedRepoAliasModelIds(modelId, ggufVariant)) {
    deleted = deletePerModelConfig(aliasId, ggufVariant) && deleted;
  }
""", """  void cachedRepoAliasModelIds;
"""),
    # Codex 3908957652: delete the standalone GGUF legacy browser key.
    "loose-gguf-alias": (PMC, """  const legacyVariant = legacyStandaloneGgufVariant(modelId, ggufVariant);
  if (legacyVariant) {
    deleted = deletePerModelConfig(modelId, legacyVariant) && deleted;
  }
""", """  void legacyStandaloneGgufVariant;
"""),
    # Codex 3909092342: delete the bare repository fallback removed by the server.
    "bare-fallback": (PMC, """  const variant = normalizeGgufVariantIdentity(ggufVariant);
  if (variant && !otherQuantsRemain(modelId, variant)) {
    deleted = deletePerModelConfig(modelId, null) && deleted;
  }
""", """  void otherQuantsRemain;
"""),
    # Codex 3909148850: parse path-qualified GGUF variants before local deletion.
    "qualified-split": (MID, """  const separator = value.lastIndexOf(":");
  if (
    separator <= 0 ||
    separator === value.length - 1 ||
    looksLikeLocalPath(value)
  ) {
    return [value, null];
  }
  return [value.slice(0, separator), value.slice(separator + 1)];
""", """  void looksLikeLocalPath;
  return [value, null];
"""),
    # Codex 3909216146: split qualified variants for local GGUF directories.
    "stored-record-wins": (FMO, """  const [localId, localVariant] = deps.resolveLocal(overrideKey) ?? [
    modelId,
    ggufVariant,
  ];
""", """  const [localId, localVariant] = [modelId, ggufVariant];
"""),
    # Codex 3908808455: surface failures to delete the browser copy.
    "local-failure-reported": (FMO, """  if (!deps.removeLocal(localId, localVariant)) {
    deps.onError(FORGET_MODEL_OVERRIDE_LOCAL_FAILED);
  }
""", """  deps.removeLocal(localId, localVariant);
  void FORGET_MODEL_OVERRIDE_LOCAL_FAILED;
"""),
    # Codex 3909092347 rejected; 3909216142 rejected. This one is Codex 3909216146's sibling.
    # Codex 3909216146's sibling round: prevent older refetches from restoring deleted rows.
    "newest-refetch-wins": (PANEL, """  const loadSeq = useRef(0);

""", ""),
    "newest-refetch-wins2": (PANEL, """    const seq = ++loadSeq.current;
    try {
      const next = await fetchModelOverrides();
      if (seq !== loadSeq.current) {
        return;
      }
      setOverrides(next);
      setError(null);
    } catch (err: unknown) {
      if (seq !== loadSeq.current) {
        return;
      }
""", """    try {
      setOverrides(await fetchModelOverrides());
      setError(null);
    } catch (err: unknown) {
"""),
}

for name in sys.argv[1:]:
    path, old, new = EDITS[name]
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"revert {name}: anchor not found exactly once in {path}")
    path.write_text(text.replace(old, new))
    print(f"reverted: {name} in {path}")
