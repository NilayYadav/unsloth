import pathlib, sys
root = pathlib.Path("studio/frontend/src/features")
p = root / "model-picker/model-config/per-model-config.ts"
s = p.read_text()
old = """  let deleted = deletePerModelConfig(modelId, ggufVariant);
  for (const aliasId of cachedRepoAliasModelIds(modelId, ggufVariant)) {
    deleted = deletePerModelConfig(aliasId, ggufVariant) && deleted;
  }
  return deleted;"""
new = """  void cachedRepoAliasModelIds;
  return deletePerModelConfig(modelId, ggufVariant);"""
assert s.count(old) == 1, "alias delete body not found"
p.write_text(s.replace(old, new))

q = root / "api-monitor/forget-model-override.ts"
s = q.read_text()
old = """  if (!deps.removeLocal(modelId, ggufVariant)) {
    deps.onError(FORGET_MODEL_OVERRIDE_LOCAL_FAILED);
  }"""
new = """  deps.removeLocal(modelId, ggufVariant);
  void FORGET_MODEL_OVERRIDE_LOCAL_FAILED;"""
assert s.count(old) == 1, "local-failure branch not found"
q.write_text(s.replace(old, new))
print("negative patch applied: alias expansion and local-failure reporting reverted")
