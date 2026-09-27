// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import {
  AUTO_TASKS,
  type AutoRouterModel,
  type AutoRouterSettings,
  type AutoTask,
  listAutoRouterCandidates,
  loadAutoRouterSettings,
  saveAutoRouterSettings,
} from "../api/auto-router";
import { SettingsSection } from "./settings-section";

function suggestedModel(id: string): AutoRouterModel {
  const name = id.toLowerCase();
  const vision = /(^|[-_/])(vl|vision|multimodal)([-_/]|$)/.test(name);
  const task: AutoTask = vision
    ? "vision"
    : /coder|code|devstral/.test(name)
      ? "code"
      : /math|thinking|reason/.test(name)
        ? "reasoning"
        : /writer|creative/.test(name)
          ? "writing"
          : "general";
  return {
    id,
    tasks: [task],
    vision,
    tools: false,
    context_length: null,
  };
}

export function AutoRouterSection() {
  const t = useT();
  const [settings, setSettings] = useState<AutoRouterSettings | null>(null);
  const [candidates, setCandidates] = useState<{ id: string; name: string }[]>([]);
  const [loadingCandidates, setLoadingCandidates] = useState(true);
  const [candidate, setCandidate] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let live = true;
    loadAutoRouterSettings().then(
      (next) => live && setSettings(next),
      (err) => live && setError(err instanceof Error ? err.message : t("settings.apiKeys.autoRouter.loadError")),
    );
    listAutoRouterCandidates().then(
      (available) => live && setCandidates(available),
      (err) => live && setError(err instanceof Error ? err.message : t("settings.apiKeys.autoRouter.loadError")),
    ).finally(() => live && setLoadingCandidates(false));
    return () => { live = false; };
  }, [t]);

  const updateModel = (id: string, patch: Partial<AutoRouterModel>) => {
    setSettings((current) => current && ({
      ...current,
      models: current.models.map((model) => model.id === id ? { ...model, ...patch } : model),
    }));
  };

  const addModel = () => {
    if (!candidate || !settings || settings.models.some((model) => model.id === candidate)) return;
    setSettings({
      ...settings,
      models: [...settings.models, suggestedModel(candidate)],
      default_model: settings.default_model || candidate,
    });
    setCandidate("");
  };

  const removeModel = (id: string) => {
    setSettings((current) => {
      if (!current) return current;
      const models = current.models.filter((model) => model.id !== id);
      return {
        ...current,
        models,
        default_model: current.default_model === id ? models[0]?.id || null : current.default_model,
        rules: current.rules.filter((rule) => rule.model !== id),
      };
    });
  };

  const save = async () => {
    if (!settings) return;
    setBusy(true);
    setError(null);
    try {
      setSettings(await saveAutoRouterSettings({
        ...settings,
        rules: settings.rules.filter((rule) => rule.contains.trim()),
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("settings.apiKeys.autoRouter.saveError"));
    } finally {
      setBusy(false);
    }
  };

  const available = candidates.filter((model) => !settings?.models.some((entry) => entry.id === model.id));

  return (
    <SettingsSection
      title={t("settings.apiKeys.autoRouter.title")}
      description={t("settings.apiKeys.autoRouter.description")}
    >
      <div className="flex flex-col gap-3 py-3">
        <div className="flex flex-wrap gap-2">
          <Select value={candidate} onValueChange={setCandidate} disabled={loadingCandidates}>
            <SelectTrigger aria-label={t("settings.apiKeys.autoRouter.addModel")} className="min-w-48 flex-1">
              <SelectValue placeholder={t(loadingCandidates ? "settings.apiKeys.autoRouter.loadingModels" : "settings.apiKeys.autoRouter.addModel")} />
            </SelectTrigger>
            <SelectContent>
              {available.map((model) => <SelectItem key={model.id} value={model.id}>{model.name}</SelectItem>)}
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={addModel} disabled={!candidate || busy}>{t("settings.apiKeys.autoRouter.add")}</Button>
        </div>
        {settings?.models.map((model) => (
          <div key={model.id} className="rounded-lg border border-border p-3">
            <div className="flex items-start justify-between gap-2">
              <span className="min-w-0 break-all text-sm font-medium">{candidates.find((item) => item.id === model.id)?.name || model.id}</span>
              <Button variant="ghost" size="sm" onClick={() => removeModel(model.id)} aria-label={`${t("settings.apiKeys.autoRouter.remove")} ${model.id}`}>{t("settings.apiKeys.autoRouter.remove")}</Button>
            </div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-2">
              {AUTO_TASKS.map((task) => (
                <label key={task} className="flex items-center gap-1.5 text-xs">
                  <Checkbox
                    checked={model.tasks.includes(task)}
                    onCheckedChange={(checked) => updateModel(model.id, {
                      tasks: checked ? [...model.tasks, task] : model.tasks.filter((item) => item !== task),
                    })}
                  />
                  {t(`settings.apiKeys.autoRouter.task.${task}`)}
                </label>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-1.5 text-xs">
                <Checkbox checked={model.vision} onCheckedChange={(checked) => updateModel(model.id, { vision: checked === true })} />
                {t("settings.apiKeys.autoRouter.vision")}
              </label>
              <label className="flex items-center gap-1.5 text-xs">
                <Checkbox checked={model.tools} onCheckedChange={(checked) => updateModel(model.id, { tools: checked === true })} />
                {t("settings.apiKeys.autoRouter.tools")}
              </label>
              <label className="flex items-center gap-1.5 text-xs">
                {t("settings.apiKeys.autoRouter.context")}
                <Input
                  className="h-8 w-28"
                  type="number"
                  min={256}
                  value={model.context_length ?? ""}
                  placeholder={t("settings.apiKeys.autoRouter.unknown")}
                  onChange={(event) => updateModel(model.id, { context_length: event.target.value ? Number(event.target.value) : null })}
                />
              </label>
            </div>
          </div>
        ))}
        {settings && settings.models.length > 0 && (
          <div className="flex items-center gap-3 text-sm">
            <span>{t("settings.apiKeys.autoRouter.default")}</span>
            <Select value={settings.default_model || ""} onValueChange={(value) => setSettings({ ...settings, default_model: value })}>
              <SelectTrigger aria-label={t("settings.apiKeys.autoRouter.default")} className="min-w-44 flex-1"><SelectValue /></SelectTrigger>
              <SelectContent>{settings.models.map((model) => <SelectItem key={model.id} value={model.id}>{model.id}</SelectItem>)}</SelectContent>
            </Select>
          </div>
        )}
        {settings?.rules.map((rule, index) => (
          <div key={index} className="flex flex-wrap items-center gap-2">
            <Input
              className="min-w-40 flex-1"
              aria-label={t("settings.apiKeys.autoRouter.ruleText")}
              placeholder={t("settings.apiKeys.autoRouter.ruleText")}
              value={rule.contains}
              onChange={(event) => setSettings({ ...settings, rules: settings.rules.map((item, position) => position === index ? { ...item, contains: event.target.value } : item) })}
            />
            <Select value={rule.model} onValueChange={(value) => setSettings({ ...settings, rules: settings.rules.map((item, position) => position === index ? { ...item, model: value } : item) })}>
              <SelectTrigger aria-label={t("settings.apiKeys.autoRouter.ruleModel")} className="min-w-40 flex-1"><SelectValue /></SelectTrigger>
              <SelectContent>{settings.models.map((model) => <SelectItem key={model.id} value={model.id}>{model.id}</SelectItem>)}</SelectContent>
            </Select>
            <Button variant="ghost" size="sm" onClick={() => setSettings({ ...settings, rules: settings.rules.filter((_, position) => position !== index) })}>{t("settings.apiKeys.autoRouter.remove")}</Button>
          </div>
        ))}
        {settings && settings.models.length > 0 && (
          <Button variant="outline" size="sm" className="self-start" onClick={() => setSettings({ ...settings, rules: [...settings.rules, { contains: "", model: settings.default_model || settings.models[0].id }] })}>
            {t("settings.apiKeys.autoRouter.addRule")}
          </Button>
        )}
        <div className="flex items-center justify-between gap-3">
          {error ? <span className="text-xs text-destructive" role="alert">{error}</span> : <span />}
          <Button onClick={save} disabled={!settings || busy}>{busy ? t("common.saving") : t("common.save")}</Button>
        </div>
      </div>
    </SettingsSection>
  );
}
