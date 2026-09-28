// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useEffect, useState } from "react";
import {
  type AutoRouterCandidate,
  type AutoRouterSettings,
  type NamedRouter,
  ROUTER_SLOTS,
  type RouterPreview,
  type RouterSlot,
  listAutoRouterCandidates,
  loadAutoRouterSettings,
  loadNamedRouters,
  previewRouter,
  resetAutoRouterSettings,
  routerIdFromName,
  saveNamedRouters,
} from "../api/auto-router";
import { SettingsSection } from "./settings-section";

const NONE = "__none__";

const LAYA_STATUS_KEYS = {
  ready: "settings.chat.autoRouter.laya.ready",
  loading: "settings.chat.autoRouter.laya.loading",
  unavailable: "settings.chat.autoRouter.laya.unavailable",
} as const;

interface Draft {
  originalId: string | null;
  name: string;
  slots: Partial<Record<RouterSlot, string>>;
}

function shortName(model: string): string {
  return model.slice(model.lastIndexOf("/") + 1) || model;
}

function errorMessage(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

export function RoutersSection() {
  const t = useT();
  const [auto, setAuto] = useState<AutoRouterSettings | null>(null);
  const [routers, setRouters] = useState<NamedRouter[]>([]);
  const [candidates, setCandidates] = useState<AutoRouterCandidate[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testModel, setTestModel] = useState("auto");
  const [testPrompt, setTestPrompt] = useState("");
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState<RouterPreview | null>(null);
  const [testError, setTestError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const fail = (err: unknown) => {
      if (live) setError(errorMessage(err, t("settings.chat.autoRouter.loadError")));
    };
    loadAutoRouterSettings({ force: true }).then((next) => live && setAuto(next), fail);
    loadNamedRouters({ force: true }).then((next) => live && setRouters(next), fail);
    listAutoRouterCandidates().then((next) => live && setCandidates(next), fail);
    return () => {
      live = false;
    };
  }, [t]);

  const persist = async (next: NamedRouter[]) => {
    setBusy(true);
    setError(null);
    try {
      const saved = await saveNamedRouters(next);
      setRouters(saved);
      if (!saved.some((router) => router.model === testModel)) setTestModel("auto");
      toast.success(t("settings.chat.autoRouter.saved"));
      return true;
    } catch (err) {
      const message = errorMessage(err, t("settings.chat.autoRouter.saveError"));
      setError(message);
      toast.error(message);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const saveDraft = async () => {
    if (!draft) return;
    const name = draft.name.trim();
    if (!name) {
      setError(t("settings.chat.autoRouter.nameRequired"));
      return;
    }
    if (!draft.slots.general) {
      setError(t("settings.chat.autoRouter.generalRequired"));
      return;
    }
    const others = routers.filter((router) => router.id !== draft.originalId);
    const router: NamedRouter = {
      id: draft.originalId ?? routerIdFromName(name, others.map((item) => item.id)),
      name,
      slots: draft.slots,
    };
    const next = draft.originalId
      ? routers.map((item) => (item.id === draft.originalId ? router : item))
      : [...routers, router];
    if (await persist(next)) setDraft(null);
  };

  const resetAuto = async () => {
    setBusy(true);
    setError(null);
    try {
      setAuto(await resetAutoRouterSettings());
      toast.success(t("settings.chat.autoRouter.saved"));
    } catch (err) {
      setError(errorMessage(err, t("settings.chat.autoRouter.saveError")));
    } finally {
      setBusy(false);
    }
  };

  const runTest = async () => {
    if (!testPrompt.trim()) return;
    setTesting(true);
    setTestError(null);
    setResult(null);
    try {
      setResult(await previewRouter(testModel, testPrompt));
    } catch (err) {
      setTestError(errorMessage(err, t("settings.chat.autoRouter.loadError")));
    } finally {
      setTesting(false);
    }
  };

  const optionsFor = (slot: RouterSlot) =>
    slot === "vision" ? candidates.filter((model) => model.vision) : candidates;
  const layaStatus =
    auto?.laya && auto.laya in LAYA_STATUS_KEYS
      ? LAYA_STATUS_KEYS[auto.laya as keyof typeof LAYA_STATUS_KEYS]
      : null;

  return (
    <SettingsSection
      title={t("settings.chat.autoRouter.title")}
      description={t("settings.chat.autoRouter.description")}
    >
      <div className="flex flex-col gap-4 py-3">
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium">{t("settings.chat.autoRouter.auto")}</span>
          {auto && (
            <p className="text-xs text-muted-foreground">
              {t(
                auto.automatic
                  ? "settings.chat.autoRouter.automaticNote"
                  : "settings.chat.autoRouter.customNote",
              )}
            </p>
          )}
          {layaStatus && (
            <p className="text-xs text-muted-foreground">
              {t("settings.chat.autoRouter.laya.label")}: {t(layaStatus)}
            </p>
          )}
          {auto && !auto.automatic && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="self-start"
              onClick={resetAuto}
              disabled={busy}
            >
              {t("settings.chat.autoRouter.resetAutomatic")}
            </Button>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium">{t("settings.chat.autoRouter.routersTitle")}</span>
          {routers.length === 0 && !draft && (
            <p className="text-xs text-muted-foreground">{t("settings.chat.autoRouter.routersEmpty")}</p>
          )}
          {routers.map((router) =>
            draft?.originalId === router.id ? null : (
              <div
                key={router.id}
                className="flex items-start justify-between gap-3 rounded-lg border border-border p-3"
              >
                <div className="min-w-0">
                  <div className="text-sm font-medium">{router.name}</div>
                  <div className="text-xs text-muted-foreground break-words">
                    {ROUTER_SLOTS.filter((slot) => router.slots[slot])
                      .map((slot) => `${t(`settings.chat.autoRouter.task.${slot}`)}: ${shortName(router.slots[slot]!)}`)
                      .join(" · ")}
                  </div>
                  {router.ready === false && (
                    <div className="text-xs text-destructive">{t("settings.chat.autoRouter.notReady")}</div>
                  )}
                </div>
                <div className="flex shrink-0 gap-1">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={busy || draft !== null}
                    onClick={() => {
                      setError(null);
                      setDraft({ originalId: router.id, name: router.name, slots: { ...router.slots } });
                    }}
                  >
                    {t("settings.chat.autoRouter.edit")}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    disabled={busy || draft !== null}
                    onClick={() => void persist(routers.filter((item) => item.id !== router.id))}
                  >
                    {t("common.delete")}
                  </Button>
                </div>
              </div>
            ),
          )}

          {draft ? (
            <div className="flex flex-col gap-3 rounded-lg border border-border p-3">
              <label className="flex flex-col gap-1 text-xs">
                {t("settings.chat.autoRouter.routerName")}
                <Input
                  value={draft.name}
                  maxLength={80}
                  placeholder={t("settings.chat.autoRouter.routerNamePlaceholder")}
                  onChange={(event) => setDraft({ ...draft, name: event.target.value })}
                />
              </label>
              {ROUTER_SLOTS.map((slot) => (
                <label key={slot} className="flex flex-col gap-1 text-xs">
                  {t(`settings.chat.autoRouter.task.${slot}`)}
                  <Select
                    value={draft.slots[slot] || NONE}
                    onValueChange={(value) =>
                      setDraft({
                        ...draft,
                        slots: { ...draft.slots, [slot]: value === NONE ? undefined : value },
                      })
                    }
                  >
                    <SelectTrigger aria-label={t(`settings.chat.autoRouter.task.${slot}`)}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {slot !== "general" && (
                        <SelectItem value={NONE}>{t("settings.chat.autoRouter.slotNone")}</SelectItem>
                      )}
                      {slot === "general" && !draft.slots.general && (
                        <SelectItem value={NONE} disabled={true}>
                          {t("settings.chat.autoRouter.slotNone")}
                        </SelectItem>
                      )}
                      {optionsFor(slot).map((model) => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </label>
              ))}
              <p className="text-xs text-muted-foreground">{t("settings.chat.autoRouter.slotHint")}</p>
              <div className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setError(null);
                    setDraft(null);
                  }}
                  disabled={busy}
                >
                  {t("common.cancel")}
                </Button>
                <Button type="button" onClick={saveDraft} disabled={busy}>
                  {busy ? t("common.saving") : t("common.save")}
                </Button>
              </div>
            </div>
          ) : (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="self-start"
              disabled={busy}
              onClick={() => {
                setError(null);
                setDraft({ originalId: null, name: "", slots: {} });
              }}
            >
              {t("settings.chat.autoRouter.newRouter")}
            </Button>
          )}
          {error && (
            <span className="text-xs text-destructive" role="alert">
              {error}
            </span>
          )}
        </div>

        <div className="flex flex-col gap-2">
          <span className="text-sm font-medium">{t("settings.chat.autoRouter.testTitle")}</span>
          <div className="flex flex-wrap gap-2">
            <Select value={testModel} onValueChange={setTestModel}>
              <SelectTrigger aria-label={t("settings.chat.autoRouter.testTitle")} className="w-44">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">{t("settings.chat.autoRouter.auto")}</SelectItem>
                {routers.map((router) => (
                  <SelectItem key={router.id} value={router.model ?? `router/${router.id}`}>
                    {router.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              className="min-w-48 flex-1"
              value={testPrompt}
              placeholder={t("settings.chat.autoRouter.testPlaceholder")}
              onChange={(event) => setTestPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void runTest();
              }}
            />
            <Button type="button" variant="outline" onClick={runTest} disabled={testing || !testPrompt.trim()}>
              {t("settings.chat.autoRouter.testButton")}
            </Button>
          </div>
          {result && (
            <p className="text-sm">
              → <span className="font-medium">{shortName(result.model)}</span>
              <span className="text-muted-foreground">
                {" · "}
                {result.reason}
                {result.loaded ? "" : ` · ${t("settings.chat.autoRouter.willLoad")}`}
              </span>
            </p>
          )}
          {testError && (
            <span className="text-xs text-destructive" role="alert">
              {testError}
            </span>
          )}
          <p className="text-xs text-muted-foreground">{t("settings.chat.autoRouter.testNoLoad")}</p>
        </div>
      </div>
    </SettingsSection>
  );
}
