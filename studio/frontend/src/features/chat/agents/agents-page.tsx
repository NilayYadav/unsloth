// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Alert02Icon,
  Download01Icon,
  Edit03Icon,
  PlusSignIcon,
  Robot01Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ChangeEvent, Suspense, lazy, useEffect, useId, useRef, useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useIsAccountOwner } from "@/features/auth";
import { type TranslationKey, useT } from "@/i18n";
import { Copy01Icon } from "@/lib/copy-icon";
import { TrashIcon } from "@/lib/hugeicons-derived";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { type McpServerConfig, listMcpServers } from "../api/mcp-servers-api";
import { useSkillsCatalog } from "../api/skills-api";
import { useModelChoices } from "../hooks/use-model-choices";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  AGENT_FILE_MAX_BYTES,
  agentCopyName,
  agentFileName,
  agentFileText,
  parseAgentFile,
  uniqueAgentName,
} from "./agent-file";
import type { AgentRecord, AgentTools } from "./agent-start";
import { useChatAgentsStore } from "./agents-store";
import {
  errorText,
  modelLabel,
  usableSkills,
  useModelsListed,
  useStartAgentChat,
} from "./use-start-agent-chat";

const CodingAgents = lazy(() =>
  import("@/features/settings/tabs/agents-tab").then((module) => ({ default: module.CodingAgents })),
);

const NO_MODEL = "__none__";
const TOOLS = [
  ["webSearch", "agentsPage.tools.webSearch"],
  ["code", "agentsPage.tools.code"],
  ["images", "agentsPage.tools.images"],
] as const satisfies readonly (readonly [keyof AgentTools, TranslationKey])[];
const ROW_BUTTON =
  "flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition hover:bg-[rgb(0_0_0_/_calc(0.05*var(--contrast-wash-gain,1)))] hover:text-foreground dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]";
const FIELD_LABEL = "text-ui-13 font-medium text-foreground";

function selectableServers(servers: readonly McpServerConfig[]): McpServerConfig[] {
  return servers.filter((server) => !server.builtin_id);
}

function toggled<T>(list: readonly T[], value: T, on: boolean): T[] {
  return on ? [...list.filter((entry) => entry !== value), value] : list.filter((entry) => entry !== value);
}

function PickList({
  label,
  empty,
  options,
  selected,
  disabled,
  onChange,
}: {
  label: string;
  empty: string;
  options: readonly { value: string; label: string }[];
  selected: readonly string[];
  disabled: boolean;
  onChange: (next: string[]) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className={FIELD_LABEL}>{label}</legend>
      {options.length === 0 ? (
        <p className="text-ui-13 text-muted-foreground">{empty}</p>
      ) : (
        <div className="flex max-h-36 flex-col gap-2 overflow-y-auto pt-2">
          {options.map((option) => (
            <label key={option.value} className="flex cursor-pointer items-center gap-2.5 text-ui-14">
              <Checkbox
                checked={selected.includes(option.value)}
                disabled={disabled}
                onCheckedChange={(on) => onChange(toggled(selected, option.value, on === true))}
              />
              <span className="min-w-0 truncate">{option.label}</span>
            </label>
          ))}
        </div>
      )}
    </fieldset>
  );
}

function AgentDialog({
  agent,
  servers,
  onClose,
}: {
  agent: AgentRecord | null;
  servers: readonly McpServerConfig[] | null;
  onClose: () => void;
}) {
  const t = useT();
  const ids = useId();
  const save = useChatAgentsStore((state) => state.save);
  const catalog = useSkillsCatalog();
  const skills = usableSkills(catalog.skills);
  const [name, setName] = useState(agent?.name ?? "");
  const [description, setDescription] = useState(agent?.description ?? "");
  const [modelId, setModelId] = useState(agent?.modelId ?? null);
  const [systemPrompt, setSystemPrompt] = useState(agent?.systemPrompt ?? "");
  const [tools, setTools] = useState<AgentTools>(
    agent?.tools ?? { webSearch: false, code: false, images: false },
  );
  const [mcpServerIds, setMcpServerIds] = useState(agent?.mcpServerIds ?? []);
  const [skillNames, setSkillNames] = useState(agent?.skills ?? []);
  const [busy, setBusy] = useState(false);
  const models = useModelChoices(modelId);
  const modelsListed = useModelsListed();

  async function submit() {
    if (!name.trim() || busy) return;
    setBusy(true);
    const loaded = useChatRuntimeStore.getState();
    try {
      await save({
        id: agent?.id ?? crypto.randomUUID(),
        name: name.trim(),
        description: description.trim(),
        modelId,
        modelGgufVariant:
          modelId !== null && modelId === loaded.params.checkpoint
            ? loaded.activeGgufVariant
            : modelId === agent?.modelId
              ? (agent?.modelGgufVariant ?? null)
              : null,
        systemPrompt,
        tools,
        mcpServerIds: servers
          ? mcpServerIds.filter((id) => servers.some((server) => server.id === id))
          : mcpServerIds,
        skills:
          catalog.initialized && !catalog.error
            ? skillNames.filter((skill) => skills.some((entry) => entry.name === skill))
            : skillNames,
        createdAt: agent?.createdAt,
      });
      onClose();
    } catch (error) {
      toast.error(t("agentsPage.saveFailed"), { description: errorText(error) });
      setBusy(false);
    }
  }

  return (
    <Dialog open={true} onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="corner-squircle dialog-soft-surface max-h-[90dvh] gap-4 overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="text-ui-21">
            {agent ? t("agentsPage.editTitle") : t("agentsPage.createTitle")}
          </DialogTitle>
          <DialogDescription className="sr-only">{t("agentsPage.description")}</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <div className="flex flex-col gap-1.5">
            <label htmlFor={`${ids}-name`} className={FIELD_LABEL}>
              {t("agentsPage.form.name")}
            </label>
            <Input
              id={`${ids}-name`}
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={200}
              disabled={busy}
              placeholder={t("agentsPage.form.namePlaceholder")}
              autoFocus={true}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label htmlFor={`${ids}-description`} className={FIELD_LABEL}>
              {t("agentsPage.form.description")}
            </label>
            <Input
              id={`${ids}-description`}
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              maxLength={2000}
              disabled={busy}
              placeholder={t("agentsPage.form.descriptionPlaceholder")}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label htmlFor={`${ids}-model`} className={FIELD_LABEL}>
              {t("agentsPage.form.model")}
            </label>
            <Select
              value={modelId ?? NO_MODEL}
              onValueChange={(value) => setModelId(value === NO_MODEL ? null : value)}
              disabled={busy}
            >
              <SelectTrigger id={`${ids}-model`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-w-[min(36rem,90vw)]">
                <SelectItem value={NO_MODEL}>{t("agentsPage.form.modelAny")}</SelectItem>
                {models.missing && (
                  <SelectItem value={models.missing}>
                    {modelsListed
                      ? t("agentsPage.form.modelMissing", { model: modelLabel(models.missing) })
                      : modelLabel(models.missing)}
                  </SelectItem>
                )}
                {modelsListed === null && (
                  <p className="px-3 py-2 text-ui-12 text-muted-foreground">
                    {t("agentsPage.form.modelsLoading")}
                  </p>
                )}
                {models.local.length > 0 && (
                  <SelectGroup>
                    <SelectLabel>{t("agentsPage.form.modelOnDevice")}</SelectLabel>
                    {models.local.map((option) => (
                      <SelectItem key={option.id} value={option.id}>
                        {option.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                )}
                {models.connections.map(
                  (group) =>
                    group.models.length > 0 && (
                      <SelectGroup key={group.provider.id}>
                        <SelectLabel>{group.provider.name}</SelectLabel>
                        {group.models.map((option) => (
                          <SelectItem key={option.id} value={option.id}>
                            {option.modelId}
                          </SelectItem>
                        ))}
                      </SelectGroup>
                    ),
                )}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1.5">
            <label htmlFor={`${ids}-prompt`} className={FIELD_LABEL}>
              {t("agentsPage.form.systemPrompt")}
            </label>
            <Textarea
              id={`${ids}-prompt`}
              value={systemPrompt}
              onChange={(event) => setSystemPrompt(event.target.value)}
              rows={5}
              disabled={busy}
              placeholder={t("agentsPage.form.systemPromptPlaceholder")}
              className="max-h-60 min-h-28"
            />
          </div>
          <fieldset className="flex flex-col gap-2">
            <legend className={FIELD_LABEL}>{t("agentsPage.form.tools")}</legend>
            <div className="flex flex-wrap gap-x-6 gap-y-2 pt-2">
              {TOOLS.map(([key, label]) => (
                <label key={key} className="flex cursor-pointer items-center gap-2.5 text-ui-14">
                  <Switch
                    checked={tools[key]}
                    disabled={busy}
                    onCheckedChange={(on) => setTools((current) => ({ ...current, [key]: on }))}
                  />
                  {t(label)}
                </label>
              ))}
            </div>
          </fieldset>
          <PickList
            label={t("agentsPage.form.mcpServers")}
            empty={t("agentsPage.form.mcpServersEmpty")}
            options={selectableServers(servers ?? []).map((server) => ({
              value: server.id,
              label: server.display_name,
            }))}
            selected={mcpServerIds}
            disabled={busy}
            onChange={setMcpServerIds}
          />
          <PickList
            label={t("agentsPage.form.skills")}
            empty={t("agentsPage.form.skillsEmpty")}
            options={skills.map((skill) => ({ value: skill.name, label: skill.name }))}
            selected={skillNames}
            disabled={busy}
            onChange={setSkillNames}
          />
          <DialogFooter>
            <Button type="button" variant="outline" disabled={busy} onClick={onClose}>
              {t("common.cancel")}
            </Button>
            <Button type="submit" variant="dark" disabled={busy || !name.trim()}>
              {agent ? t("agentsPage.form.save") : t("agentsPage.form.create")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function AgentRow({
  agent,
  servers,
  skills,
  modelMissing,
  onStart,
  onEdit,
  onDuplicate,
  onExport,
  onDelete,
}: {
  agent: AgentRecord;
  servers: readonly McpServerConfig[] | null;
  skills: readonly string[] | null;
  modelMissing: boolean;
  onStart: () => Promise<boolean>;
  onEdit: () => void;
  onDuplicate: () => void;
  onExport: () => void;
  onDelete: () => void;
}) {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const missingServers = servers
    ? agent.mcpServerIds.filter((id) => !servers.some((server) => server.id === id)).length
    : 0;
  const missingSkills = skills
    ? agent.skills.filter((name) => !skills.includes(name)).length
    : 0;
  const tools = TOOLS.filter(([key]) => agent.tools[key]).map(([, label]) => t(label));
  const summary = [
    agent.modelId ? modelLabel(agent.modelId) : t("agentsPage.anyModel"),
    ...tools,
    agent.mcpServerIds.length > missingServers
      ? t("agentsPage.mcpCount", { count: agent.mcpServerIds.length - missingServers })
      : null,
    agent.skills.length > missingSkills
      ? t("agentsPage.skillCount", { count: agent.skills.length - missingSkills })
      : null,
  ].filter(Boolean);
  const warnings = [
    modelMissing && agent.modelId
      ? t("agentsPage.modelUnavailable", { model: modelLabel(agent.modelId) })
      : null,
    missingServers > 0 ? t("agentsPage.mcpUnavailable", { count: missingServers }) : null,
    missingSkills > 0 ? t("agentsPage.skillUnavailable", { count: missingSkills }) : null,
  ].filter(Boolean);

  async function start() {
    setBusy(true);
    if (!(await onStart())) setBusy(false);
  }

  return (
    <div className="flex items-center gap-4 rounded-xl px-5 py-4 max-sm:flex-wrap max-sm:gap-x-1 transition-colors duration-150 hover:bg-muted/70 dark:hover:bg-[rgb(255_255_255_/_calc(0.055*var(--contrast-wash-gain,1)))]">
      <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-muted text-foreground/70">
        <HugeiconsIcon icon={Robot01Icon} strokeWidth={1.75} className="size-5" />
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <button
          type="button"
          onClick={onEdit}
          className="min-w-0 cursor-pointer self-start truncate rounded-md text-left text-ui-15 font-semibold text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          {agent.name}
        </button>
        {agent.description && (
          <p className="truncate text-ui-13 text-muted-foreground">{agent.description}</p>
        )}
        <p className="truncate text-ui-13 text-muted-foreground">{summary.join(" · ")}</p>
        {warnings.length > 0 && (
          <p role="status" className="flex items-center gap-1.5 text-ui-13 text-amber-600 dark:text-amber-400">
            <HugeiconsIcon icon={Alert02Icon} strokeWidth={1.75} className="size-4 shrink-0" />
            <span className="min-w-0 truncate">{warnings.join(" ")}</span>
          </p>
        )}
      </div>
      <Button variant="outline" disabled={busy} onClick={() => void start()}>
        {t("agentsPage.startChat")}
      </Button>
      <button
        type="button"
        aria-label={t("agentsPage.editAgent", { name: agent.name })}
        title={t("agentsPage.edit")}
        onClick={onEdit}
        className={ROW_BUTTON}
      >
        <HugeiconsIcon icon={Edit03Icon} strokeWidth={1.75} className="size-5" />
      </button>
      <button
        type="button"
        aria-label={t("agentsPage.duplicateAgent", { name: agent.name })}
        title={t("agentsPage.duplicate")}
        onClick={onDuplicate}
        className={ROW_BUTTON}
      >
        <HugeiconsIcon icon={Copy01Icon} strokeWidth={1.75} className="size-5" />
      </button>
      <button
        type="button"
        aria-label={t("agentsPage.exportAgent", { name: agent.name })}
        title={t("agentsPage.export")}
        onClick={onExport}
        className={ROW_BUTTON}
      >
        <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-5" />
      </button>
      <button
        type="button"
        aria-label={t("agentsPage.deleteAgent", { name: agent.name })}
        title={t("common.delete")}
        onClick={onDelete}
        className={ROW_BUTTON}
      >
        <HugeiconsIcon icon={TrashIcon} strokeWidth={1.75} className="size-5" />
      </button>
    </div>
  );
}

export function AgentsPage() {
  const t = useT();
  const agents = useChatAgentsStore((state) => state.agents);
  const status = useChatAgentsStore((state) => state.status);
  const error = useChatAgentsStore((state) => state.error);
  const refresh = useChatAgentsStore((state) => state.refresh);
  const remove = useChatAgentsStore((state) => state.remove);
  const save = useChatAgentsStore((state) => state.save);
  const isOwner = useIsAccountOwner();
  const importRef = useRef<HTMLInputElement>(null);
  const [servers, setServers] = useState<McpServerConfig[] | null>(null);
  const [editing, setEditing] = useState<AgentRecord | null | undefined>(undefined);
  const [deleting, setDeleting] = useState<AgentRecord | null>(null);
  const { modelMissing, start } = useStartAgentChat();
  const catalog = useSkillsCatalog();
  const skills =
    catalog.initialized && !catalog.error
      ? usableSkills(catalog.skills).map((skill) => skill.name)
      : null;

  useEffect(() => {
    void refresh();
    void listMcpServers().then(setServers, () => undefined);
  }, [refresh]);

  async function addCopy(setup: Omit<AgentRecord, "id" | "createdAt" | "updatedAt">) {
    try {
      await save({ ...setup, id: crypto.randomUUID(), createdAt: undefined });
    } catch (saveError) {
      toast.error(t("agentsPage.saveFailed"), { description: errorText(saveError) });
    }
  }

  function exportAgent(agent: AgentRecord) {
    downloadFile(agentFileText(agent), agentFileName(agent), "application/json").catch(
      (exportError) => {
        if (!isDownloadCancelled(exportError)) {
          toast.error(t("agentsPage.exportFailed"), { description: errorText(exportError) });
        }
      },
    );
  }

  async function importAgent(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const setup = file.size <= AGENT_FILE_MAX_BYTES ? parseAgentFile(await file.text()) : null;
    if (!setup) {
      toast.error(t("agentsPage.importFailed"), {
        description: t("agentsPage.importInvalid"),
      });
      return;
    }
    const names = useChatAgentsStore.getState().agents.map((agent) => agent.name);
    await addCopy({ ...setup, name: uniqueAgentName(setup.name, names) });
  }

  async function confirmDelete() {
    const target = deleting;
    if (!target) return;
    setDeleting(null);
    try {
      await remove(target.id);
    } catch (removeError) {
      toast.error(t("agentsPage.deleteFailed"), { description: errorText(removeError) });
    }
  }

  const loading = status === "idle" || status === "loading";

  return (
    <main className="mx-auto w-full max-w-5xl 4xl:max-w-6xl px-6 pb-10 pt-8 max-sm:px-4 font-heading sm:px-10">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
          {t("agentsPage.title")}
        </h1>
        <div className="flex items-center gap-2">
          <input
            ref={importRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(event) => void importAgent(event)}
          />
          <Button variant="outline" onClick={() => importRef.current?.click()}>
            <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.75} className="size-icon" />
            {t("agentsPage.import")}
          </Button>
          <Button variant="dark" onClick={() => setEditing(null)}>
            <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
            {t("agentsPage.newAgent")}
          </Button>
        </div>
      </div>
      <p className="mt-4 text-ui-15 text-muted-foreground">{t("agentsPage.description")}</p>

      {loading && agents.length === 0 ? (
        <div className="mt-12">
          {Array.from({ length: 3 }).map((_, index) => (
            <div key={index} className="flex flex-col gap-2 rounded-xl px-5 py-4">
              <Skeleton className="h-4 w-48 rounded-[8px]" />
              <Skeleton className="h-3.5 w-72 rounded-[8px]" />
            </div>
          ))}
        </div>
      ) : status === "error" && agents.length === 0 ? (
        <div className="mt-16 flex flex-col items-center gap-3 text-center">
          <p className="text-sm text-muted-foreground">{t("agentsPage.loadFailed")}</p>
          {error && <p className="max-w-md text-ui-13 text-muted-foreground">{error}</p>}
          <Button variant="outline" onClick={() => void refresh()}>
            {t("agentsPage.retry")}
          </Button>
        </div>
      ) : agents.length === 0 ? (
        <div className="mt-16 flex flex-col items-center gap-2 text-center">
          <span className="mb-2 flex size-12 items-center justify-center rounded-2xl bg-muted text-foreground/70">
            <HugeiconsIcon icon={Robot01Icon} strokeWidth={1.75} className="size-6" />
          </span>
          <p className="text-ui-15 font-semibold text-foreground">{t("agentsPage.emptyTitle")}</p>
          <p className="max-w-sm text-sm text-muted-foreground">{t("agentsPage.emptyDescription")}</p>
          <Button
            variant="outline"
            className="mt-3 border-none bg-background shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-card dark:shadow-none"
            onClick={() => setEditing(null)}
          >
            <HugeiconsIcon icon={PlusSignIcon} strokeWidth={1.75} className="size-icon" />
            {t("agentsPage.createFirst")}
          </Button>
        </div>
      ) : (
        <div className="mt-10 flex flex-col">
          {agents.map((agent) => (
            <AgentRow
              key={agent.id}
              agent={agent}
              servers={servers}
              skills={skills}
              modelMissing={modelMissing(agent.modelId)}
              onStart={() => start(agent)}
              onEdit={() => setEditing(agent)}
              onDuplicate={() =>
                void addCopy({
                  ...agent,
                  name: agentCopyName(
                    agent.name,
                    agents.map((entry) => entry.name),
                  ),
                })
              }
              onExport={() => exportAgent(agent)}
              onDelete={() => setDeleting(agent)}
            />
          ))}
        </div>
      )}

      {isOwner && (
        <section aria-labelledby="coding-agents-heading" className="mt-14">
          <h2 id="coding-agents-heading" className="text-ui-21 font-semibold text-foreground">
            {t("agentsPage.codingAgents")}
          </h2>
          <div className="mt-4 font-sans">
            <Suspense fallback={<Skeleton className="h-40 w-full rounded-xl" />}>
              <CodingAgents embedded={true} />
            </Suspense>
          </div>
        </section>
      )}

      {editing !== undefined && (
        <AgentDialog
          key={editing?.id ?? "new"}
          agent={editing}
          servers={servers}
          onClose={() => setEditing(undefined)}
        />
      )}
      <AlertDialog open={deleting !== null} onOpenChange={(open) => !open && setDeleting(null)}>
        <AlertDialogContent onOverlayClick={() => setDeleting(null)}>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("agentsPage.deleteTitle")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("agentsPage.deleteDescription", { name: deleting?.name ?? "" })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => void confirmDelete()}>
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </main>
  );
}
