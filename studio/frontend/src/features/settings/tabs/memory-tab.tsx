// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  buildUserMemoryBlock,
  bulkSaveMemoryEntries,
  deleteMemoryEntry,
  estimateMemoryTokens,
  listMemoryEntries,
  MEMORY_EXTRACTION_PROMPT,
  MEMORY_TOKEN_LIMIT,
  type MemoryEntry,
  memoryKey,
  type MemorySource,
  parseImportedMemories,
  saveMemoryEntry,
  selectMemoriesForPrompt,
  useChatRuntimeStore,
} from "@/features/chat";
import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { PencilIcon, Trash2Icon } from "lucide-react";
import {
  type FormEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

const MEMORY_MAX_CHARS = 2000;

function newMemory(text: string, source: MemorySource, age = 0): MemoryEntry {
  const now = Date.now();
  return {
    id: crypto.randomUUID().replace(/-/g, "").slice(0, 12),
    text,
    source,
    enabled: true,
    createdAt: now - age,
    updatedAt: now,
  };
}

function updatedMemory(
  entry: MemoryEntry,
  patch: Pick<Partial<MemoryEntry>, "text" | "enabled">,
): MemoryEntry {
  return { ...entry, ...patch, updatedAt: Date.now() };
}

export function MemoryTab() {
  const t = useT();
  const memoryEnabled = useChatRuntimeStore((state) => state.memoryEnabled);
  const setMemoryEnabled = useChatRuntimeStore(
    (state) => state.setMemoryEnabled,
  );
  const hydratePersistedSettings = useChatRuntimeStore(
    (state) => state.hydratePersistedSettings,
  );
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(
    null,
  );
  const [importOpen, setImportOpen] = useState(false);

  const reload = useCallback(
    () =>
      listMemoryEntries()
        .then(setEntries)
        .catch(() => toast.error(t("settings.memory.loadFailed"))),
    [t],
  );

  useEffect(() => {
    void hydratePersistedSettings();
    void reload();
  }, [hydratePersistedSettings, reload]);

  const enabledTexts = useMemo(
    () => entries.filter((e) => e.enabled).map((e) => e.text),
    [entries],
  );
  const sentCount = selectMemoriesForPrompt(enabledTexts).length;
  const tokens = estimateMemoryTokens(buildUserMemoryBlock(enabledTexts));

  const persist = async (entry: MemoryEntry) => {
    setEntries((prev) =>
      prev.some((e) => e.id === entry.id)
        ? prev.map((e) => (e.id === entry.id ? entry : e))
        : [entry, ...prev],
    );
    try {
      await saveMemoryEntry(entry);
    } catch {
      toast.error(t("settings.memory.saveFailed"));
      void reload();
    }
  };

  const add = (event: FormEvent) => {
    event.preventDefault();
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    void persist(newMemory(text, "manual"));
  };

  const remove = async (id: string) => {
    setEntries((prev) => prev.filter((e) => e.id !== id));
    try {
      await deleteMemoryEntry(id);
    } catch {
      toast.error(t("settings.memory.saveFailed"));
      void reload();
    }
  };

  const saveEdit = (entry: MemoryEntry) => {
    const text = editing?.text.trim() ?? "";
    setEditing(null);
    if (!text || text === entry.text) return;
    void persist(updatedMemory(entry, { text }));
  };

  return (
    <div className="settings-page">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.memory.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.memory.description")}
        </p>
      </header>

      <SettingsSection title={t("settings.memory.enable")} hideHeading>
        <SettingsRow
          label={t("settings.memory.enable")}
          description={t("settings.memory.enableDescription")}
        >
          <Switch
            aria-label={t("settings.memory.enable")}
            checked={memoryEnabled}
            onCheckedChange={setMemoryEnabled}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.memory.import")}
          description={t("settings.memory.importDescription")}
        >
          <Button variant="outline" onClick={() => setImportOpen(true)}>
            {t("settings.memory.startImport")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.memory.listTitle")}
        description={
          sentCount < enabledTexts.length
            ? `${t("settings.memory.usage", { tokens, limit: MEMORY_TOKEN_LIMIT })} ${t("settings.memory.overLimit", { count: enabledTexts.length - sentCount })}`
            : t("settings.memory.usage", { tokens, limit: MEMORY_TOKEN_LIMIT })
        }
      >
        <form onSubmit={add} className="flex gap-2 py-3">
          <Input
            value={draft}
            maxLength={MEMORY_MAX_CHARS}
            placeholder={t("settings.memory.placeholder")}
            aria-label={t("settings.memory.add")}
            onChange={(event) => setDraft(event.target.value)}
          />
          <Button type="submit" disabled={!draft.trim()}>
            {t("settings.memory.add")}
          </Button>
        </form>
        {entries.length === 0 ? (
          <p className="py-3 text-xs text-muted-foreground">
            {t("settings.memory.empty")}
          </p>
        ) : (
          <ul className="flex flex-col divide-y divide-border/60">
            {entries.map((entry) => (
              <li key={entry.id} className="flex items-center gap-3 py-2.5">
                {editing?.id === entry.id ? (
                  <Textarea
                    autoFocus={true}
                    value={editing.text}
                    maxLength={MEMORY_MAX_CHARS}
                    aria-label={t("settings.memory.edit")}
                    className="min-h-0 py-2"
                    onChange={(event) =>
                      setEditing({ id: entry.id, text: event.target.value })
                    }
                    onBlur={() => saveEdit(entry)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        saveEdit(entry);
                      } else if (event.key === "Escape") {
                        event.stopPropagation();
                        setEditing(null);
                      }
                    }}
                  />
                ) : (
                  <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                    <span
                      className={cn(
                        "text-sm break-words",
                        !entry.enabled && "text-muted-foreground line-through",
                      )}
                    >
                      {entry.text}
                    </span>
                    {entry.source === "import" ? (
                      <span className="text-ui-11 text-muted-foreground">
                        {t("settings.memory.importedLabel")}
                      </span>
                    ) : null}
                  </div>
                )}
                <Switch
                  aria-label={t("settings.memory.use")}
                  checked={entry.enabled}
                  onCheckedChange={(enabled) =>
                    void persist(updatedMemory(entry, { enabled }))
                  }
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-8"
                  aria-label={t("settings.memory.edit")}
                  onClick={() => setEditing({ id: entry.id, text: entry.text })}
                >
                  <PencilIcon className="size-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-8"
                  aria-label={t("settings.memory.delete")}
                  onClick={() => void remove(entry.id)}
                >
                  <Trash2Icon className="size-4" />
                </Button>
              </li>
            ))}
          </ul>
        )}
      </SettingsSection>

      <MemoryImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        existing={entries}
        onSaved={reload}
      />
    </div>
  );
}

function ImportStep({
  step,
  title,
  last = false,
  children,
}: {
  step: number;
  title: string;
  last?: boolean;
  children: ReactNode;
}) {
  return (
    <li className="relative flex gap-3">
      {last ? null : (
        <span
          aria-hidden={true}
          className="absolute top-7 bottom-0 left-3 w-px -translate-x-1/2 bg-border"
        />
      )}
      <span className="flex size-6 shrink-0 items-center justify-center rounded-full border border-border text-ui-12 font-medium">
        {step}
      </span>
      <div
        className={cn("flex min-w-0 flex-1 flex-col gap-2", !last && "pb-5")}
      >
        <h3 className="pt-0.5 text-sm font-medium">{title}</h3>
        {children}
      </div>
    </li>
  );
}

function MemoryImportDialog({
  open,
  onOpenChange,
  existing,
  onSaved,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  existing: MemoryEntry[];
  onSaved: () => void;
}) {
  const t = useT();
  const [pasted, setPasted] = useState("");
  const [saving, setSaving] = useState(false);

  const saved = useMemo(
    () => new Set(existing.map((e) => memoryKey(e.text))),
    [existing],
  );
  const parsed = useMemo(
    () =>
      parseImportedMemories(pasted).filter(
        (text) => text.length <= MEMORY_MAX_CHARS,
      ),
    [pasted],
  );
  const fresh = parsed.filter((text) => !saved.has(memoryKey(text)));

  const close = (next: boolean) => {
    if (saving) return;
    onOpenChange(next);
    if (!next) setPasted("");
  };

  const save = async () => {
    setSaving(true);
    try {
      const count = await bulkSaveMemoryEntries(
        fresh.map((text, index) => newMemory(text, "import", index)),
      );
      const skipped = parsed.length - fresh.length;
      toast.success(
        skipped
          ? `${t("settings.memory.imported", { count })} ${t("settings.memory.alreadySaved", { count: skipped })}`
          : t("settings.memory.imported", { count }),
      );
      onSaved();
      setSaving(false);
      close(false);
    } catch {
      setSaving(false);
      toast.error(t("settings.memory.saveFailed"));
    }
  };

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("settings.memory.importTitle")}</DialogTitle>
        </DialogHeader>
        <ol className="flex min-w-0 flex-col">
          <ImportStep step={1} title={t("settings.memory.importStep1")}>
            <div className="relative rounded-xl bg-muted">
              <p className="max-h-40 overflow-y-auto p-3 pb-11 text-xs leading-relaxed text-muted-foreground">
                {MEMORY_EXTRACTION_PROMPT}
              </p>
              <Button
                size="sm"
                variant="outline"
                className="absolute right-2 bottom-2"
                onClick={() => {
                  void copyToClipboard(MEMORY_EXTRACTION_PROMPT).then((ok) =>
                    ok
                      ? toast.success(t("settings.memory.promptCopied"))
                      : toast.error(t("settings.memory.copyFailed")),
                  );
                }}
              >
                <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
                {t("settings.memory.copyPrompt")}
              </Button>
            </div>
          </ImportStep>
          <ImportStep
            step={2}
            title={t("settings.memory.importStep2")}
            last={true}
          >
            <Textarea
              fieldSizing="fixed"
              rows={7}
              value={pasted}
              aria-label={t("settings.memory.importStep2")}
              placeholder={t("settings.memory.pastePlaceholder")}
              onChange={(event) => setPasted(event.target.value)}
            />
          </ImportStep>
        </ol>
        <DialogFooter>
          <Button variant="ghost" onClick={() => close(false)}>
            {t("common.cancel")}
          </Button>
          <Button
            disabled={saving || fresh.length === 0}
            onClick={() => void save()}
          >
            {t("settings.memory.addToMemory")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
