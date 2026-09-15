// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { Add01Icon, ServerStack01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type FormEvent, useId, useState } from "react";
import {
  addClusterNode,
  removeClusterNode,
  testClusterNode,
  updateClusterMode,
  updateClusterNode,
} from "../api/cluster";
import {
  type ClusterActive,
  type ClusterNode,
  type ClusterNodeStatus,
  type ClusterState,
  clusterErrorMessage,
  deviceSummary,
  familyLabel,
  formatMbps,
  formatMib,
  formatRtt,
  formatSeconds,
  incompatibleReasonMessage,
  linkKindLabel,
} from "../api/cluster-state";
import type { ClusterController, ClusterRun } from "../hooks/use-cluster-state";
import { SettingsRow } from "./settings-row";

type NodeNote = { text: string; destructive: boolean };

const NODE_STATUS_LABEL: Record<ClusterNodeStatus, string> = {
  online: "Online",
  busy: "Busy",
  offline: "Offline",
  unpaired: "Not paired",
};

const NODE_DOT_CLASS: Record<ClusterNodeStatus, string> = {
  online: "bg-emerald-500",
  busy: "bg-amber-500",
  offline: "bg-muted-foreground",
  unpaired: "bg-red-500",
};

const SLOW_LINK_WARNING =
  "This link is slow for a cluster: first loads take much longer and replies are slower. Use Ethernet or a Thunderbolt cable if you can.";

function ModeStatus({ state }: { state: ClusterState | null }) {
  let label = "Unavailable";
  if (state) {
    label = state.head.mode === "off" ? "Off" : "On";
    if (state.head.active) {
      label = "In use";
    }
  }
  return (
    <output
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
      aria-live="polite"
    >
      <span
        className={cn(
          "size-2 rounded-full",
          state?.head.mode === "auto"
            ? "bg-emerald-500"
            : "bg-muted-foreground",
        )}
      />
      {label}
    </output>
  );
}

function ActiveCluster({ active }: { active: ClusterActive | null }) {
  if (!active) {
    return null;
  }
  const names = active.nodes
    .map((node) => node.name ?? node.host ?? node.id)
    .join(", ");
  return (
    <div className="flex flex-col gap-1 border-t border-border/60 px-4 py-2.5">
      <p className="text-sm font-medium text-foreground">
        Running across {names} · {formatMib(active.capacityMib)} added
      </p>
      {active.failure ? (
        <p className="text-xs leading-snug text-destructive">
          {active.failure}
        </p>
      ) : null}
    </div>
  );
}

function nodeHardware(node: ClusterNode): string {
  return [familyLabel(node.family), deviceSummary(node.devices, node.freeMib)]
    .filter(Boolean)
    .join(" · ");
}

function nodeLinkLine(node: ClusterNode): string | null {
  const { link } = node;
  const kind = linkKindLabel(link.kind);
  const parts = [
    link.speedMbps === null ? kind : `${kind} ${formatMbps(link.speedMbps)}`,
  ];
  if (link.rttMs !== null) {
    parts.push(formatRtt(link.rttMs));
  }
  if (link.throughputMbps !== null) {
    parts.push(`measured ${formatMbps(link.throughputMbps)}`);
  }
  if (node.loadSecondsPerGib !== null) {
    parts.push(
      `≈ ${formatSeconds(node.loadSecondsPerGib)}s per GB on first load`,
    );
  }
  return parts.length === 1 && link.kind === "unknown"
    ? null
    : parts.join(" · ");
}

function nodeNotes(node: ClusterNode, localFamily: string | null): NodeNote[] {
  const notes: NodeNote[] = [];
  if (node.status === "unpaired") {
    notes.push({
      text: "This computer no longer trusts this one. Remove it and pair again.",
      destructive: true,
    });
  } else if (node.lastError) {
    notes.push({
      text: clusterErrorMessage(node.lastError),
      destructive: node.status === "offline",
    });
  }
  if (node.incompatibleReason) {
    notes.push({
      text: incompatibleReasonMessage(
        node.incompatibleReason,
        node.family,
        localFamily,
      ),
      destructive: true,
    });
  }
  if (node.warning === "build_mismatch") {
    notes.push({
      text: "Different Unsloth versions. Update both computers if loading fails.",
      destructive: false,
    });
  }
  if (node.link.kind === "wifi" || node.link.kind === "vpn") {
    notes.push({ text: SLOW_LINK_WARNING, destructive: false });
  }
  return notes;
}

function NodeRow({
  node,
  localFamily,
  cluster,
  onRemove,
}: {
  node: ClusterNode;
  localFamily: string | null;
  cluster: ClusterController;
  onRemove: (node: ClusterNode) => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const testing = cluster.busy.has(`test:${node.id}`);
  const locked =
    testing ||
    cluster.busy.has(`enable:${node.id}`) ||
    cluster.busy.has(`remove:${node.id}`);
  const hardware = nodeHardware(node);
  const link = nodeLinkLine(node);

  const act = async (key: string, request: () => Promise<ClusterState>) => {
    setError(null);
    const failure = await cluster.run(`${key}:${node.id}`, request);
    if (failure) {
      setError(clusterErrorMessage(failure));
    }
  };

  return (
    <li className="flex flex-col gap-1.5 border-t border-border/60 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <div className="flex min-w-0 flex-col gap-0.5">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-sm font-medium text-foreground">
              {node.name}
            </span>
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span
                className={cn(
                  "size-2 rounded-full",
                  NODE_DOT_CLASS[node.status],
                )}
              />
              {NODE_STATUS_LABEL[node.status]}
            </span>
          </div>
          {hardware ? (
            <span className="text-xs text-muted-foreground">{hardware}</span>
          ) : null}
          {link ? (
            <span className="text-xs text-muted-foreground">{link}</span>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={locked || node.status === "unpaired"}
            onClick={() => act("test", () => testClusterNode(node.id))}
          >
            {testing ? "Testing…" : "Test"}
          </Button>
          <Switch
            checked={node.enabled}
            disabled={locked}
            onCheckedChange={(enabled) =>
              act("enable", () => updateClusterNode(node.id, enabled))
            }
            aria-label={`Use ${node.name}`}
          />
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={locked}
            onClick={() => onRemove(node)}
          >
            Remove
          </Button>
        </div>
      </div>
      {nodeNotes(node, localFamily).map((note) => (
        <p
          key={note.text}
          className={cn(
            "text-xs leading-snug",
            note.destructive ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {note.text}
        </p>
      ))}
      {error ? (
        <output className="block text-xs leading-snug text-destructive">
          {error}
        </output>
      ) : null}
    </li>
  );
}

function AddComputerDialog({
  run,
  busy,
  disabled,
}: {
  run: ClusterRun;
  busy: boolean;
  disabled: boolean;
}) {
  const addressId = useId();
  const codeId = useId();
  const [open, setOpen] = useState(false);
  const [address, setAddress] = useState("");
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const canSubmit = !busy && address.trim() !== "" && code.trim().length >= 4;

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) {
      return;
    }
    setError(null);
    const failure = await run("pair", () =>
      addClusterNode(address.trim(), code.trim()),
    );
    if (failure) {
      setError(clusterErrorMessage(failure));
      return;
    }
    setOpen(false);
    setAddress("");
    setCode("");
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        setError(null);
      }}
    >
      <DialogTrigger asChild={true}>
        <Button
          type="button"
          size="sm"
          className="min-w-20 shrink-0 gap-1.5"
          disabled={disabled}
        >
          <HugeiconsIcon icon={Add01Icon} className="size-3.5" />
          Add computer
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Add a computer</DialogTitle>
          <DialogDescription>
            On the other computer, open Settings &gt; Cluster and turn on Share
            this computer. Enter one of the addresses and the code it shows.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={addressId}>Address</Label>
            <Input
              id={addressId}
              value={address}
              placeholder="192.168.1.20"
              autoComplete="off"
              spellCheck={false}
              onChange={(event) => {
                setAddress(event.target.value);
                setError(null);
              }}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={codeId}>Pairing code</Label>
            <Input
              id={codeId}
              value={code}
              placeholder="ABCD-2345"
              autoComplete="off"
              autoCapitalize="characters"
              spellCheck={false}
              maxLength={32}
              className="font-mono tracking-wider"
              onChange={(event) => {
                setCode(event.target.value.toUpperCase());
                setError(null);
              }}
            />
          </div>
          <output
            aria-live="polite"
            className="block min-h-4 text-xs leading-snug text-destructive"
          >
            {error}
          </output>
          <DialogFooter>
            <Button type="submit" disabled={!canSubmit}>
              {busy ? "Pairing…" : "Pair"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function RemoveNodeDialog({
  node,
  open,
  cluster,
  onOpenChange,
}: {
  node: ClusterNode | null;
  open: boolean;
  cluster: ClusterController;
  onOpenChange: (open: boolean) => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const removing = node ? cluster.busy.has(`remove:${node.id}`) : false;

  const confirm = async () => {
    if (!node) {
      return;
    }
    setError(null);
    const failure = await cluster.run(`remove:${node.id}`, () =>
      removeClusterNode(node.id),
    );
    if (failure) {
      setError(clusterErrorMessage(failure));
      return;
    }
    onOpenChange(false);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        setError(null);
        onOpenChange(next);
      }}
    >
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Remove {node?.name}?</DialogTitle>
          <DialogDescription>
            This computer stops using its GPU memory. To use it again, pair with
            a new code.
          </DialogDescription>
        </DialogHeader>
        {error ? (
          <output className="block text-xs leading-snug text-destructive">
            {error}
          </output>
        ) : null}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            disabled={removing}
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            disabled={removing}
            onClick={confirm}
          >
            {removing ? "Removing…" : "Remove"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function ClusterNodesSection({
  cluster,
}: {
  cluster: ClusterController;
}) {
  const { state, busy, run } = cluster;
  const [modeError, setModeError] = useState<string | null>(null);
  const [removeTarget, setRemoveTarget] = useState<ClusterNode | null>(null);
  const [removeOpen, setRemoveOpen] = useState(false);
  const nodes = state?.head.nodes ?? [];

  const setMode = async (enabled: boolean) => {
    setModeError(null);
    const failure = await run("mode", () =>
      updateClusterMode(enabled ? "auto" : "off"),
    );
    if (failure) {
      setModeError(clusterErrorMessage(failure));
    }
  };

  return (
    <section
      data-settings-label="Use other computers"
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <HugeiconsIcon
              icon={ServerStack01Icon}
              className="size-4 text-foreground"
            />
          </div>
          <div className="flex min-w-0 flex-col gap-0.5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-base font-semibold font-heading text-foreground">
                Use other computers
              </h2>
              <ModeStatus state={state} />
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Load models that don't fit on this computer by adding GPU memory
              from your other computers on the same network.
            </p>
          </div>
        </div>
        <AddComputerDialog
          run={run}
          busy={busy.has("pair")}
          disabled={state === null}
        />
      </div>

      {modeError ? (
        <p className="border-t border-border/60 px-4 py-2.5 text-xs leading-snug text-destructive">
          {modeError}
        </p>
      ) : null}
      <ActiveCluster active={state?.head.active ?? null} />

      <div className="border-t border-border/60 px-4 py-1">
        <SettingsRow
          label="Use when a model doesn't fit"
          description="Unsloth only reaches for other computers when a model doesn't fit here, and picks the fewest, fastest ones."
        >
          <Switch
            checked={state?.head.mode === "auto"}
            disabled={state === null || busy.has("mode")}
            onCheckedChange={setMode}
            aria-label="Use other computers"
          />
        </SettingsRow>
      </div>

      {state !== null && nodes.length === 0 ? (
        <p className="border-t border-border/60 px-4 py-6 text-center text-xs text-muted-foreground">
          No computers added yet. Turn on Share this computer on another
          computer, then add it here.
        </p>
      ) : null}
      {nodes.length > 0 ? (
        <ul className="flex flex-col">
          {nodes.map((node) => (
            <NodeRow
              key={node.id}
              node={node}
              localFamily={state?.head.local.family ?? null}
              cluster={cluster}
              onRemove={(target) => {
                setRemoveTarget(target);
                setRemoveOpen(true);
              }}
            />
          ))}
        </ul>
      ) : null}

      <RemoveNodeDialog
        node={removeTarget}
        open={removeOpen}
        cluster={cluster}
        onOpenChange={setRemoveOpen}
      />
    </section>
  );
}
