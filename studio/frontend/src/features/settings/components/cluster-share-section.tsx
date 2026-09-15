// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ComputerArrowUpIcon,
  Copy01Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useRef, useState } from "react";
import {
  clearClusterShareCache,
  newClusterPairingCode,
  revokeClusterHead,
  startClusterShare,
  stopClusterShare,
  updateClusterShareSettings,
} from "../api/cluster";
import {
  type ClusterAddress,
  type ClusterDevice,
  type ClusterPairedHead,
  type ClusterPairing,
  type ClusterShare,
  type ClusterShareState,
  type ClusterState,
  bytesToGib,
  clusterErrorMessage,
  formatAgo,
  formatBytes,
  formatClock,
  formatCountdown,
  formatMbps,
  formatMib,
  formatSeconds,
  linkKindLabel,
  serverNowMs,
  shareAddressLabel,
  transferMbps,
} from "../api/cluster-state";
import type { ClusterController } from "../hooks/use-cluster-state";
import { SettingsRow } from "./settings-row";

type ShareAct = (
  key: string,
  request: () => Promise<ClusterState>,
) => Promise<void>;

const STATE_LABEL: Record<ClusterShareState, string> = {
  off: "Off",
  starting: "Starting",
  online: "Online",
  error: "Error",
};

const STATE_DOT_CLASS: Record<ClusterShareState, string> = {
  off: "bg-muted-foreground",
  starting: "animate-pulse bg-amber-500",
  online: "bg-emerald-500",
  error: "bg-red-500",
};

const UNAVAILABLE_MESSAGE =
  "This install doesn't include ggml-rpc-server yet. Update Unsloth to share this computer.";

function useNow(): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}

function ShareStatus({ share }: { share: ClusterShare | null }) {
  return (
    <output
      className="flex items-center gap-1.5 text-xs text-muted-foreground"
      aria-live="polite"
    >
      <span
        className={cn(
          "size-2 rounded-full",
          STATE_DOT_CLASS[share?.state ?? "off"],
        )}
      />
      {share ? STATE_LABEL[share.state] : "Unavailable"}
    </output>
  );
}

function CopyCodeButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<number | null>(null);
  useEffect(() => {
    return () => {
      if (copyTimer.current !== null) {
        window.clearTimeout(copyTimer.current);
      }
    };
  }, []);
  return (
    <Button
      type="button"
      size="sm"
      variant="outline"
      className="gap-1.5"
      aria-label={copied ? "Copied pairing code" : "Copy pairing code"}
      onClick={async () => {
        if (!(await copyToClipboard(code))) {
          return;
        }
        setCopied(true);
        if (copyTimer.current !== null) {
          window.clearTimeout(copyTimer.current);
        }
        copyTimer.current = window.setTimeout(() => setCopied(false), 1800);
      }}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        className="size-3.5"
      />
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

function PairingPanel({
  pairing,
  now,
  busy,
  onNewCode,
}: {
  pairing: ClusterPairing | null;
  now: number;
  busy: boolean;
  onNewCode: () => void;
}) {
  const secondsLeft = pairing ? pairing.expiresAt - now / 1000 : 0;
  let expiry = "No code yet.";
  if (pairing) {
    expiry =
      secondsLeft > 0
        ? `Expires in ${formatCountdown(secondsLeft)}`
        : "Expired. A new code is on its way.";
  }
  return (
    <div className="flex flex-col gap-2 border-t border-border/60 p-4">
      <span className="text-sm font-medium text-foreground">Pairing code</span>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <code className="font-mono text-2xl font-semibold tracking-widest text-foreground">
          {pairing?.code ?? "---------"}
        </code>
        <div className="flex shrink-0 items-center gap-2">
          {pairing ? <CopyCodeButton code={pairing.code} /> : null}
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="gap-1.5"
            disabled={busy}
            onClick={onNewCode}
          >
            <HugeiconsIcon icon={RefreshIcon} className="size-3.5" />
            New code
          </Button>
        </div>
      </div>
      <span className="text-xs text-muted-foreground tabular-nums">
        {expiry}
      </span>
    </div>
  );
}

function AddressList({
  addresses,
  controlPort,
  online,
}: {
  addresses: ClusterAddress[];
  controlPort: number | null;
  online: boolean;
}) {
  if (addresses.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1.5 border-t border-border/60 p-4">
      <span
        className={cn(
          "text-sm font-medium",
          online ? "text-foreground" : "text-muted-foreground",
        )}
      >
        {online ? "Addresses" : "This computer's addresses"}
      </span>
      <ul className="flex flex-col gap-1.5">
        {addresses.map((entry) => (
          <li
            key={entry.address}
            className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground"
          >
            <code
              className={cn(
                "rounded-md border border-border bg-muted/40 px-2 py-1 font-mono text-xs",
                online ? "text-foreground" : "text-muted-foreground",
              )}
            >
              {shareAddressLabel(entry.address, controlPort)}
            </code>
            <span>
              {linkKindLabel(entry.kind)}
              {entry.speedMbps ? ` · ${formatMbps(entry.speedMbps)}` : ""}
            </span>
            {entry.public ? (
              <span className="text-destructive">Public internet address</span>
            ) : null}
          </li>
        ))}
      </ul>
      {online ? (
        <span className="text-xs text-muted-foreground leading-snug">
          Enter one of these on the other computer. Wired and Thunderbolt
          addresses are much faster than Wi-Fi.
        </span>
      ) : null}
    </div>
  );
}

function deviceLine(device: ClusterDevice): string {
  const name = device.description || device.name;
  if (device.freeMib !== null && device.totalMib !== null) {
    return `${name} · ${formatMib(device.freeMib)} free of ${formatMib(device.totalMib)}`;
  }
  return device.totalMib === null
    ? name
    : `${name} · ${formatMib(device.totalMib)}`;
}

function DeviceList({ devices }: { devices: ClusterDevice[] }) {
  if (devices.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1.5 border-t border-border/60 p-4">
      <span className="text-sm font-medium text-foreground">
        {devices.length === 1 ? "Shared GPU" : "Shared GPUs"}
      </span>
      <ul className="flex flex-col gap-1">
        {devices.map((device) => (
          <li key={device.name} className="text-xs text-muted-foreground">
            {deviceLine(device)}
          </li>
        ))}
      </ul>
    </div>
  );
}

function transferLine(share: ClusterShare): string | null {
  const transfer = share.lastTransfer;
  if (!transfer || transfer.bytesIn <= 0) {
    return null;
  }
  const rate = transferMbps(transfer.bytesIn, transfer.seconds);
  const suffix = rate === null ? "" : ` (${formatMbps(rate)})`;
  return `Last load received ${formatBytes(transfer.bytesIn)} in ${formatSeconds(transfer.seconds)} s${suffix}`;
}

function ShareActivity({
  share,
  clockOffsetSeconds,
}: {
  share: ClusterShare;
  clockOffsetSeconds: number;
}) {
  const { session, rejected } = share;
  const transfer = transferLine(share);
  if (!(session || transfer || rejected)) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1 border-t border-border/60 px-4 py-2.5">
      {session ? (
        <p className="text-sm font-medium text-foreground">
          In use by {session.head ?? session.peer ?? "another computer"}
          {session.since === null
            ? ""
            : ` since ${formatClock(session.since, clockOffsetSeconds)}`}
        </p>
      ) : null}
      {transfer ? (
        <p className="text-xs text-muted-foreground">{transfer}</p>
      ) : null}
      {rejected ? (
        <p className="text-xs text-muted-foreground">
          Blocked a connection from {rejected.peer ?? "an unknown address"}
        </p>
      ) : null}
    </div>
  );
}

function PairedHeads({
  heads,
  now,
  busy,
  onRevoke,
}: {
  heads: ClusterPairedHead[];
  now: number;
  busy: ReadonlySet<string>;
  onRevoke: (headId: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5 border-t border-border/60 p-4">
      <span className="text-sm font-medium text-foreground">
        Computers that can use this one
      </span>
      {heads.length === 0 ? (
        <span className="text-xs text-muted-foreground">None yet.</span>
      ) : (
        <ul className="flex flex-col divide-y divide-border/60">
          {heads.map((head) => {
            const revoking = busy.has(`revoke:${head.id}`);
            const detail = [
              head.lastAddress,
              head.lastSeen === null
                ? null
                : `last seen ${formatAgo(head.lastSeen, now)}`,
            ]
              .filter(Boolean)
              .join(" · ");
            return (
              <li
                key={head.id}
                className="flex items-center justify-between gap-3 py-2"
              >
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="truncate text-sm text-foreground">
                    {head.name ?? "Unsloth"}
                  </span>
                  {detail ? (
                    <span className="text-xs text-muted-foreground">
                      {detail}
                    </span>
                  ) : null}
                </div>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="shrink-0"
                  disabled={revoking}
                  onClick={() => onRevoke(head.id)}
                >
                  {revoking ? "Revoking…" : "Revoke"}
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function TensorCacheControls({
  share,
  initial,
  cluster,
}: {
  share: ClusterShare | null;
  initial: number | null;
  cluster: ClusterController;
}) {
  const [draft, setDraft] = useState(initial === null ? "" : String(initial));
  const [error, setError] = useState<string | null>(null);
  const locked =
    share === null ||
    cluster.busy.has("cache-cap") ||
    cluster.busy.has("cache-clear");
  const value = Number(draft);
  const invalid =
    draft.trim() === "" ||
    !Number.isFinite(value) ||
    value <= 0 ||
    value > 100000;

  const submit = async (key: string, request: () => Promise<ClusterState>) => {
    setError(null);
    const failure = await cluster.run(key, request);
    if (failure) {
      setError(clusterErrorMessage(failure, true));
    }
  };

  return (
    <div className="flex flex-col items-end gap-1.5">
      <span className="text-xs text-muted-foreground tabular-nums">
        {share
          ? `${formatBytes(share.cache.bytes)} of ${formatBytes(share.cache.capBytes)} used`
          : ""}
      </span>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <Input
          type="number"
          min={1}
          step={1}
          value={draft}
          disabled={locked}
          aria-label="Tensor cache limit in GB"
          aria-invalid={draft !== "" && invalid}
          className="h-8 w-20"
          onChange={(event) => {
            setDraft(event.target.value);
            setError(null);
          }}
        />
        <span className="text-xs text-muted-foreground">GB</span>
        <Button
          type="button"
          size="sm"
          disabled={locked || invalid || value === initial}
          onClick={() =>
            submit("cache-cap", () =>
              updateClusterShareSettings({ cacheCapGib: value }),
            )
          }
        >
          Save
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={locked || share?.cacheCapGib === null}
          onClick={() =>
            submit("cache-cap", () =>
              updateClusterShareSettings({ resetCacheCap: true }),
            )
          }
        >
          Default
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={locked || share?.cache.bytes === 0}
          onClick={() => submit("cache-clear", clearClusterShareCache)}
        >
          Clear
        </Button>
      </div>
      {error ? (
        <output className="block text-xs leading-snug text-destructive">
          {error}
        </output>
      ) : null}
    </div>
  );
}

function ShareSettingsRows({
  share,
  cluster,
  act,
}: {
  share: ClusterShare | null;
  cluster: ClusterController;
  act: ShareAct;
}) {
  const cacheCap = share
    ? (share.cacheCapGib ??
      Math.round(bytesToGib(share.cache.capBytes) * 10) / 10)
    : null;
  return (
    <div className="border-t border-border/60 px-4 py-1">
      <SettingsRow
        label="Direct connection"
        description="Fastest, and required for RDMA over Thunderbolt 5, but anyone who can reach this computer can use its GPU. Only turn on for a direct cable or a network you trust."
      >
        <Switch
          checked={share?.direct ?? false}
          disabled={
            share === null ||
            share.state === "starting" ||
            cluster.busy.has("direct")
          }
          onCheckedChange={(direct) =>
            act("direct", () => updateClusterShareSettings({ direct }))
          }
          aria-label="Direct connection"
          className="data-checked:bg-destructive"
        />
      </SettingsRow>
      <SettingsRow
        label="Tensor cache"
        description="Other computers' model layers are kept here so the next load skips the network copy."
        alignTop={true}
      >
        <TensorCacheControls
          key={cacheCap ?? "unknown"}
          share={share}
          initial={cacheCap}
          cluster={cluster}
        />
      </SettingsRow>
    </div>
  );
}

function shareMessage(
  share: ClusterShare | null,
  requestError: string | null,
): { text: string; destructive: boolean } | null {
  if (requestError) {
    return { text: requestError, destructive: true };
  }
  if (share && !share.available && share.state !== "online") {
    return { text: UNAVAILABLE_MESSAGE, destructive: false };
  }
  if (share?.state === "error" && share.error) {
    return { text: clusterErrorMessage(share.error, true), destructive: true };
  }
  return null;
}

function ShareHeader({
  share,
  busy,
  act,
}: {
  share: ClusterShare | null;
  busy: ReadonlySet<string>;
  act: ShareAct;
}) {
  const online = share?.state === "online";
  const starting = busy.has("share-start") || share?.state === "starting";
  const stopping = busy.has("share-stop");
  let label = online ? "Stop sharing" : "Share";
  if (starting) {
    label = "Starting…";
  } else if (stopping) {
    label = "Stopping…";
  }
  const disabled =
    share === null || starting || stopping || !(online || share.available);

  return (
    <div className="flex items-center justify-between gap-4 bg-muted/30 p-4">
      <div className="flex min-w-0 items-start gap-3">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
          <HugeiconsIcon
            icon={ComputerArrowUpIcon}
            className="size-4 text-foreground"
          />
        </div>
        <div className="flex min-w-0 flex-col gap-0.5">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-base font-semibold font-heading text-foreground">
              Share this computer
            </h2>
            <ShareStatus share={share} />
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            Let your other Unsloth computers use this computer's GPU memory.
          </p>
        </div>
      </div>
      <Button
        type="button"
        size="sm"
        variant={online ? "outline" : "default"}
        className="min-w-20 shrink-0"
        disabled={disabled}
        onClick={() =>
          online
            ? act("share-stop", stopClusterShare)
            : act("share-start", startClusterShare)
        }
      >
        {label}
      </Button>
    </div>
  );
}

export function ClusterShareSection({
  cluster,
}: {
  cluster: ClusterController;
}) {
  const share = cluster.state?.share ?? null;
  const clockOffsetSeconds = cluster.state?.clockOffsetSeconds ?? 0;
  const [requestError, setRequestError] = useState<string | null>(null);
  const now = serverNowMs(clockOffsetSeconds, useNow());
  const message = shareMessage(share, requestError);

  const act: ShareAct = async (key, request) => {
    setRequestError(null);
    const failure = await cluster.run(key, request);
    if (failure) {
      setRequestError(clusterErrorMessage(failure, true));
    }
  };

  return (
    <section
      data-settings-label="Share this computer"
      className="overflow-hidden rounded-lg border border-border/70"
    >
      <ShareHeader share={share} busy={cluster.busy} act={act} />

      {message ? (
        <p
          className={cn(
            "border-t border-border/60 px-4 py-2.5 text-xs leading-snug",
            message.destructive ? "text-destructive" : "text-muted-foreground",
          )}
        >
          {message.text}
        </p>
      ) : null}

      {share?.state === "online" ? (
        <>
          <PairingPanel
            pairing={share.pairing}
            now={now}
            busy={cluster.busy.has("pairing-code")}
            onNewCode={() => act("pairing-code", newClusterPairingCode)}
          />
          <AddressList
            addresses={share.addresses}
            controlPort={share.controlPort}
            online={true}
          />
          <DeviceList devices={share.devices} />
        </>
      ) : null}
      {share && share.state !== "online" ? (
        <AddressList
          addresses={share.addresses}
          controlPort={share.controlPort}
          online={false}
        />
      ) : null}
      {share ? (
        <ShareActivity share={share} clockOffsetSeconds={clockOffsetSeconds} />
      ) : null}

      <PairedHeads
        heads={share?.pairedHeads ?? []}
        now={now}
        busy={cluster.busy}
        onRevoke={(headId) =>
          act(`revoke:${headId}`, () => revokeClusterHead(headId))
        }
      />

      <ShareSettingsRows share={share} cluster={cluster} act={act} />
    </section>
  );
}
