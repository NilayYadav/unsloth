// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ClusterMode = "off" | "auto";
export type ClusterLinkKind =
  | "thunderbolt"
  | "ethernet"
  | "wifi"
  | "vpn"
  | "unknown";
export type ClusterNodeStatus = "online" | "busy" | "offline" | "unpaired";
export type ClusterShareState = "off" | "starting" | "online" | "error";
export type ClusterIncompatibleReason =
  | "family_unknown"
  | "unsupported_family"
  | "family_mismatch"
  | "rpc_version";

export type ClusterAddress = {
  address: string;
  interfaceName: string | null;
  kind: ClusterLinkKind;
  speedMbps: number | null;
  public: boolean;
};

export type ClusterDevice = {
  name: string;
  description: string;
  totalMib: number | null;
  freeMib: number | null;
};

export type ClusterLink = {
  address: string | null;
  kind: ClusterLinkKind;
  remoteKind: ClusterLinkKind | null;
  localKind: ClusterLinkKind | null;
  localAddress: string | null;
  speedMbps: number | null;
  rttMs: number | null;
  throughputMbps: number | null;
  measuredAt: number | null;
};

export type ClusterNode = {
  id: string;
  name: string;
  host: string;
  enteredHost: string | null;
  controlPort: number | null;
  rpcPort: number | null;
  direct: boolean;
  enabled: boolean;
  status: ClusterNodeStatus;
  lastError: string | null;
  lastSeen: number | null;
  pairedAt: number | null;
  platform: string | null;
  machine: string | null;
  family: string | null;
  build: string | null;
  rpcProto: number[] | null;
  transport: string | null;
  addresses: ClusterAddress[];
  devices: ClusterDevice[];
  freeMib: number | null;
  totalMib: number | null;
  link: ClusterLink;
  compatible: boolean;
  incompatibleReason: ClusterIncompatibleReason | null;
  warning: "build_mismatch" | null;
  score: number | null;
  effectiveMbps: number | null;
  loadSecondsPerGib: number | null;
};

export type ClusterActiveNode = {
  id: string;
  name: string | null;
  host: string | null;
  rpcPort: number | null;
  usableMib: number | null;
};

export type ClusterActive = {
  nodes: ClusterActiveNode[];
  endpoints: string[];
  deficitMib: number;
  capacityMib: number;
  startedAt: number | null;
  failure: string | null;
};

export type ClusterLocal = {
  nodeId: string | null;
  name: string | null;
  family: string | null;
  build: string | null;
  rpcProto: number[] | null;
};

export type ClusterHead = {
  mode: ClusterMode;
  local: ClusterLocal;
  nodes: ClusterNode[];
  active: ClusterActive | null;
};

export type ClusterPairing = { code: string; expiresAt: number };

export type ClusterSession = {
  peer: string | null;
  since: number | null;
  head: string | null;
};

export type ClusterTransfer = {
  peer: string | null;
  head: string | null;
  bytesIn: number;
  bytesOut: number;
  seconds: number;
  at: number | null;
};

export type ClusterRejected = {
  peer: string | null;
  reason: string | null;
  at: number | null;
};

export type ClusterLease = {
  headId: string;
  name: string | null;
  address: string | null;
  expiresAt: number | null;
};

export type ClusterPairedHead = {
  id: string;
  name: string | null;
  pairedAt: number | null;
  lastSeen: number | null;
  lastAddress: string | null;
};

export type ClusterShare = {
  available: boolean;
  state: ClusterShareState;
  error: string | null;
  autoStart: boolean;
  direct: boolean;
  controlPort: number | null;
  rpcPort: number | null;
  cache: { bytes: number; files: number; capBytes: number };
  cacheCapGib: number | null;
  family: string | null;
  build: string | null;
  devices: ClusterDevice[];
  transport: string | null;
  addresses: ClusterAddress[];
  pairing: ClusterPairing | null;
  session: ClusterSession | null;
  lastTransfer: ClusterTransfer | null;
  rejected: ClusterRejected | null;
  leases: ClusterLease[];
  startedAt: number | null;
  pairedHeads: ClusterPairedHead[];
};

export type ClusterState = {
  now: number | null;
  clockOffsetSeconds: number;
  head: ClusterHead;
  share: ClusterShare;
};

export const CLUSTER_POLL_MS = 3000;
export const CLUSTER_STARTING_POLL_MS = 1000;
export const CLUSTER_DEFAULT_CONTROL_PORT = 50051;

const BYTES_PER_GIB = 1024 ** 3;
const LINK_KINDS = [
  "thunderbolt",
  "ethernet",
  "wifi",
  "vpn",
  "unknown",
] as const;
const NODE_STATUSES = ["online", "busy", "offline", "unpaired"] as const;
const SHARE_STATES = ["off", "starting", "online", "error"] as const;
const INCOMPATIBLE_REASONS = [
  "family_unknown",
  "unsupported_family",
  "family_mismatch",
  "rpc_version",
] as const;
const PORT_IN_USE_RE = /^port_in_use:(\d+)$/;
const WHITESPACE_RE = /\s/;

type Raw = Record<string, unknown>;

function record(value: unknown): Raw | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Raw)
    : null;
}

function records(value: unknown): Raw[] {
  return Array.isArray(value)
    ? value.map(record).filter((entry): entry is Raw => entry !== null)
    : [];
}

function text(value: unknown): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function oneOf<T extends string>(
  value: unknown,
  options: readonly T[],
): T | null {
  return options.find((option) => option === value) ?? null;
}

function numbers(value: unknown): number[] | null {
  return Array.isArray(value) &&
    value.every((entry) => typeof entry === "number")
    ? (value as number[])
    : null;
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((entry): entry is string => typeof entry === "string")
    : [];
}

function normalizeAddresses(value: unknown): ClusterAddress[] {
  return records(value).flatMap((raw) => {
    const address = text(raw.address);
    if (!address) {
      return [];
    }
    return [
      {
        address,
        interfaceName: text(raw.interface),
        kind: oneOf(raw.kind, LINK_KINDS) ?? "unknown",
        speedMbps: finite(raw.speed_mbps),
        public: raw.public === true,
      },
    ];
  });
}

function normalizeDevices(value: unknown): ClusterDevice[] {
  return records(value).map((raw) => ({
    name: text(raw.name) ?? "GPU",
    description: text(raw.description) ?? "",
    totalMib: finite(raw.total_mib),
    freeMib: finite(raw.free_mib),
  }));
}

function normalizeLink(value: unknown): ClusterLink {
  const raw = record(value) ?? {};
  return {
    address: text(raw.address),
    kind: oneOf(raw.kind, LINK_KINDS) ?? "unknown",
    remoteKind: oneOf(raw.remote_kind, LINK_KINDS),
    localKind: oneOf(raw.local_kind, LINK_KINDS),
    localAddress: text(raw.local_address),
    speedMbps: finite(raw.speed_mbps),
    rttMs: finite(raw.rtt_ms),
    throughputMbps: finite(raw.throughput_mbps),
    measuredAt: finite(raw.measured_at),
  };
}

function normalizeNode(raw: Raw): ClusterNode | null {
  const id = text(raw.id);
  if (!id) {
    return null;
  }
  const host = text(raw.host) ?? text(raw.entered_host) ?? "";
  return {
    id,
    name: text(raw.name) ?? (host || id),
    host,
    enteredHost: text(raw.entered_host),
    controlPort: finite(raw.control_port),
    rpcPort: finite(raw.rpc_port),
    direct: raw.direct === true,
    enabled: raw.enabled !== false,
    status: oneOf(raw.status, NODE_STATUSES) ?? "offline",
    lastError: text(raw.last_error),
    lastSeen: finite(raw.last_seen),
    pairedAt: finite(raw.paired_at),
    platform: text(raw.platform),
    machine: text(raw.machine),
    family: text(raw.family),
    build: text(raw.build),
    rpcProto: numbers(raw.rpc_proto),
    transport: text(raw.transport),
    addresses: normalizeAddresses(raw.addresses),
    devices: normalizeDevices(raw.devices),
    freeMib: finite(raw.free_mib),
    totalMib: finite(raw.total_mib),
    link: normalizeLink(raw.link),
    compatible: raw.compatible !== false,
    incompatibleReason: oneOf(raw.incompatible_reason, INCOMPATIBLE_REASONS),
    warning: raw.warning === "build_mismatch" ? "build_mismatch" : null,
    score: finite(raw.score),
    effectiveMbps: finite(raw.effective_mbps),
    loadSecondsPerGib: finite(raw.load_seconds_per_gib),
  };
}

function normalizeActive(value: unknown): ClusterActive | null {
  const raw = record(value);
  if (!raw) {
    return null;
  }
  return {
    nodes: records(raw.nodes).map((node) => ({
      id: text(node.id) ?? "",
      name: text(node.name),
      host: text(node.host),
      rpcPort: finite(node.rpc_port),
      usableMib: finite(node.usable_mib),
    })),
    endpoints: strings(raw.endpoints),
    deficitMib: finite(raw.deficit_mib) ?? 0,
    capacityMib: finite(raw.capacity_mib) ?? 0,
    startedAt: finite(raw.started_at),
    failure: text(raw.failure),
  };
}

function normalizeHead(value: unknown): ClusterHead {
  const raw = record(value) ?? {};
  const local = record(raw.local) ?? {};
  return {
    mode: raw.mode === "auto" ? "auto" : "off",
    local: {
      nodeId: text(local.node_id),
      name: text(local.name),
      family: text(local.family),
      build: text(local.build),
      rpcProto: numbers(local.rpc_proto),
    },
    nodes: records(raw.nodes)
      .map(normalizeNode)
      .filter((node): node is ClusterNode => node !== null),
    active: normalizeActive(raw.active),
  };
}

function normalizePairing(value: unknown): ClusterPairing | null {
  const raw = record(value);
  const code = text(raw?.code);
  const expiresAt = finite(raw?.expires_at);
  return code && expiresAt !== null ? { code, expiresAt } : null;
}

function normalizeSession(value: unknown): ClusterSession | null {
  const raw = record(value);
  return raw
    ? { peer: text(raw.peer), since: finite(raw.since), head: text(raw.head) }
    : null;
}

function normalizeTransfer(value: unknown): ClusterTransfer | null {
  const raw = record(value);
  if (!raw) {
    return null;
  }
  return {
    peer: text(raw.peer),
    head: text(raw.head),
    bytesIn: finite(raw.bytes_in) ?? 0,
    bytesOut: finite(raw.bytes_out) ?? 0,
    seconds: finite(raw.seconds) ?? 0,
    at: finite(raw.at),
  };
}

function normalizeRejected(value: unknown): ClusterRejected | null {
  const raw = record(value);
  return raw
    ? { peer: text(raw.peer), reason: text(raw.reason), at: finite(raw.at) }
    : null;
}

function normalizeLeases(value: unknown): ClusterLease[] {
  return records(value).flatMap((raw) => {
    const headId = text(raw.head_id);
    return headId
      ? [
          {
            headId,
            name: text(raw.name),
            address: text(raw.address),
            expiresAt: finite(raw.expires_at),
          },
        ]
      : [];
  });
}

function normalizePairedHeads(value: unknown): ClusterPairedHead[] {
  return records(value).flatMap((raw) => {
    const id = text(raw.id);
    return id
      ? [
          {
            id,
            name: text(raw.name),
            pairedAt: finite(raw.paired_at),
            lastSeen: finite(raw.last_seen),
            lastAddress: text(raw.last_address),
          },
        ]
      : [];
  });
}

function normalizeShare(value: unknown): ClusterShare {
  const raw = record(value) ?? {};
  const cache = record(raw.cache) ?? {};
  return {
    available: raw.available !== false,
    state: oneOf(raw.state, SHARE_STATES) ?? "off",
    error: text(raw.error),
    autoStart: raw.auto_start === true,
    direct: raw.direct === true,
    controlPort: finite(raw.control_port),
    rpcPort: finite(raw.rpc_port),
    cache: {
      bytes: finite(cache.bytes) ?? 0,
      files: finite(cache.files) ?? 0,
      capBytes: finite(cache.cap_bytes) ?? 0,
    },
    cacheCapGib: finite(raw.cache_cap_gib),
    family: text(raw.family),
    build: text(raw.build),
    devices: normalizeDevices(raw.devices),
    transport: text(raw.transport),
    addresses: normalizeAddresses(raw.addresses),
    pairing: normalizePairing(raw.pairing),
    session: normalizeSession(raw.session),
    lastTransfer: normalizeTransfer(raw.last_transfer),
    rejected: normalizeRejected(raw.rejected),
    leases: normalizeLeases(raw.leases),
    startedAt: finite(raw.started_at),
    pairedHeads: normalizePairedHeads(raw.paired_heads),
  };
}

export function normalizeClusterState(
  value: unknown,
  receivedAtMs = Date.now(),
): ClusterState {
  const raw = record(value) ?? {};
  const now = finite(raw.now);
  return {
    now,
    clockOffsetSeconds: now === null ? 0 : now - receivedAtMs / 1000,
    head: normalizeHead(raw.head),
    share: normalizeShare(raw.share),
  };
}

export function serverNowMs(
  clockOffsetSeconds: number,
  browserNowMs: number,
): number {
  return browserNowMs + clockOffsetSeconds * 1000;
}

function trimNumber(value: number): string {
  return value >= 10
    ? String(Math.round(value))
    : String(Math.round(value * 10) / 10);
}

export function bytesToGib(bytes: number): number {
  return bytes / BYTES_PER_GIB;
}

export function formatGib(gib: number): string {
  return `${trimNumber(Math.max(0, gib))} GB`;
}

export function formatMib(mib: number): string {
  return formatGib(mib / 1024);
}

export function formatBytes(bytes: number): string {
  return formatGib(bytesToGib(bytes));
}

export function formatMbps(mbps: number): string {
  return mbps >= 1000
    ? `${trimNumber(mbps / 1000)} Gbps`
    : `${trimNumber(Math.max(0, mbps))} Mbps`;
}

export function formatRtt(ms: number): string {
  return ms < 0.1 ? "< 0.1 ms" : `${trimNumber(ms)} ms`;
}

export function formatSeconds(seconds: number): string {
  return trimNumber(Math.max(0, seconds));
}

export function transferMbps(bytes: number, seconds: number): number | null {
  return seconds > 0 && bytes > 0 ? (bytes * 8) / 1_000_000 / seconds : null;
}

export function formatCountdown(seconds: number): string {
  const total = Math.max(0, Math.ceil(seconds));
  const minutes = String(Math.floor(total / 60)).padStart(2, "0");
  return `${minutes}:${String(total % 60).padStart(2, "0")}`;
}

export function formatAgo(epochSeconds: number, nowMs: number): string {
  const seconds = Math.max(0, nowMs / 1000 - epochSeconds);
  if (seconds < 60) {
    return "just now";
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes} min ago`;
  }
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours} h ago` : `${Math.floor(hours / 24)} d ago`;
}

export function formatClock(
  epochSeconds: number,
  clockOffsetSeconds = 0,
): string {
  const local = epochSeconds - clockOffsetSeconds;
  return new Date(local * 1000).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

const LINK_KIND_LABEL: Record<ClusterLinkKind, string> = {
  thunderbolt: "Thunderbolt",
  ethernet: "Ethernet",
  wifi: "Wi-Fi",
  vpn: "VPN",
  unknown: "Network",
};

export function linkKindLabel(kind: ClusterLinkKind | null): string {
  return LINK_KIND_LABEL[kind ?? "unknown"];
}

const FAMILY_LABEL: Record<string, string> = {
  metal: "Apple Silicon",
  cuda: "NVIDIA",
  rocm: "AMD",
  vulkan: "Vulkan",
  cpu: "CPU",
};

export function familyLabel(family: string | null): string | null {
  return family !== null && Object.hasOwn(FAMILY_LABEL, family)
    ? FAMILY_LABEL[family]
    : null;
}

export function deviceSummary(
  devices: ClusterDevice[],
  freeMib: number | null,
): string | null {
  const counts = new Map<string, number>();
  for (const device of devices) {
    const label = device.description || device.name;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const names = [...counts].map(([label, count]) =>
    count > 1 ? `${count} × ${label}` : label,
  );
  const deviceFree = devices.map((device) => device.freeMib);
  const free =
    freeMib ??
    (deviceFree.length > 0 && deviceFree.every((mib) => mib !== null)
      ? deviceFree.reduce<number>((sum, mib) => sum + (mib ?? 0), 0)
      : null);
  const parts = names.length > 0 ? [names.join(", ")] : [];
  if (free !== null) {
    parts.push(`${formatMib(free)} free`);
  }
  return parts.length > 0 ? parts.join(" · ") : null;
}

export function shareAddressLabel(
  address: string,
  controlPort: number | null,
): string {
  if (controlPort === null || controlPort === CLUSTER_DEFAULT_CONTROL_PORT) {
    return address;
  }
  return address.includes(":")
    ? `[${address}]:${controlPort}`
    : `${address}:${controlPort}`;
}

export function incompatibleReasonMessage(
  reason: ClusterIncompatibleReason,
  nodeFamily: string | null,
  localFamily: string | null,
): string {
  switch (reason) {
    case "family_unknown":
      return "Unsloth can't tell what kind of GPU one of these computers has, so it won't use this one.";
    case "unsupported_family":
      return "Only Apple Silicon, NVIDIA, AMD and Vulkan GPUs can join a cluster.";
    case "family_mismatch": {
      const theirs = familyLabel(nodeFamily);
      const ours = familyLabel(localFamily);
      return theirs && ours
        ? `It has ${theirs} GPUs and this computer has ${ours}. Every computer needs the same kind of GPU.`
        : "Its GPUs are a different kind from this computer's. Every computer needs the same kind of GPU.";
    }
    case "rpc_version":
      return "Its llama.cpp version can't talk to this computer's. Update Unsloth on both computers.";
    default:
      return "This computer can't join the cluster.";
  }
}

function knownErrorMessage(code: string): string | null {
  switch (code) {
    case "unreachable":
      return "Couldn't reach that computer. Check the address and that both computers are on the same network.";
    case "timeout":
      return "That computer took too long to answer. Check the address and your network.";
    case "not_unsloth":
      return "Something answered at that address, but it isn't Unsloth. Check the address.";
    case "not_sharing":
      return "That computer isn't sharing. Turn on Share this computer there, then try again.";
    case "code_invalid":
      return "That pairing code isn't right. Check the code shown on the other computer.";
    case "code_expired":
      return "That pairing code expired. Use the new code shown on the other computer.";
    case "self_pairing":
      return "That address is this computer. Enter the address of another computer.";
    case "busy":
      return "That computer is busy with another computer's model. Try again when it's done.";
    case "not_found":
      return "That computer is no longer in the list.";
    case "address_required":
      return "Enter the other computer's address.";
    case "port_invalid":
      return "That port isn't valid. Use a number from 1 to 65535.";
    case "bad_response":
      return "That computer sent a reply Unsloth didn't understand. Update Unsloth on both computers.";
    case "rpc_server_missing":
      return "This install doesn't include ggml-rpc-server yet. Update Unsloth to share this computer.";
    case "control_listener_failed":
      return "Couldn't open the sharing port. Check the logs for details.";
    case "rpc_server_exited":
      return "The GPU sharing server stopped unexpectedly. Check the logs, then try again.";
    case "token_rejected":
      return "This computer no longer trusts this one. Remove it and pair again.";
    case "identity_changed":
      return "A different computer now answers at this address. Remove it and pair again.";
    case "rpc_port_unreachable":
      return "Reached that computer, but not its GPU port. A firewall may be blocking it.";
    case "load_failed":
      return "The last model load that used this computer failed.";
    default:
      return null;
  }
}

export function clusterErrorMessage(code: string, local = false): string {
  if (local && code === "busy") {
    return "Another computer is using this one right now. Try again when it's done.";
  }
  const portInUse = PORT_IN_USE_RE.exec(code);
  if (portInUse) {
    return `Port ${portInUse[1]} is already in use. Close the app using it, then try again.`;
  }
  const known = knownErrorMessage(code);
  if (known) {
    return known;
  }
  return WHITESPACE_RE.test(code)
    ? code
    : `Something went wrong (${code}). Check the logs for details.`;
}
