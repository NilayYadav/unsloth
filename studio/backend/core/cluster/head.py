# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.cluster import planner, store
from core.cluster.client import WorkerClient, WorkerError, endpoint, probe_data_path, probe_throughput
from core.cluster.netinfo import KIND_RANK, route_to, tcp_rtt_ms
from core.cluster.rpc_server import (
    build_identity,
    detect_local_family,
    find_llama_server_binary,
    install_marker,
    port_is_listening,
)
from core.cluster.worker import host_name
from loggers import get_logger

logger = get_logger(__name__)

LEASE_TTL_S = 90.0
LEASE_RENEW_S = 20.0
REFRESH_TIMEOUT_S = 2.0
PLAN_REFRESH_TIMEOUT_S = 1.5
DEAD_CHECKS_BEFORE_RELEASE = 3
STATUS_REFRESH_S = 10.0
LOCAL_CACHE_S = 60.0

_INFO_FIELDS = (
    "name",
    "platform",
    "machine",
    "family",
    "build",
    "rpc_proto",
    "transport",
    "control_port",
    "rpc_port",
    "direct",
    "addresses",
    "devices",
    "free_mib",
    "total_mib",
)

_RPC_FAILURES = (
    ("RPC server version mismatch", "version"),
    ("RPC handshake failed", "handshake"),
    ("Remote RPC server crashed", "crashed"),
    ("Failed to parse endpoint", "endpoint"),
)


class ClusterError(RuntimeError):
    def __init__(self, code: str, status: int = 400):
        super().__init__(code)
        self.code = code
        self.status = status


def parse_address(value: str) -> tuple[str, int]:
    text = (value or "").strip()
    for prefix in ("http://", "https://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    text = text.split("/", 1)[0]
    if not text:
        raise ClusterError("address_required")
    port_text = ""
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port_text = rest.lstrip(":")
    elif text.count(":") == 1:
        host, port_text = text.split(":")
    else:
        host = text
    if not host:
        raise ClusterError("address_required")
    if not port_text:
        return host, store.DEFAULT_CONTROL_PORT
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ClusterError("port_invalid") from exc
    if not 1 <= port <= 65535:
        raise ClusterError("port_invalid")
    return host, port


def public_node(node: dict, local: dict) -> dict:
    view = {k: v for k, v in node.items() if k != "token"}
    compatible, reason, warning = planner.compatibility(local, node)
    link = node.get("link") or {}
    mbps = planner.link_mbps(link) if link else None
    view.update(
        compatible = compatible,
        incompatible_reason = reason,
        warning = warning,
        score = round(planner.node_score(node), 1),
        effective_mbps = round(mbps, 1) if mbps else None,
        load_seconds_per_gib = round(8 * 1024 / mbps, 1) if mbps else None,
    )
    return view


def cluster_failure_message(output: str, attachment: "ClusterAttachment") -> Optional[str]:
    text = output or ""
    kind = next((k for marker, k in _RPC_FAILURES if marker in text), None)
    if kind is None and any(f"Failed to connect to {ep}" in text for ep in attachment.endpoints):
        kind = "connect"
    if kind is None:
        return None
    names = attachment.names
    if kind == "version":
        return (
            f"The cluster computer ({names}) runs a different llama.cpp version. "
            "Update Unsloth on every computer, then load the model again."
        )
    if kind == "crashed":
        return (
            f"A cluster computer ({names}) stopped responding while the model was loading. "
            "Check that it is still sharing and on the network, or turn it off in Settings > Cluster."
        )
    return (
        f"Could not reach the cluster computer ({names}). Check that it is still sharing and "
        "on the same network, or turn it off in Settings > Cluster."
    )


@dataclass
class ClusterAttachment:
    nodes: list[dict]
    endpoints: list[str]
    deficit_mib: int
    capacity_mib: int
    started_at: float = field(default_factory = time.time)
    stop_event: threading.Event = field(default_factory = threading.Event)
    failure: Optional[str] = None

    @property
    def names(self) -> str:
        return ", ".join(str(n.get("name") or n.get("host")) for n in self.nodes)

    def summary(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n["id"],
                    "name": n.get("name"),
                    "host": n.get("host"),
                    "rpc_port": n.get("rpc_port"),
                    "usable_mib": n.get("usable_mib"),
                }
                for n in self.nodes
            ],
            "endpoints": list(self.endpoints),
            "deficit_mib": self.deficit_mib,
            "capacity_mib": self.capacity_mib,
            "started_at": self.started_at,
            "failure": self.failure,
        }


class ClusterHead:
    def __init__(self):
        self._lock = threading.RLock()
        # a link test releases its short lease; a load must not lease in between
        self._link_lock = threading.Lock()
        self._attachment: Optional[ClusterAttachment] = None
        self._local: Optional[tuple[float, dict]] = None
        self._last_refresh = 0.0
        self._refreshing = False

    def local(self) -> dict:
        cached = self._local
        now = time.monotonic()
        if cached is not None and now - cached[0] < LOCAL_CACHE_S:
            return cached[1]
        binary = find_llama_server_binary()
        value = {
            "node_id": store.identity()["node_id"],
            "name": host_name(),
            "family": detect_local_family(binary),
            "build": build_identity(install_marker(binary)),
            "rpc_proto": store.local_rpc_proto(),
        }
        self._local = (now, value)
        return value

    def has_enabled_nodes(self) -> bool:
        return store.mode() == "auto" and any(n.get("enabled", True) and n.get("token") for n in store.nodes())

    def active(self) -> Optional[ClusterAttachment]:
        return self._attachment

    def status(self) -> dict:
        self._maybe_background_refresh()
        local = self.local()
        attachment = self._attachment
        return {
            "mode": store.mode(),
            "local": local,
            "nodes": [public_node(n, local) for n in store.nodes()],
            "active": attachment.summary() if attachment else None,
        }

    def _maybe_background_refresh(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._refreshing or now - self._last_refresh < STATUS_REFRESH_S or not store.nodes():
                return
            self._refreshing = True
            self._last_refresh = now

        def run():
            try:
                self.refresh_nodes(store.nodes(), REFRESH_TIMEOUT_S)
            except Exception:
                logger.debug("cluster refresh failed", exc_info = True)
            finally:
                with self._lock:
                    self._refreshing = False

        threading.Thread(target = run, name = "cluster-refresh", daemon = True).start()

    def _save_fields(self, node_id: str, fields: dict) -> Optional[dict]:
        return store.update_node(node_id, lambda n: {**n, **fields})

    def refresh_nodes(self, nodes: list[dict], timeout: float) -> list[dict]:
        if not nodes:
            return []
        with ThreadPoolExecutor(max_workers = min(8, len(nodes))) as pool:
            return [n for n in pool.map(lambda n: self._refresh(n, timeout), nodes) if n is not None]

    def _refresh(self, node: dict, timeout: float) -> Optional[dict]:
        try:
            info = WorkerClient(node["host"], node["control_port"], node.get("token"), timeout).info()
        except WorkerError as exc:
            if exc.status == 401:
                return self._save_fields(node["id"], {"status": "unpaired", "last_error": "token_rejected"})
            relocated = self._relocate(node, timeout)
            if relocated is None:
                return self._save_fields(node["id"], {"status": "offline", "last_error": exc.code})
            node = relocated
            try:
                info = WorkerClient(node["host"], node["control_port"], node.get("token"), timeout).info()
            except WorkerError as retry:
                return self._save_fields(node["id"], {"status": "offline", "last_error": retry.code})
        return self._apply_info(node, info)

    def _apply_info(self, node: dict, info: dict) -> Optional[dict]:
        if info.get("node_id") and info["node_id"] != node["id"]:
            return self._save_fields(node["id"], {"status": "offline", "last_error": "identity_changed"})
        fields = {k: info.get(k) for k in _INFO_FIELDS if k in info}
        link = dict(node.get("link") or {})
        link.update(self._link_for(node["host"], fields.get("addresses") or node.get("addresses") or []))
        fields.update(
            status = "busy" if info.get("busy") else "online",
            last_seen = time.time(),
            last_error = None,
            link = link,
        )
        return self._save_fields(node["id"], fields)

    @staticmethod
    def _link_for(host: str, addresses: list) -> dict:
        remote = next((a for a in addresses if a.get("address") == host), None) or {}
        local = route_to(host) or {}
        kinds = [k for k in (remote.get("kind"), local.get("kind")) if k]
        kind = min(kinds, key = lambda k: KIND_RANK.get(k, 0)) if kinds else "unknown"
        speeds = [s for s in (remote.get("speed_mbps"), local.get("speed_mbps")) if s]
        return {
            "address": host,
            "kind": kind,
            "remote_kind": remote.get("kind"),
            "local_kind": local.get("kind"),
            "local_address": local.get("address"),
            "speed_mbps": min(speeds) if speeds else None,
        }

    def _measure_addresses(self, node: dict, timeout: float) -> list[dict]:
        hosts = [a.get("address") for a in node.get("addresses") or [] if not a.get("public")]
        hosts += [node.get("entered_host"), node.get("host")]
        hosts = [h for h in dict.fromkeys(hosts) if h]
        port = node["control_port"]

        def measure(host):
            link = self._link_for(host, node.get("addresses") or [])
            link["rtt_ms"] = tcp_rtt_ms(host, port, samples = 3, timeout = min(timeout, 0.8))
            return link

        if not hosts:
            return []
        with ThreadPoolExecutor(max_workers = min(8, len(hosts))) as pool:
            return list(pool.map(measure, hosts))

    def _relocate(self, node: dict, timeout: float) -> Optional[dict]:
        best = planner.best_address(self._measure_addresses(node, timeout))
        if best is None:
            return None
        link = {**(node.get("link") or {}), **best}
        return self._save_fields(node["id"], {"host": best["address"], "link": link})

    def pair(self, address: str, code: str) -> dict:
        host, port = parse_address(address)
        client = WorkerClient(host, port, timeout = 4.0)
        try:
            hello = client.hello()
        except WorkerError as exc:
            raise ClusterError("unreachable" if exc.code in ("unreachable", "timeout") else exc.code, 502) from exc
        if not hello.get("sharing"):
            raise ClusterError("not_sharing", 409)
        identity = store.identity()
        try:
            info = client.pair(code, identity["node_id"], host_name())
        except WorkerError as exc:
            raise ClusterError(exc.code, 403 if exc.status == 403 else 502) from exc
        token = info.pop("token", None)
        if not token or not info.get("node_id"):
            raise ClusterError("bad_response", 502)
        existing = next((n for n in store.nodes() if n["id"] == info["node_id"]), None)
        node = {
            "id": info["node_id"],
            "token": token,
            "entered_host": host,
            "host": host,
            "enabled": existing.get("enabled", True) if existing else True,
            "paired_at": time.time(),
            "link": {},
        }
        node.update({k: info.get(k) for k in _INFO_FIELDS if k in info})
        node["control_port"] = int(info.get("control_port") or port)
        best = planner.best_address(self._measure_addresses(node, 2.0))
        if best is not None:
            node["host"] = best["address"]
            node["link"] = best
        else:
            node["link"] = {**self._link_for(host, node.get("addresses") or []), "rtt_ms": tcp_rtt_ms(host, port, 3)}
        node.update(status = "busy" if info.get("busy") else "online", last_seen = time.time(), last_error = None)
        store.upsert_node(node)
        threading.Thread(target = self._background_test, args = (node["id"],), daemon = True).start()
        return public_node(node, self.local())

    def _background_test(self, node_id: str) -> None:
        try:
            self.test(node_id)
        except Exception:
            logger.debug("cluster link test failed", exc_info = True)

    def _node(self, node_id: str) -> dict:
        node = next((n for n in store.nodes() if n["id"] == node_id), None)
        if node is None:
            raise ClusterError("not_found", 404)
        return node

    def set_enabled(self, node_id: str, enabled: bool) -> dict:
        self._node(node_id)
        return public_node(self._save_fields(node_id, {"enabled": bool(enabled)}), self.local())

    def remove(self, node_id: str) -> None:
        node = store.remove_node(node_id)
        if node is None:
            raise ClusterError("not_found", 404)

        def unpair():
            try:
                WorkerClient(node["host"], node["control_port"], node.get("token"), 2.0).unpair()
            except WorkerError:
                pass

        threading.Thread(target = unpair, daemon = True).start()

    def _attached(self, node_id: str) -> bool:
        attachment = self._attachment
        return bool(attachment and any(n["id"] == node_id for n in attachment.nodes))

    def test(self, node_id: str) -> dict:
        node = self._refresh(self._node(node_id), REFRESH_TIMEOUT_S) or self._node(node_id)
        if node.get("status") not in ("online", "busy"):
            return public_node(node, self.local())
        best = planner.best_address(self._measure_addresses(node, 2.0))
        if best is not None and best["address"] != node["host"]:
            node = self._save_fields(node_id, {"host": best["address"]})
        link = dict(node.get("link") or {})
        link.update(self._link_for(node["host"], node.get("addresses") or []))
        link["rtt_ms"] = tcp_rtt_ms(node["host"], node["control_port"], samples = 7)
        with self._link_lock:
            # never flood a link that is carrying a loaded model's tokens
            if not node.get("direct") and node.get("status") == "online" and not self._attached(node_id):
                client = WorkerClient(node["host"], node["control_port"], node.get("token"), REFRESH_TIMEOUT_S)
                try:
                    client.lease(30.0)
                except WorkerError:
                    pass
                else:
                    throughput = probe_throughput(node["host"], node["rpc_port"])
                    if throughput:
                        link["throughput_mbps"] = throughput
                    try:
                        client.release()
                    except WorkerError:
                        pass
        link["measured_at"] = time.time()
        return public_node(self._save_fields(node_id, {"link": link}), self.local())

    def plan_for_load(self, deficit_mib: int, alive: Callable[[], bool]) -> Optional[ClusterAttachment]:
        previous = self.release_active(release = False)
        attachment = None
        try:
            attachment = self._plan(deficit_mib, alive)
            return attachment
        finally:
            if previous is not None:
                kept = {n["id"] for n in attachment.nodes} if attachment else set()
                stale = [n for n in previous.nodes if n["id"] not in kept]
                if stale:
                    with self._link_lock:
                        self._release_nodes(stale, 1.0)

    def _plan(self, deficit_mib: int, alive: Callable[[], bool]) -> Optional[ClusterAttachment]:
        if deficit_mib <= 0 or store.mode() != "auto":
            return None
        local = self.local()
        if local.get("family") not in planner.GPU_FAMILIES:
            return None
        candidates = [n for n in store.nodes() if n.get("enabled", True) and n.get("token")]
        if not candidates:
            return None
        usable = []
        for node in self.refresh_nodes(candidates, PLAN_REFRESH_TIMEOUT_S):
            compatible, reason, _warning = planner.compatibility(local, node)
            if not compatible:
                logger.info("Cluster: skipping %s (%s)", node.get("name"), reason)
                continue
            if node.get("status") != "online":
                continue
            node = dict(node)
            node["usable_mib"] = planner.usable_mib(node.get("free_mib"), len(node.get("devices") or ()) or 1)
            usable.append(node)
        chosen = planner.choose_nodes(usable, deficit_mib)
        if not chosen:
            logger.info("Cluster: no reachable computer can cover a %d MiB shortfall", deficit_mib)
            return None
        if not self._link_lock.acquire(timeout = 15.0):
            logger.warning("Cluster: a link test is still running; loading without the cluster")
            return None
        try:
            leased = self._lease_all(chosen)
        finally:
            self._link_lock.release()
        if not leased:
            return None
        ordered = planner.rpc_order(leased)
        attachment = ClusterAttachment(
            nodes = ordered,
            endpoints = [endpoint(n["host"], n["rpc_port"]) for n in ordered],
            deficit_mib = int(deficit_mib),
            capacity_mib = sum(int(n["usable_mib"]) for n in ordered),
        )
        with self._lock:
            self._attachment = attachment
        threading.Thread(
            target = self._keep,
            args = (attachment, alive),
            name = "cluster-lease",
            daemon = True,
        ).start()
        logger.info(
            "Cluster: attaching %s for a %d MiB shortfall (%d MiB usable) via --rpc %s",
            attachment.names,
            attachment.deficit_mib,
            attachment.capacity_mib,
            ",".join(attachment.endpoints),
        )
        return attachment

    def _lease_all(self, chosen: list[dict]) -> list[dict]:
        leased = []
        for node in chosen:
            client = WorkerClient(node["host"], node["control_port"], node["token"], REFRESH_TIMEOUT_S)
            try:
                client.lease(LEASE_TTL_S)
            except WorkerError as exc:
                logger.warning("Cluster: %s refused the lease (%s)", node.get("name"), exc.code)
                self._save_fields(node["id"], {"status": "busy" if exc.code == "busy" else "offline", "last_error": exc.code})
                continue
            if node.get("direct"):
                reachable = port_is_listening(node["host"], node["rpc_port"], timeout = 1.0)
            else:
                reachable = probe_data_path(node["host"], node["rpc_port"]) is not None
            if not reachable:
                logger.warning("Cluster: %s answered but its RPC port is unreachable", node.get("name"))
                self._save_fields(node["id"], {"last_error": "rpc_port_unreachable"})
                try:
                    client.release()
                except WorkerError:
                    pass
                continue
            leased.append(node)
        return leased

    def _keep(self, attachment: ClusterAttachment, alive: Callable[[], bool]) -> None:
        dead = 0
        while not attachment.stop_event.wait(LEASE_RENEW_S):
            try:
                running = bool(alive())
            except Exception:
                running = True
            dead = 0 if running else dead + 1
            if dead >= DEAD_CHECKS_BEFORE_RELEASE:
                with self._lock:
                    if self._attachment is attachment:
                        self._attachment = None
                # a stopped attachment belongs to a newer load that may have re-leased these nodes
                with self._link_lock:
                    if not attachment.stop_event.is_set():
                        attachment.stop_event.set()
                        self._release_nodes(attachment.nodes, REFRESH_TIMEOUT_S)
                return
            for node in attachment.nodes:
                if attachment.stop_event.is_set():
                    return
                try:
                    WorkerClient(node["host"], node["control_port"], node["token"], REFRESH_TIMEOUT_S).lease(LEASE_TTL_S)
                except WorkerError as exc:
                    logger.warning("Cluster: lease renewal for %s failed (%s)", node.get("name"), exc.code)

    @staticmethod
    def _release_nodes(nodes: list[dict], timeout: float) -> None:
        for node in nodes:
            try:
                WorkerClient(node["host"], node["control_port"], node["token"], timeout).release()
            except WorkerError:
                pass

    def release_active(self, release: bool = True, timeout: float = 1.0) -> Optional[ClusterAttachment]:
        with self._lock:
            attachment, self._attachment = self._attachment, None
        if attachment is None:
            return None
        attachment.stop_event.set()
        if release:
            with self._link_lock:
                self._release_nodes(attachment.nodes, timeout)
        return attachment

    def note_failure(self, attachment: ClusterAttachment, message: str) -> None:
        attachment.failure = message
        for node in attachment.nodes:
            self._save_fields(node["id"], {"last_error": "load_failed"})


_head: Optional[ClusterHead] = None
_head_lock = threading.Lock()


def get_cluster_head() -> ClusterHead:
    global _head
    with _head_lock:
        if _head is None:
            _head = ClusterHead()
        return _head


def release_cluster_attachment() -> None:
    head = _head
    if head is not None:
        head.release_active(release = True)
