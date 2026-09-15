# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ipaddress
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

KIND_RANK = {"thunderbolt": 4, "ethernet": 3, "unknown": 2, "wifi": 1, "vpn": 0}
NOMINAL_MBPS = {"thunderbolt": 10000, "ethernet": 1000, "unknown": 500, "wifi": 300, "vpn": 50}

_VIRTUAL_PREFIXES = (
    "docker",
    "veth",
    "br-",
    "virbr",
    "vmnet",
    "vboxnet",
    "cni",
    "flannel",
    "kube",
    "lxc",
    "lxd",
    "awdl",
    "llw",
    "anpi",
    "gif",
    "stf",
    "p2p",
    "vethernet",
    "lo",
)
_VPN_PREFIXES = ("utun", "tun", "tap", "wg", "ppp", "ipsec", "zt")
_VPN_MARKERS = ("tailscale", "zerotier", "wireguard", "openvpn", "nordlynx", "vpn")
_WIFI_MARKERS = ("wi-fi", "wifi", "wlan", "wireless", "802.11", "airport")

_PORTS_TTL = 300.0
_ports_lock = threading.Lock()
_ports_cache: Optional[tuple[float, dict[str, str]]] = None


def _macos_hardware_ports() -> dict[str, str]:
    global _ports_cache
    with _ports_lock:
        now = time.monotonic()
        if _ports_cache is not None and now - _ports_cache[0] < _PORTS_TTL:
            return _ports_cache[1]
        ports: dict[str, str] = {}
        try:
            out = subprocess.run(
                ["networksetup", "-listallhardwareports"],
                capture_output = True,
                text = True,
                timeout = 3,
                check = False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        port = None
        for line in out.splitlines():
            if line.startswith("Hardware Port:"):
                port = line.split(":", 1)[1].strip()
            elif line.startswith("Device:") and port:
                ports[line.split(":", 1)[1].strip()] = port
                port = None
        _ports_cache = (now, ports)
        return ports


def is_virtual_interface(name: str) -> bool:
    lowered = name.casefold()
    if lowered.startswith("lo") and not lowered.startswith("local"):
        return True
    return lowered.startswith(_VIRTUAL_PREFIXES[:-1])


def interface_kind(name: str, platform: Optional[str] = None) -> str:
    platform = platform or sys.platform
    lowered = name.casefold()
    if lowered.startswith(_VPN_PREFIXES) or any(m in lowered for m in _VPN_MARKERS):
        return "vpn"
    if platform == "darwin":
        port = _macos_hardware_ports().get(name, "").casefold()
        if not port:
            return "unknown"
        if "wi-fi" in port or "airport" in port:
            return "wifi"
        if "thunderbolt" in port and "ethernet" not in port:
            return "thunderbolt"
        if "ethernet" in port or "lan" in port or "gbe" in port:
            return "ethernet"
        return "unknown"
    if platform.startswith("linux"):
        base = Path("/sys/class/net") / name
        if (base / "wireless").exists() or (base / "phy80211").exists():
            return "wifi"
        if lowered.startswith("thunderbolt"):
            return "thunderbolt"
        if (base / "device").exists():
            return "ethernet"
        return "unknown"
    if any(m in lowered for m in _WIFI_MARKERS):
        return "wifi"
    if "thunderbolt" in lowered:
        return "thunderbolt"
    if "ethernet" in lowered or lowered.startswith(("eth", "en")):
        return "ethernet"
    return "unknown"


def _link_speed_mbps(name: str, stats) -> Optional[int]:
    speed = getattr(stats, "speed", 0) or 0
    if speed <= 0 and sys.platform.startswith("linux"):
        try:
            speed = int((Path("/sys/class/net") / name / "speed").read_text().strip())
        except (OSError, ValueError):
            speed = 0
    return speed if speed > 0 else None


def local_addresses() -> list[dict]:
    try:
        import psutil
    except ImportError:
        psutil = None
    results: list[dict] = []
    if psutil is None:
        source = _default_route_source()
        if source:
            results.append(
                {"address": source, "interface": None, "kind": "unknown", "speed_mbps": None, "public": False}
            )
        return results
    try:
        stats = psutil.net_if_stats()
        addresses = psutil.net_if_addrs()
    except Exception:
        return results
    for name, entries in addresses.items():
        st = stats.get(name)
        if st is not None and not st.isup:
            continue
        if is_virtual_interface(name):
            continue
        kind = interface_kind(name)
        for entry in entries:
            if entry.family != socket.AF_INET:
                continue
            try:
                ip = ipaddress.ip_address(entry.address)
            except ValueError:
                continue
            if ip.is_loopback or ip.is_multicast or ip.is_unspecified:
                continue
            # Thunderbolt Bridge peers normally sit on self-assigned 169.254/16
            if ip.is_link_local and kind != "thunderbolt":
                continue
            results.append(
                {
                    "address": str(ip),
                    "interface": name,
                    "kind": kind,
                    "speed_mbps": _link_speed_mbps(name, st),
                    "public": bool(ip.is_global),
                }
            )
    results.sort(key = lambda a: (-KIND_RANK.get(a["kind"], 0), -(a["speed_mbps"] or 0)))
    return results


def _default_route_source() -> Optional[str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return None


def route_to(peer: str) -> Optional[dict]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect((peer, 9))
            source = probe.getsockname()[0]
    except OSError:
        return None
    for entry in local_addresses():
        if entry["address"] == source:
            return entry
    return {"address": source, "interface": None, "kind": "unknown", "speed_mbps": None, "public": False}


def tcp_rtt_ms(host: str, port: int, samples: int = 5, timeout: float = 1.0) -> Optional[float]:
    times: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        try:
            sock = socket.create_connection((host, port), timeout = timeout)
        except OSError:
            continue
        times.append((time.perf_counter() - started) * 1000.0)
        try:
            sock.close()
        except OSError:
            pass
    return round(statistics.median(times), 3) if times else None


def nominal_mbps(kind: Optional[str], speed_mbps: Optional[int] = None) -> float:
    nominal = float(NOMINAL_MBPS.get(kind or "unknown", NOMINAL_MBPS["unknown"]))
    if speed_mbps:
        return float(min(nominal, speed_mbps)) if kind == "wifi" else float(speed_mbps)
    return nominal
