# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import json
import time

import pytest

from core.cluster import head as head_module
from core.cluster import planner, rpc_server, store
from core.cluster import worker as worker_module
from core.cluster.worker import ClusterServiceError, ControlServer, SharingService, build_control_app


@pytest.fixture
def worker_store():
    store.use_memory_backend()
    yield store
    store.use_database_backend()


@pytest.fixture
def service(worker_store, tmp_path, monkeypatch):
    monkeypatch.setattr(SharingService, "allow_file", property(lambda self: tmp_path / "allow.json"))
    svc = SharingService()
    monkeypatch.setattr(svc, "_memory", lambda: (24000, 24564))
    svc.state = "online"
    svc._family = "cuda"
    svc._build = "abc123"
    svc._relay = object()
    svc._pairing = svc._new_pairing()
    return svc


def _code(svc):
    return svc._pairing["code"]


def _pair(svc, head_id = "head-0000001", name = "Laptop", peer = "10.0.0.2"):
    token = svc.pair(_code(svc), head_id, name, peer)["token"]
    return token, svc.authenticate(token)


def _allow(tmp_path):
    return json.loads((tmp_path / "allow.json").read_text())["allow"]


def test_a_pairing_code_works_in_any_case_and_with_its_dash(service):
    code = _code(service)
    result = service.pair(f"{code[:4].lower()}-{code[4:]}", "head-0000001", "Laptop", "10.0.0.2")
    assert result["family"] == "cuda" and result["free_mib"] == 24000
    assert service.authenticate(result["token"])["name"] == "Laptop"
    assert result["token"] not in json.dumps(store.paired_heads())


def test_a_code_pairs_once(service):
    code = _code(service)
    service.pair(code, "head-0000001", "Laptop", "10.0.0.2")
    with pytest.raises(ClusterServiceError) as exc:
        service.pair(code, "head-0000002", "Desktop", "10.0.0.3")
    assert exc.value.code == "code_invalid"


def test_wrong_guesses_burn_the_code(service, monkeypatch):
    monkeypatch.setattr(worker_module, "PAIRING_MAX_ATTEMPTS", 3)
    real = _code(service)
    for _ in range(3):
        with pytest.raises(ClusterServiceError):
            service.pair("ZZZZZZZZ", "head-0000001", "x", "10.0.0.2")
    with pytest.raises(ClusterServiceError) as exc:
        service.pair(real, "head-0000001", "x", "10.0.0.2")
    assert exc.value.code == "code_expired"


def test_an_expired_code_is_refused(service):
    service._pairing["expires_at"] = time.time() - 1
    with pytest.raises(ClusterServiceError) as exc:
        service.pair(_code(service), "head-0000001", "x", "10.0.0.2")
    assert exc.value.code == "code_expired"


def test_a_computer_cannot_pair_with_itself(service):
    with pytest.raises(ClusterServiceError) as exc:
        service.pair(_code(service), store.identity()["node_id"], "me", "127.0.0.1")
    assert exc.value.code == "self_pairing"


def test_a_lease_opens_the_relay_to_that_address_only(service, tmp_path):
    _token, head = _pair(service)
    service.lease(head, "10.0.0.2", 60)
    allow = _allow(tmp_path)
    assert list(allow) == ["10.0.0.2"] and allow["10.0.0.2"] > time.time()
    service.release(head)
    assert _allow(tmp_path) == {}


def test_one_computer_uses_a_worker_at_a_time(service):
    _t1, first = _pair(service, "head-0000001", "Laptop", "10.0.0.2")
    _t2, second = _pair(service, "head-0000002", "Desktop", "10.0.0.3")
    service.lease(first, "10.0.0.2", 60)
    with pytest.raises(ClusterServiceError) as exc:
        service.lease(second, "10.0.0.3", 60)
    assert exc.value.code == "busy"
    service.lease(first, "10.0.0.2", 60)
    service._leases[first["id"]]["expires_at"] = time.time() - 1
    service.lease(second, "10.0.0.3", 60)


def test_revoking_drops_the_token_and_the_lease(service, tmp_path):
    token, head = _pair(service)
    service.lease(head, "10.0.0.2", 60)
    assert service.revoke_head(head["id"])
    assert service.authenticate(token) is None
    assert _allow(tmp_path) == {}


def test_the_control_api_only_serves_paired_computers(service):
    from fastapi.testclient import TestClient

    client = TestClient(build_control_app(service))
    assert client.get("/cluster/v1/hello").json() == {"service": "unsloth-cluster", "api": 1, "sharing": True}
    assert client.get("/cluster/v1/info").status_code == 401
    assert client.get("/cluster/v1/info", headers = {"Authorization": "Bearer nope"}).status_code == 401
    wrong = client.post("/cluster/v1/pair", json = {"code": "WRONG123", "head_id": "head-0000001"})
    assert wrong.status_code == 403 and wrong.json()["detail"] == "code_invalid"
    paired = client.post(
        "/cluster/v1/pair",
        json = {"code": _code(service), "head_id": "head-0000001", "head_name": "Laptop"},
    )
    auth = {"Authorization": f"Bearer {paired.json()['token']}"}
    assert client.get("/cluster/v1/info", headers = auth).json()["free_mib"] == 24000
    assert client.post("/cluster/v1/lease", json = {"ttl": 30}, headers = auth).status_code == 200
    assert client.post("/cluster/v1/release", headers = auth).json() == {"released": True}
    assert client.post("/cluster/v1/unpair", headers = auth).status_code == 200
    assert client.get("/cluster/v1/info", headers = auth).status_code == 401


def test_addresses_parse_in_the_forms_people_type():
    assert head_module.parse_address("192.168.1.20") == ("192.168.1.20", store.DEFAULT_CONTROL_PORT)
    assert head_module.parse_address(" http://mac-studio.local:6000/ ") == ("mac-studio.local", 6000)
    assert head_module.parse_address("[fe80::1]:7000") == ("fe80::1", 7000)
    for bad in ("", "host:99999", "host:abc"):
        with pytest.raises(head_module.ClusterError):
            head_module.parse_address(bad)


def _separate_store():
    spec = importlib.util.spec_from_file_location("cluster_store_for_head", store.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.use_memory_backend()
    return module


@pytest.fixture
def head_store(monkeypatch):
    separate = _separate_store()
    monkeypatch.setattr(head_module, "store", separate)
    monkeypatch.setattr(head_module, "find_llama_server_binary", lambda: None)
    monkeypatch.setattr(head_module, "detect_local_family", lambda binary = None: "cuda")
    monkeypatch.setattr(head_module, "install_marker", lambda binary: {})
    monkeypatch.setattr(head_module, "probe_data_path", lambda host, port, timeout = 2.0: 0.4)
    monkeypatch.setattr(head_module, "probe_throughput", lambda host, port, seconds = 1.5, timeout = 15.0: 940.0)
    return separate


def _wait(predicate, timeout = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return bool(predicate())


@pytest.fixture
def live_worker(service):
    control_port = rpc_server.free_local_port()
    service._ports = {"control": control_port, "rpc": 50052}
    server = ControlServer(build_control_app(service), control_port)
    server.start()
    yield control_port
    server.stop()


def test_a_head_pairs_then_attaches_only_when_short_of_memory(service, head_store, live_worker):
    cluster = head_module.ClusterHead()
    try:
        node = cluster.pair(f"127.0.0.1:{live_worker}", _code(service))
        assert node["compatible"] and "token" not in node and node["free_mib"] == 24000
        assert _wait(lambda: (head_store.nodes()[0].get("link") or {}).get("measured_at"))
        assert head_store.nodes()[0]["link"]["throughput_mbps"] == 940.0
        assert _wait(lambda: not service._leases)

        assert cluster.plan_for_load(0, alive = lambda: True) is None
        assert not service._leases

        first = cluster.plan_for_load(10000, alive = lambda: True)
        assert first is not None
        host = head_store.nodes()[0]["host"]
        assert first.endpoints == [f"{host}:50052"]
        assert first.capacity_mib == planner.usable_mib(24000, 1)
        assert len(service._leases) == 1

        second = cluster.plan_for_load(10000, alive = lambda: True)
        assert second is not None and first.stop_event.is_set()
        time.sleep(0.3)
        assert len(service._leases) == 1, "a reload must keep the lease the new load uses"

        cluster.release_active(release = True)
        assert not service._leases
    finally:
        cluster.release_active(release = True)


def test_a_different_gpu_family_is_listed_but_never_attached(service, head_store, live_worker):
    service._family = "metal"
    cluster = head_module.ClusterHead()
    try:
        node = cluster.pair(f"127.0.0.1:{live_worker}", _code(service))
        assert not node["compatible"] and node["incompatible_reason"] == "family_mismatch"
        assert cluster.plan_for_load(10000, alive = lambda: True) is None
        assert not service._leases
    finally:
        cluster.release_active(release = True)


def test_cluster_mode_off_never_attaches(service, head_store, live_worker):
    cluster = head_module.ClusterHead()
    try:
        cluster.pair(f"127.0.0.1:{live_worker}", _code(service))
        head_store.set_mode("off")
        assert not cluster.has_enabled_nodes()
        assert cluster.plan_for_load(10000, alive = lambda: True) is None
    finally:
        cluster.release_active(release = True)


class _RecordingService:
    def __init__(self):
        self.calls = []

    def start(self, persist = True):
        self.calls.append(persist)
        return {"state": "online", "pairing": {"code": "ABCD-2345"}, "addresses": []}


@pytest.mark.parametrize(
    "env,remembered,expected",
    [
        ("1", False, [True]),
        (None, True, [False]),
        (None, False, []),
    ],
)
def test_sharing_auto_starts_from_the_cli_flag_or_the_remembered_toggle(
    worker_store, monkeypatch, env, remembered, expected
):
    recording = _RecordingService()
    monkeypatch.setattr(worker_module, "get_sharing_service", lambda: recording)
    if env is None:
        monkeypatch.delenv("UNSLOTH_CLUSTER_SHARE", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_CLUSTER_SHARE", env)
    store.update_share_settings(auto_start = remembered)
    assert worker_module.maybe_auto_start_sharing() is bool(expected)
    assert _wait(lambda: recording.calls == expected, timeout = 5.0)


def test_rpc_failures_in_llama_server_output_become_a_clear_message():
    attachment = head_module.ClusterAttachment(
        nodes = [{"id": "a", "name": "Studio"}],
        endpoints = ["10.0.0.5:50052"],
        deficit_mib = 1,
        capacity_mib = 1,
    )
    assert "Studio" in head_module.cluster_failure_message("Failed to connect to 10.0.0.5:50052\n", attachment)
    assert "version" in head_module.cluster_failure_message("RPC server version mismatch: 7.0.0", attachment)
    assert head_module.cluster_failure_message("Failed to connect to huggingface.co", attachment) is None
    assert head_module.cluster_failure_message("CUDA error: out of memory", attachment) is None
