# Unsloth Remote — Design

Unsloth Remote lets users author training locally and execute it on any SSH-reachable
GPU machine with a single flag, then run inference locally on the artifacts. It is the
"write Python on your laptop, train in the cloud, chat on your laptop" experience:

```
unsloth train --config train.yaml --remote gpu1     # runs on the cloud GPU
unsloth chat job-42                                 # local inference, minutes later
```

---

## 1. Product principles

1. **One flag between local and remote.** Every command that works locally works with
   `--remote <name>`. Same configs, same CLI, same Studio UI. No new mental model.
2. **The laptop is the source of truth; the GPU is disposable.** Code, configs, job
   history, and pulled artifacts live locally. Any VM can be destroyed and recreated
   without losing anything but time.
3. **Jobs survive everything.** Laptop sleep, SSH drops, terminal closes, and spot
   preemption never kill or orphan a training run. Detached-by-default is an invariant.
4. **No trust expansion.** SSH is the only transport, the user's existing keys are the
   only credentials, and no inbound port is ever opened on the VM.
5. **Reproducible by construction.** Every job records the exact environment (lockfile
   hash), config, code snapshot, and data hashes it ran with.

---

## 2. System architecture

```
┌─ LAPTOP ─────────────────────────────┐      ┌─ GPU VM ─────────────────────────────┐
│                                      │      │                                      │
│  unsloth CLI ──┐                     │      │  unsloth agent (headless Studio      │
│  Python SDK ───┼── tunnel manager ═══╪═SSH══╪═▶ backend in worker mode)            │
│  Studio UI ────┘   (port-forward)    │      │    • job supervisor + queue          │
│                                      │      │    • event stream (loss, VRAM, util) │
│  ~/.unsloth/remotes.toml             │      │    • log buffer (cursor streaming)   │
│  ~/.unsloth/registry/  (models)      │      │    • env manager (uv, lockfiles)     │
│  ~/.unsloth/jobs/      (job cache)   │      │                                      │
│                                      │      │  ~/.unsloth/store/  (content-        │
│  artifacts (adapters, GGUF) ◀────────┼──────┼── addressed: datasets, base-model    │
│                                      │      │   cache, checkpoints, adapters)      │
└──────────────────────────────────────┘      └──────────────────────────────────────┘
                                                       ▲
                                      HF Hub ──────────┘  (datasets & base models
                                                           fetched directly on the VM,
                                                           never routed via laptop)
```

Components:

| Component | Lives | Role |
|---|---|---|
| CLI (`unsloth remote`, `--remote` flag) | laptop | user surface; submits jobs, streams logs, pulls artifacts |
| Python SDK (`unsloth.remote`) | laptop / notebook | programmatic + decorator API over the same job system |
| Tunnel manager | laptop | maintains SSH connections and port-forwards to agents |
| Agent | VM | headless Studio backend in worker mode: job supervisor, metrics, env manager |
| Content-addressed store | VM | dedup'd datasets, model caches, checkpoints, artifacts |
| Provider plugins | laptop | optional VM provisioning (RunPod, Lambda, Vast, AWS, GCP, …) |
| Studio UI | laptop | Remotes view + remote target on the existing training flow |

---

## 3. Client

### 3.1 CLI surface

Remote lifecycle:

```
unsloth remote add <name> <user@host> [--key PATH] [--port N]   # bring-your-own-box
unsloth remote create <name> --provider <p> --gpu <type>        # provision a VM
       [--idle-timeout 30m] [--region ...] [--spot]
unsloth remote list | status <name> | doctor <name>
unsloth remote start <name> | stop <name> | rm <name>
unsloth remote gc <name> [--keep-days N]                        # store garbage collection
```

Execution — the `--remote` flag on existing commands:

```
unsloth train --config train.yaml --remote gpu1 [--detach] [--sweep k=v1,v2,...]
unsloth run   train.py            --remote gpu1 [--with pkg==ver ...]
unsloth export <job|checkpoint> --format gguf --remote gpu1
unsloth chat  <job> [--remote gpu1]              # remote trial inference
unsloth inference <job> [--remote gpu1]
```

Job management:

```
unsloth jobs [--remote gpu1] [--all]
unsloth logs <job> [-f] [--since ...]
unsloth cancel <job>
unsloth resume <job> [--remote other]            # cross-remote resume
unsloth pull  <job> [artifact]                   # explicit artifact fetch
```

Default behaviors:

- `--remote` jobs run **attached by default** (live progress in the terminal) but the
  job itself is always detached on the VM; Ctrl-C detaches the client, never kills the
  job (`unsloth cancel` kills). `--detach` returns immediately after submit.
- On success, the primary artifact (LoRA adapter) is **auto-pulled** and registered in
  the local model registry, so `unsloth chat <job>` works immediately.

### 3.2 Python SDK

```python
import unsloth.remote as ur

gpu1 = ur.Remote("gpu1")                      # by name from remotes.toml
# or provision-on-demand:
gpu1 = ur.Remote.create(provider="runpod", gpu="A100-80GB", idle_timeout="30m")

@ur.function(remote=gpu1, with_packages=["trl==0.19"])
def train(lr: float) -> ur.Artifact:
    from unsloth import FastLanguageModel
    ...
    return ur.Artifact("outputs/adapter")

adapter = train(2e-4)      # executes remotely; returns an artifact reference
adapter.pull()             # materializes it locally, registers in the registry
```

Semantics:

- The decorated function's **module source** (not a pickle of the function) plus the
  synced workdir is what executes remotely; arguments and return values are
  restricted to JSON-serializable values and `ur.Artifact` / `ur.Dataset` references.
  Large objects always travel as store references, never through serialization.
- This is safe where naive cloudpickle-over-SSH is not, because both sides run the
  same uv lockfile (§5) — version skew is controlled by construction.
- `ur.map(train, [1e-4, 2e-4, 5e-5])` fans out as a sweep across the remote's queue
  (and across multiple remotes if given a list).

Notebook integration: `ur.Session("gpu1")` opens a persistent remote kernel;
`%%unsloth_remote gpu1` cell magic executes a cell in that session with outputs,
tracebacks, and tqdm/trainer progress streamed back inline. Sessions survive
client disconnects and can be reattached (`ur.Session.attach("gpu1")`).

### 3.3 Local state

- `~/.unsloth/remotes.toml` — named remotes: endpoint, key path, host-key fingerprint,
  probed capabilities, env hash, provider metadata (for provisioned VMs).
- `~/.unsloth/jobs/` — cached job records mirrored from agents (jobs remain listable
  when a VM is stopped or destroyed).
- `~/.unsloth/registry/` — local model registry mapping job IDs → pulled artifacts,
  base model, chat template; consumed by `unsloth chat/inference/export`.

```toml
# ~/.unsloth/remotes.toml
[remotes.gpu1]
host = "1.2.3.4"
user = "ubuntu"
port = 22
identity_file = "~/.ssh/id_ed25519"
host_key_fingerprint = "SHA256:..."
env_hash = "c0ffee..."
[remotes.gpu1.capabilities]
gpus = [{ model = "A100-80GB", vram_gb = 80 }]
cuda = "12.4"
driver = "550.54"
arch = "x86_64"
disk_free_gb = 412
[remotes.gpu1.provider]           # present only for provisioned VMs
name = "runpod"
instance_id = "..."
idle_timeout = "30m"
spot = true
```

### 3.4 Tunnel manager

A small client-side component (shared by CLI, SDK, and Studio) that maintains one
multiplexed SSH connection per remote (ControlMaster-style) and forwards the agent's
API port to a local ephemeral port. All API traffic is HTTP/JSON over this tunnel —
authentication *is* SSH. It transparently reconnects with backoff; consumers see a
stable local URL per remote.

---

## 4. Remote agent

The agent is the **Studio backend running headless in worker mode** (`unsloth agent`)
— not a separate daemon. This reuses the existing server, API conventions, and the
`unsloth connect` attach story, and keeps one codebase for local Studio and remote
workers.

- Installed by bootstrap into the managed env; supervised by `systemd --user`
  (fallback: tmux with a watchdog) so it restarts on crash and starts on boot.
- Binds to `localhost` only. Reachable exclusively through the SSH tunnel.
- Responsibilities: job supervision and queueing (§6), the event/log APIs, environment
  management (§5), store management (§7), preemption watch (§6.5), idle-timeout
  reporting (§8).

Agent API (tunneled HTTP/JSON):

| Endpoint | Purpose |
|---|---|
| `POST /jobs` | submit a job spec |
| `GET /jobs`, `GET /jobs/{id}` | list / inspect |
| `POST /jobs/{id}/cancel`, `POST /jobs/{id}/resume` | control |
| `GET /jobs/{id}/events?cursor=` | structured event stream (SSE) |
| `GET /jobs/{id}/logs?cursor=` | raw log stream (SSE, cursor-resumable) |
| `GET /metrics` | GPU util, VRAM, disk, queue depth |
| `GET /store`, `POST /store/gc`, `HEAD /store/{hash}` | store inspection, GC, dedup checks |
| `POST /env/verify` | env-hash check (used on every submit) |
| `GET /health` | heartbeat |

**Agentless fallback.** For restricted boxes where a persistent process is not
allowed, every operation degrades to pure SSH: jobs run under tmux writing the same
job-record/event/log files, and the client reads them with `ssh cat`/`tail`. The
on-disk formats are identical, so the two modes are interchangeable per remote
(`mode = "agentless"` in `remotes.toml`); features that require a live agent
(queueing, preemption watch, sessions) are cleanly reported as unavailable.

---

## 5. Environment management

The environment contract is a **hardware-matched uv lockfile**, generated on the
client and enforced on the VM.

1. **Probe** (during `remote add`/`create`, re-run by `doctor`): GPU vendor/model/VRAM,
   driver and CUDA/ROCm version, Python, arch, disk. Probe failures produce actionable
   errors *before* any install ("driver 535 found; torch 2.6+cu124 needs ≥ 550 —
   upgrade with …"), never a cryptic pip failure minutes in.
2. **Resolve**: the client resolves a lockfile for `unsloth` + `unsloth_zoo` + the
   correct torch/CUDA (or ROCm) wheels for the probed hardware.
3. **Build**: bootstrap installs uv on the VM and builds `~/.unsloth/env` from the
   lockfile. Never touches system Python. Idempotent; rebuilds are cheap.
4. **Pin**: the lockfile hash (`env_hash`) is recorded in `remotes.toml` and in every
   job record. Every job submit calls `POST /env/verify`; drift is caught at submit
   time, not three hours into a run.
5. **Overlays**: `--with pkg==ver` (CLI) / `with_packages=` (SDK) create per-job
   overlay envs layered on the base; the base env never mutates.

Because job records carry the env hash, config, code snapshot hash, and data hashes,
any past run is exactly reproducible on any remote.

---

## 6. Job system

### 6.1 Job spec

Submitted as JSON (generated from CLI flags, YAML config, or the SDK):

```jsonc
{
  "id": "job-42",
  "kind": "train | script | export | inference | session",
  "command": { "config": "train.yaml" },          // or {"script": "train.py", "args": [...]}
  "env": { "base_hash": "c0ffee...", "overlay": ["trl==0.19"] },
  "code": { "snapshot": "sha256:...", "workdir": true },
  "data": [ { "source": "hf://user/dataset", "hash": "sha256:..." } ],
  "resources": { "gpus": 1, "nodes": 1 },          // schema supports >1 from day one
  "secrets": ["HF_TOKEN", "WANDB_API_KEY"],        // names only; values injected at spawn
  "artifacts": { "auto_pull": ["adapter"] },
  "policy": { "on_preempt": "checkpoint_and_requeue", "max_retries": 1 }
}
```

### 6.2 State machine

```
queued → preparing → running → uploading → succeeded
                        │          └──────→ failed
                        └─────────────────→ preempted → (policy) requeued
any state ── cancel ──→ cancelled
```

Every transition is appended to the job record with a timestamp and reason. Job
records are plain JSON on the VM (`~/.unsloth/jobs/<id>/record.json`) and mirrored
to the client's job cache whenever it connects.

### 6.3 Events and logs

A trainer callback (installed automatically for `train` jobs; importable as
`unsloth.remote.report()` for custom scripts) emits structured events:

```json
{"t": 1719849600.1, "type": "step", "step": 120, "loss": 1.42, "lr": 1.8e-4,
 "vram_gb": 21.3, "tokens_per_sec": 3120}
```

plus `checkpoint`, `epoch`, `eval`, and `artifact` events. Both event and log streams
are **cursor-based**: `unsloth logs -f` after a two-hour disconnect resumes exactly
where it left off. The CLI renders events as a live progress bar with loss; Studio
renders them with its existing charts components.

### 6.4 Queue, scheduling, sweeps

- Each agent runs a queue scheduled by GPU availability (jobs declare `resources`).
- `--sweep lr=1e-4,2e-4,5e-5` (or `ur.map`) expands into one job per combination,
  grouped under a sweep ID; `unsloth jobs` shows the group with per-run loss, and
  Studio overlays the runs on one chart.
- Given multiple remotes, sweeps fan out across them (simple client-side placement:
  fill the least-loaded compatible remote first).

### 6.5 Checkpointing, resume, preemption

- Checkpoints are written into the content-addressed store and referenced from the
  job record, so they survive workdir cleanup and can be copied between remotes.
- `unsloth resume job-42 [--remote other]` restarts from the latest checkpoint —
  **including on a different machine**: the client (or agents directly, when both are
  reachable) transfers the checkpoint store objects, then resubmits with the same spec.
  This is what makes VMs genuinely disposable.
- **Spot preemption**: the agent watches the provider metadata endpoint (plugin
  supplies the URL/shape). On a preemption notice it signals the trainer to emergency-
  checkpoint, marks the job `preempted`, and applies the job's policy — requeue
  locally, or (for provisioned remotes) trigger reprovision-and-resume.

### 6.6 Multi-GPU and multi-node

- **Single-node multi-GPU** is transparent: the agent knows the GPU count from the
  probe; when `resources.gpus > 1` (or the config requires it) it wraps the launch in
  `accelerate`/`torchrun`. The user experience remains one flag.
- **Multi-node** (`resources.nodes > 1`): one agent acts as the rendezvous head; the
  client establishes tunnels to all members and the head coordinates `torchrun`
  rendezvous over the VMs' private network. The job spec and store are already
  node-count-agnostic; only the launcher differs. (LoRA/QLoRA workloads make
  single-node the overwhelmingly common case; multi-node adds no new user surface
  beyond `--nodes`.)

---

## 7. Data plane

### 7.1 Content-addressed store

`~/.unsloth/store/` on each VM holds datasets, base-model caches (HF cache is
symlinked in), checkpoints, and output artifacts, keyed by content hash.

- Nothing is uploaded or downloaded twice: submits `HEAD /store/{hash}` before
  transferring anything; two jobs on the same base model share one download.
- `unsloth remote gc` reclaims space with age/size policies; `remote status` shows
  store usage.

### 7.2 Uploads

- **Code**: delta sync of the working directory (rsync algorithm) honoring
  `.unslothignore` + `.gitignore`; the synced tree is hashed into the job record as
  the code snapshot.
- **Datasets**: `hf://` sources are downloaded **directly on the VM** — never routed
  through the laptop — using an ephemeral forwarded token. Local paths go over
  resumable rsync into the store.

### 7.3 Artifacts

- On success the adapter is auto-pulled (tens of MB) and registered locally.
- Heavy exports run remotely: `unsloth export job-42 --format gguf --remote gpu1`
  performs the merge/quantization where the RAM, disk, and warm caches are, and only
  the final file is pulled.

---

## 8. Provisioning

`remote add` (BYOB) and `remote create` (provisioned) converge after minute one: a
provider plugin's only job is to produce an SSH-reachable box; the identical
bootstrap takes over from there.

Provider plugin interface (community-extensible entry point
`unsloth.remote.providers`):

```python
class Provider(Protocol):
    name: str
    def create(self, req: InstanceRequest) -> Instance: ...      # → user, host, port, key
    def stop(self, instance_id: str) -> None: ...
    def start(self, instance_id: str) -> Instance: ...
    def destroy(self, instance_id: str) -> None: ...
    def status(self, instance_id: str) -> InstanceStatus: ...
    def preemption_watch(self) -> PreemptionWatchSpec | None: ...  # metadata endpoint shape
    def list_offers(self, gpu: str) -> list[Offer]: ...            # price/availability
```

Built-in plugins: RunPod, Lambda, Vast.ai, AWS, GCP.

- **Idle timeout** — the flagship cost feature: the agent reports idle state (no
  running/queued jobs, no attached sessions); the client-side policy (or the agent
  itself via provider CLI credentials when granted) stops the instance after
  `--idle-timeout`. Because VMs are disposable (§2, §6.5), stop/start is always safe;
  `unsloth train --remote gpu1` against a stopped provisioned remote auto-starts it.
- `remote create --spot` combines spot pricing with the preemption policy for the
  cheapest reliable training.
- `unsloth remote create --gpu A100-80GB` with no provider compares `list_offers`
  across configured providers and picks the cheapest.

---

## 9. Inference

- **Local (the default loop)**: auto-pulled adapters are registered in the local
  registry; `unsloth chat job-42` / `unsloth inference job-42` resolve the artifact,
  base model, and chat template with zero flags. Export to GGUF/Ollama follows the
  existing export flow.
- **Remote trial inference**: `unsloth chat job-42 --remote gpu1` serves from the VM
  where the checkpoint already sits next to a warm GPU — sample the model mid-training
  or before deciding to pull anything. Sessions ride the same tunnel; nothing is
  exposed publicly.

---

## 10. Studio integration

Remote is a *target* on existing Studio flows, not a new app area.

- **Training section**: an execution-target selector (Local / each configured remote,
  with GPU + price metadata) next to the run button. `startTrainingRun` gains a
  `target` — the same YAML config the section already serializes is submitted through
  the tunnel instead of to the local backend. Validation extends with
  remote-capability checks (model fits VRAM, disk headroom).
- **Progress & charts sections**: consume the same event schema (§6.3) regardless of
  target — the components don't know or care that the stream is remote.
- **Remotes view**: manage remotes (add/create/start/stop/doctor), live GPU
  utilization and VRAM, queue and job history with loss curves, log viewer, store
  usage with one-click GC, and pull-and-chat on any succeeded job.
- **Agent access**: `unsloth connect <agent>` already points coding agents at Studio;
  since the remote agent *is* a Studio backend, agents get the same API for
  submitting and babysitting cloud training runs.

---

## 11. Security model

- **Transport**: SSH only. The agent binds to localhost; all API traffic flows through
  client-initiated tunnels. No inbound ports, ever; works behind NAT.
- **Host keys**: trust-on-first-use at `remote add` with the fingerprint pinned in
  `remotes.toml`; mismatches hard-fail.
- **Credentials**: the user's existing SSH keys/agent; private keys are never copied
  anywhere. Provider API keys stay client-side in the OS keychain.
- **Job secrets** (HF/W&B tokens): named in the spec, values injected into the process
  environment at spawn — never written to disk on the VM, scrubbed from logs, absent
  from job records.
- **Privilege**: everything runs as the unprivileged SSH user; bootstrap never
  requires root (uv-managed Python; driver problems are reported, not "fixed").
- **Isolation caveat**: the VM is the user's own single-tenant box; the store and
  agent assume one trusted user per machine. Multi-tenant sharing of one VM is
  explicitly out of scope.

---

## 12. Diagnostics and failure UX

- `unsloth remote doctor <name>` — full re-probe: connectivity, host key, agent
  health, env hash, GPU/driver, disk, store integrity — with a pass/fail table and
  fixes.
- **Error taxonomy mapped to remedies** at the point of failure:
  - CUDA OOM → suggest `load_in_4bit`, gradient accumulation, or smaller max seq len,
    with the exact config diff.
  - Disk full → `unsloth remote gc` with per-category sizes.
  - Driver too old → minimum version and the provider-specific upgrade command.
  - Env drift → one-command env rebuild.
- Client reconnection is silent and automatic (tunnel manager backoff); a job's
  terminal state is never ambiguous because the record on the VM is authoritative and
  mirrored on next contact.

---

## 13. Explicit non-goals

- **A hosted Unsloth control plane.** Everything here is client ↔ user's VM; there is
  no Unsloth-operated service in the path, no telemetry requirement, no account.
- **Multi-tenant VM sharing** (see §11).
- **General-purpose remote execution.** Job kinds are unsloth workflows (train, script
  with unsloth, export, inference, notebook session) — this is not a generic Modal.
- **Windows VMs as targets.** Client runs anywhere; targets are Linux.
