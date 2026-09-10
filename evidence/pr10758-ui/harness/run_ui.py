"""Run the suite's isolated before/after driver with a synthetic remote Hub."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = Path.home()/'.agents/skills'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SKILLS/'pr-ui-evidence/scripts'))
import pr_ui_diff
import pr_ui_scenes
from pr_ui_scenes.registry import REGISTRY, ScenePlan
from studio_test_kit import lifecycle
from hub_fixture import Fixture, FIXTURES, TOKEN

pr_ui_scenes.__path__.append(str(HERE))
REGISTRY[10758] = ScenePlan(
    pr=10758, scene='saved_hf_login_download',
    what='Real Studio Hub download card after clicking Download, using a cached Hugging Face login and a tiny gated remote fixture.',
    expect='BEFORE: the real download worker sends no token, receives 401, and the UI shows download failure with no complete snapshot. AFTER: the worker uses the cached login, downloads every fixture file, and the UI shows On device. Browser download requests carry no explicit Hugging Face token on either side.',
)
PROCESSES = []

def start(install, port, log_path, extra_env=None, **kwargs):
    fixture = Fixture(); FIXTURES[str(install.home)] = fixture
    hf = install.home/'hf'; hf.mkdir(parents=True,exist_ok=True)
    (hf/'token').write_text(TOKEN)
    env = {k:v for k,v in os.environ.items() if k not in ('HF_TOKEN','HF_HUB_TOKEN','HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_HUB_TOKEN','HUGGINGFACEHUB_API_TOKEN','HF_OIDC_RESOURCE','HF_OIDC_ID_TOKEN')}
    env.update(UNSLOTH_STUDIO_HOME=str(install.home), HF_HOME=str(hf), HF_TOKEN_PATH=str(hf/'token'),
               HF_HUB_CACHE=str(hf/'hub'), HUGGINGFACE_HUB_CACHE=str(hf/'hub'), HF_XET_CACHE=str(hf/'xet'),
               XDG_CACHE_HOME=str(install.home/'xdg'), HF_ENDPOINT=fixture.endpoint,
               HF_HUB_DISABLE_IMPLICIT_TOKEN='0', HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_TELEMETRY='1',
               UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE='1', UNSLOTH_DIFFUSION_ATTENTION_INSTALL='0',
               UNSLOTH_ALLOW_CPU='1', UNSLOTH_IS_PRESENT='1')
    if extra_env: env.update(extra_env)
    log_path=Path(log_path)
    with log_path.open('w') as log:
        proc=subprocess.Popen([lifecycle._find_unsloth_bin(install),'studio','-H','127.0.0.1','-p',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    PROCESSES.append(proc)
    import urllib.request
    for _ in range(180):
        if proc.poll() is not None: raise RuntimeError(f'Studio launcher exited {proc.returncode}; inspect {log_path}')
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=1) as response:
                if response.status==200: return install
        except Exception: pass
        time.sleep(1)
    raise RuntimeError(f'Studio health timeout; inspect {log_path}')

pr_ui_diff.launch_studio=start
try:
    raise SystemExit(pr_ui_diff.main())
finally:
    for proc in PROCESSES:
        if proc.poll() is None:
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)
    for fixture in FIXTURES.values(): fixture.close()
