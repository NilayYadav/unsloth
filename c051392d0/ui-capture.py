import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKILLS = Path.home() / '.agents/skills'
sys.path[:0] = [str(SKILLS / 'pr-ui-evidence/scripts'), str(SKILLS / 'pr-repro-ci/scripts')]
from pr_ui_scenes._common import studio_session
from pr_ui_scenes.registry import REGISTRY, ScenePlan
from studio_test_kit.compose import hstack_images
from responses_history_monitor import drive

plan = json.loads((ROOT / 'reports/ui-plan.json').read_text())
plan.update(pr=10761, after_sha='c051392d0368c5ea0467e2049a1ac7bcc7b09ab7')
history_param = os.environ.get('UI_HISTORY_PARAMETER', 'previous_response_id')
plan['history_parameter'] = history_param
plan['expect'] = plan['expect'].replace('previous_response_id', history_param)
REGISTRY[10753] = ScenePlan(pr=10753, scene=plan['scene'], what='API monitor after an unsupported Responses continuation', expect=plan['expect'])


async def main():
    output = ROOT / 'ui-evidence-10761' / history_param
    output.mkdir(parents=True, exist_ok=True)
    facts = {}
    shots = {}
    for label, port in [('BEFORE', 18753), ('AFTER', 18754)]:
        side = label.lower()
        home = ROOT / f'home-{side}'
        password_file = home / '.evidence-password'
        if not password_file.exists():
            password_file.write_text(secrets.token_urlsafe(24))
            password_file.chmod(0o600)
        session = studio_session(f'http://127.0.0.1:{port}', home, password_file.read_text())
        sha = subprocess.check_output(['git', '-C', str(ROOT / f'ui-{side}'), 'rev-parse', 'HEAD'], text=True).strip()
        assert sha == plan[f'{side}_sha']
        (home / '.uidiff_sha').write_text(sha)
        result, facts[label] = await drive(session, output, label, history_param=history_param)
        shots[label] = result[0]
        print(label, {k: v for k, v in facts[label].items() if k not in {'ui_body_text'}}, flush=True)
    assert facts['BEFORE']['control_response'] == facts['AFTER']['control_response']
    assert facts['BEFORE']['control_monitor_rows'] == facts['AFTER']['control_monitor_rows'] == 1
    assert facts['BEFORE']['probe_monitor_rows'] == 1 and facts['AFTER']['probe_monitor_rows'] == 0
    assert hashlib.sha256(shots['BEFORE'].read_bytes()).digest() != hashlib.sha256(shots['AFTER'].read_bytes()).digest()
    # Same-size screenshots: composition adds labels, without resizing or altering UI pixels.
    hstack_images(shots['BEFORE'], shots['AFTER'], output / 'pr10761-before-after.png',
                  label_left='BEFORE d0dbe9059 — 1 failed request',
                  label_right='AFTER c051392d0 — rejected before recording')
    (output / 'meta.json').write_text(json.dumps({'plan': plan, 'facts': facts, 'identical_images': False}, indent=2))
    print('PASS: real API-monitor row delta and different screenshots', flush=True)


asyncio.run(main())
