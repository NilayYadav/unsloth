# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise real authenticated Studio and assert report retention identically on both refs."""
import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence-output'
OUT.mkdir(exist_ok=True)
BROWSER = sys.argv[1] if len(sys.argv) > 1 else 'chromium'


async def drive(url, state):
    response = httpx.post(url + '/api/auth/login', json={'username': 'unsloth', 'password': (state / 'password').read_text()})
    response.raise_for_status()
    auth = response.json()
    fixture = json.loads((state / 'fixture.json').read_text())
    errors = []
    async with async_playwright() as p:
        browser = await getattr(p, BROWSER).launch()
        context = await browser.new_context(viewport={'width': 1440, 'height': 1000}, device_scale_factor=1)
        await context.add_init_script('localStorage.setItem("unsloth_auth_token",' + json.dumps(auth['access_token']) + ');localStorage.setItem("unsloth_refresh_token",' + json.dumps(auth['refresh_token']) + ');')
        page = await context.new_page()
        page.on('pageerror', lambda exc: errors.append(str(exc)))
        await page.goto(url + '/chat?thread=evidence-10759', wait_until='domcontentloaded')
        await page.get_by_text(re.compile('peer closed connection')).first.wait_for()
        await page.get_by_role('button', name=re.compile('View activity')).click()
        close_activity = page.get_by_role('button', name='Close research activity', exact=True)
        await close_activity.wait_for()
        timeline = page.get_by_role('log', name='Research activity timeline', exact=True)
        await timeline.get_by_text('Read recovery evidence', exact=False).first.wait_for()
        activity_text = await timeline.inner_text()
        assert 'Read recovery evidence' in activity_text
        await page.screenshot(path=str(OUT / f'{BROWSER}-activity.png'))
        await close_activity.click()
        await close_activity.wait_for(state='hidden')
        await page.reload(wait_until='domcontentloaded')
        await page.get_by_text(re.compile('peer closed connection')).first.wait_for()
        await page.get_by_text('New chat', exact=True).first.click()
        await page.get_by_text('Interrupted research evidence', exact=True).first.click()
        await page.get_by_text(re.compile('peer closed connection')).first.wait_for()
        body = await page.locator('body').inner_text()
        rendered_expected = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', fixture['expected_report'])
        expected_paragraphs = [re.sub(r'^#+\s*', '', part).strip() for part in rendered_expected.split('\n\n')]
        normalized_body = ' '.join(body.split())
        rendered_complete = all(' '.join(part.split()) in normalized_body for part in expected_paragraphs)
        response = httpx.get(url + '/api/chat/research-runs/run-1', headers={'Authorization': 'Bearer ' + auth['access_token']})
        response.raise_for_status()
        run = response.json()
        run = run.get('run', run)
        facts = {'browser': BROWSER, 'browser_version': browser.version, 'viewport': {'width': 1440, 'height': 1000}, 'checkout_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'status': run['status'], 'report_chars': len(run.get('report') or ''), 'expected_report_chars': fixture['emitted_report_chars'], 'report_exact': run.get('report') == fixture['expected_report'], 'sources': len(run['sources']), 'incomplete_visible': 'Deep research failed · Incomplete report' in body, 'report_visible': 'What remains available' in body, 'source_visible': 'Recovery guide' in body, 'error': run['error'], 'error_visible': run['error'] in body, 'private_preamble_visible': 'Private drafting preamble' in body, 'activity_opened': True, 'reload_and_recents_reopen': True, 'page_errors': errors}
        facts['every_report_paragraph_rendered'] = rendered_complete
        await page.screenshot(path=str(OUT / f'{BROWSER}-reopened.png'))
        (OUT / f'{BROWSER}-facts.json').write_text(json.dumps(facts, indent=2))
        print(json.dumps(facts, indent=2), flush=True)
        await context.close()
        await browser.close()
    assert facts['status'] == 'failed'
    assert facts['sources'] == 1 and facts['error_visible']
    assert not facts['private_preamble_visible'] and not errors
    assert facts['report_exact'] and rendered_complete and facts['incomplete_visible'] and facts['source_visible'], 'REPORT_RETENTION: failed research must retain its validated report after reload and reopening'
    print('PASS REPORT_RETENTION: exact report, incomplete label, source, error, activity and reload/reopen', flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix='pr10759-') as temp:
        state = Path(temp)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        with (state / 'server.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name('studio_fixture.py')), str(ROOT), str(state), str(port)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 90
                url = f'http://127.0.0.1:{port}'
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError('Studio setup failed: ' + (state / 'server.log').read_text()[-8000:])
                    try:
                        if (state / 'password').exists() and httpx.get(url + '/healthz', timeout=1).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(.2)
                else:
                    raise RuntimeError('Studio health deadline exceeded')
                asyncio.run(drive(url, state))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    main()
