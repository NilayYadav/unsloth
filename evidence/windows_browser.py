import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.request

from playwright.sync_api import sync_playwright
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence'
FRONT = ROOT / 'studio/frontend'
spec = importlib.util.spec_from_file_location('progress_tests', ROOT / 'tests/python/test_windows_setup_download_progress.py')
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)

entry = (OUT / 'update-entry.tsx').read_text()
entry = entry.replace("const { UpdateScreen } = await import('./components/tauri/update-screen');", "const { UpdateScreen } = await (location.search.includes('before') ? import('./components/tauri/update-screen-before') : import('./components/tauri/update-screen'));")
(FRONT / 'src/progress-evidence.tsx').write_text(entry)
shutil.copyfile(OUT / 'update-screen-before.tsx', FRONT / 'src/components/tauri/update-screen-before.tsx')
(FRONT / 'progress-evidence.html').write_text('<div id="root"></div><script type="module" src="/src/progress-evidence.tsx"></script>')
vite_log = (OUT / 'vite.log').open('w')
vite = subprocess.Popen(['node', 'node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5193', '--strictPort'], cwd=FRONT, stdout=vite_log, stderr=subprocess.STDOUT)
payload = b'x' * (20 * 1024 * 1024)
release = threading.Event()
paused = threading.Event()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        for offset in range(0, len(payload)//5, 1024*1024):
            time.sleep(.1)
            self.wfile.write(payload[offset:offset+1024*1024])
            self.wfile.flush()
        paused.set()
        release.wait(60)
        self.wfile.write(payload[len(payload)//5:])
    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
from update_environment import build_environments
environments = build_environments(ROOT, OUT)
facts = []
try:
    for _ in range(120):
        try:
            urllib.request.urlopen('http://127.0.0.1:5193/progress-evidence.html', timeout=1).close()
            break
        except Exception:
            time.sleep(.5)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for component in ['node', 'llama', 'whisper']:
            for side in ['before', 'after']:
                os.environ.pop('UNSLOTH_PROGRESS_PERCENT_STEP', None)
                os.environ.update(environments[side])
                run = OUT / f'{component}-{side}'
                run.mkdir()
                archive = run / f'{component}-windows.zip'
                child = (
                    'import sys\nfrom pathlib import Path\n'
                    f'sys.path.insert(0, {str(ROOT / "studio")!r})\n'
                    f'import install_{component}_prebuilt as installer\n'
                    'installer._LOG_TO_STDOUT = True\n'
                    'print("diagnostic before download", flush=True)\n'
                    f'installer.download_file("http://127.0.0.1:{server.server_port}/archive", Path({str(archive)!r}))\n'
                    'print("diagnostic after download", flush=True)\n'
                )
                context = browser.new_context(viewport={'width':760, 'height':560}, reduced_motion='reduce')
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                page.goto('http://127.0.0.1:5193/progress-evidence.html?' + side)
                page.get_by_role('button', name='Check for update').click(timeout=60000)
                page.get_by_role('button', name='Install update').click()
                page.wait_for_function('window.backendStarted === true')
                page.evaluate("window.emitUpdateEvent('update-progress', 'validating update')")
                paused.clear()
                release.clear()
                process, lines, reader = tests.start_installer(run, shutil.which('powershell'), component, '1', child)
                received = []
                try:
                    assert paused.wait(20)
                    expected = [5, 10, 15, 20] if side == 'after' else [5]
                    observed = []
                    for percent in expected:
                        line = lines.get(timeout=15)
                        received.append(line)
                        actual = float(line.split(': ')[1].split('%')[0])
                        observed.append(actual)
                        assert actual == percent, received
                        page.evaluate("line => window.emitUpdateEvent('update-progress', line)", line)
                        page.wait_for_function("percent => document.querySelector('progress')?.value === percent", arg=percent)
                    time.sleep(.2)
                    assert lines.empty()
                    assert process.poll() is None
                    page.wait_for_timeout(400)
                    page.screenshot(path=str(OUT / f'{component}-{side}.png'))
                    facts.append({'component':component, 'side':side, 'native_shell':'Windows PowerShell 5.1', 'download_active':True, 'percent':expected[-1], 'observed_steps':observed})
                finally:
                    release.set()
                    process.wait(timeout=20)
                    reader.join(timeout=5)
                    process.stdout.close()
                while not lines.empty():
                    line = lines.get_nowait()
                    received.append(line)
                    page.evaluate("line => window.emitUpdateEvent('update-progress', line)", line)
                assert process.returncode == 0, received
                assert hashlib.sha256(archive.read_bytes()).digest() == hashlib.sha256(payload).digest()
                assert all('diagnostic' not in line for line in received)
                captured = (run / 'captured.log').read_text(encoding='utf-8')
                assert 'diagnostic before download' in captured and 'diagnostic after download' in captured
                page.evaluate("window.emitUpdateEvent('update-progress', 'runtime installed and validated')")
                page.wait_for_function("document.querySelector('progress') === null")
                page.evaluate("window.emitUpdateEvent('update-complete', null)")
                page.get_by_text('37% downloaded', exact=True).wait_for()
                page.evaluate('window.finishShellDownload()')
                page.wait_for_function('window.shellInstalled && window.relaunched')
                assert not errors, errors
                facts[-1].update({'exit_code':0, 'checksum_verified':True, 'diagnostics_private':True, 'app_progress':37, 'browser_errors':errors})
                context.close()
        browser.close()
finally:
    release.set()
    server.shutdown()
    server.server_close()
    vite.terminate()
    vite.wait(timeout=10)
    vite_log.close()
for component in ['node', 'llama', 'whisper']:
    images = [Image.open(OUT / f'{component}-{side}.png').convert('RGB') for side in ['before', 'after']]
    assert images[0].tobytes() != images[1].tobytes()
    canvas = Image.new('RGB', (1520,600), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (side, picture) in enumerate(zip(['BEFORE', 'AFTER'], images)):
        canvas.paste(picture, (i*760,40))
        draw.text((i*760+20,12), f'{side} - Windows {component} download paused at 20%', fill='black')
    canvas.save(OUT / f'{component}-comparison.png')
report = {'production_head':os.environ.get('PRODUCTION_HEAD', '6f0719bde39c9726975b11ab9666e396312041fc'), 'boundary':'Compiled updater environment function from a1b398e7d6024b43d2db4632327599c15f401b0c before and 6f0719bde39c9726975b11ab9666e396312041fc after; actual Windows 5.1 setup blocks and 20 MiB HTTP downloads; real React hook and screen. Only progress frequency differs. Tauri IPC and app replacement/relaunch mocked.', 'facts':facts}
(OUT / 'windows-browser.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
