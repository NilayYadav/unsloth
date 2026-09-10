# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Real download-worker processes against a loopback-only synthetic gated Hub."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'studio/backend'))
PAYLOAD = b'{"model_type":"llama"}\n'
COMMIT = 'a' * 40
ETAG = hashlib.sha1(b'blob ' + str(len(PAYLOAD)).encode() + b'\0' + PAYLOAD).hexdigest()
TOKENS = ('hf_fixture_cached', 'hf_fixture_request', 'hf_fixture_environment')
requests_seen = []
accepted_token = TOKENS[0]

class Hub(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, head=False):
        authorized = accepted_token is None or self.headers.get('Authorization') == 'Bearer ' + accepted_token
        requests_seen.append({'method': self.command, 'authorized': authorized, 'has_auth': bool(self.headers.get('Authorization'))})
        if not authorized:
            self.send_response(401)
            self.send_header('X-Error-Code', 'RepoNotFound')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if '/tree/' in self.path:
            data = json.dumps([{'type': 'file', 'oid': ETAG, 'size': len(PAYLOAD), 'path': 'config.json'}]).encode()
        elif self.path.startswith('/api/models/'):
            data = json.dumps({'id': 'fixture/model', 'sha': COMMIT, 'private': True,
                               'gated': False, 'siblings': [{'rfilename': 'config.json', 'size': len(PAYLOAD)}]}).encode()
        elif '/resolve/' in self.path:
            data = PAYLOAD
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Content-Type', 'application/json')
        self.send_header('X-Repo-Commit', COMMIT)
        self.send_header('ETag', '"' + ETAG + '"')
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def do_GET(self):
        self.respond()

    def do_HEAD(self):
        self.respond(True)

with tempfile.TemporaryDirectory(prefix='pr10748-') as temp:
    home = Path(temp)
    for key in ('HF_TOKEN', 'HF_HUB_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN'):
        os.environ.pop(key, None)
    os.environ.update(HF_HOME=str(home / 'hf'), HF_HUB_CACHE=str(home / 'hf/hub'),
                      HF_TOKEN_PATH=str(home / 'token'), XDG_CACHE_HOME=str(home / 'xdg'),
                      UNSLOTH_STUDIO_HOME=str(home / 'studio'), HF_HUB_DISABLE_IMPLICIT_TOKEN='0',
                      HF_HUB_DISABLE_XET='1', HF_HUB_DISABLE_TELEMETRY='1')
    Path(os.environ['HF_TOKEN_PATH']).write_text(TOKENS[0])
    server = ThreadingHTTPServer(('127.0.0.1', 0), Hub)
    os.environ['HF_ENDPOINT'] = f'http://127.0.0.1:{server.server_port}'
    threading.Thread(target=server.serve_forever, daemon=True).start()
    from huggingface_hub import constants, __version__
    from hub.services.download_lifecycle import spawn_worker
    results = []
    scenarios = [
        ('cached_owner', True, None, False, None, TOKENS[0], True),
        ('restricted_caller', False, None, False, None, TOKENS[0], False),
        ('explicit_restricted', False, TOKENS[1], False, None, TOKENS[1], True),
        ('implicit_disabled', True, None, True, None, TOKENS[0], False),
        ('environment_precedence', True, None, False, TOKENS[2], TOKENS[2], True),
        ('corrupt_cached_public', True, None, False, None, None, True),
    ]
    try:
        for name, ambient, explicit, disabled, environment, expected_token, should_download in scenarios:
            accepted_token = expected_token
            constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = disabled
            os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1' if disabled else '0'
            if environment:
                os.environ['HF_TOKEN'] = environment
            else:
                os.environ.pop('HF_TOKEN', None)
            Path(os.environ['HF_TOKEN_PATH']).write_bytes(
                b'\xff\xfe\xff' if name == 'corrupt_cached_public' else TOKENS[0].encode())
            requests_seen.clear()
            cache = home / name / 'hub'
            try:
                proc = spawn_worker(['--repo-id', 'fixture/model'], explicit, use_xet=False,
                                    allow_ambient_token=ambient, cache_env={'HF_HUB_CACHE': str(cache),
                                    'HUGGINGFACE_HUB_CACHE': str(cache)})
            except (OSError, UnicodeError) as exc:
                row = {'scenario': name, 'spawn_error': type(exc).__name__, 'requests': len(requests_seen),
                       'downloaded_bytes': 0, 'expected_download': should_download, 'pass': False}
                results.append(row)
                print(json.dumps(row), flush=True)
                continue
            try:
                _, stderr = proc.communicate(timeout=45)
            except BaseException:
                proc.kill()
                proc.communicate()
                raise
            target = cache / 'models--fixture--model/snapshots' / COMMIT / 'config.json'
            downloaded = target.read_bytes() if target.exists() else b''
            row = {'scenario': name, 'exit_code': proc.returncode, 'requests': len(requests_seen),
                   'authorized_requests': sum(x['authorized'] for x in requests_seen),
                   'auth_header_requests': sum(x['has_auth'] for x in requests_seen),
                   'downloaded_bytes': len(downloaded), 'expected_download': should_download}
            row['pass'] = (proc.returncode == 0 and downloaded == PAYLOAD) if should_download else (
                proc.returncode != 0 and not downloaded and bool(requests_seen) and not any(x['authorized'] for x in requests_seen))
            if name == 'corrupt_cached_public':
                row['pass'] = row['pass'] and row['auth_header_requests'] == 0
            results.append(row)
            print(json.dumps(row), flush=True)
            if not row['pass']:
                # Only fixture credentials exist, but redact them from diagnostics anyway.
                diagnostic = stderr.decode(errors='replace')
                for token in TOKENS:
                    diagnostic = diagnostic.replace(token, '<fixture>')
                print(diagnostic[-2500:], flush=True)
    finally:
        server.shutdown()
        server.server_close()
    report = {'huggingface_hub': __version__, 'payload_bytes': len(PAYLOAD), 'results': results,
              'passed': sum(x['pass'] for x in results), 'total': len(results)}
    Path('pr10748-facts.json').write_text(json.dumps(report, indent=2))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write('### PR 10748 real worker / synthetic loopback Hub\n```json\n' + json.dumps(report, indent=2) + '\n```\n')
    assert all(x['pass'] for x in results), 'download authentication scenario failed'
