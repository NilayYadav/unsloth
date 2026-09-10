# SPDX-License-Identifier: AGPL-3.0-only
"""Real Studio with deterministic research evidence and an interrupted HTTP stream."""
import asyncio
import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

root, state, port = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
state.mkdir(parents=True, exist_ok=True)
for key in ('UNSLOTH_STUDIO_HOME', 'HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'XDG_CACHE_HOME', 'UNSLOTH_COMPILE_LOCATION'):
    os.environ[key] = str(state / key.lower())
os.environ.update(UNSLOTH_DISABLE_UPDATE_CHECK='1', UNSLOTH_DISABLE_MLX_AUTOREPAIR='1', UNSLOTH_STUDIO_DISABLE_TORCH_WARM='1', UNSLOTH_ALLOW_CPU='1', UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE='1', HF_HUB_OFFLINE='1')
sys.path.insert(0, str(root / 'studio/backend'))
from auth import storage as auth
from storage import studio_db
from storage import research_runs_db as db
from core import research_runs as worker

password = secrets.token_urlsafe(24)
auth.create_initial_user('unsloth', password, secrets.token_urlsafe(32))
(state / 'password').write_text(password)
(state / 'password').chmod(0o600)
REPORT = '''## Findings

The collected evidence identifies two practical constraints for recovering interrupted research. The report should retain completed analysis while making its incomplete status clear to readers. [Recovery guide](https://example.test/recovery)

### What remains available

The system has already gathered the source catalog and written this analysis before the connection is interrupted. Keeping the partial report lets the reader review the results and decide whether to retry the remaining synthesis.

### Validation still needed

This is a deterministic local provider fixture. The final comparison and conclusion had not yet been generated when the connection closed. The report therefore remains incomplete, and the run must remain failed.

'''
RAW = 'Private drafting preamble that must not be shown.\n' + worker._REPORT_BOUNDARY_MARKER + '\n' + REPORT
calls = []


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        broken = 'You are writing a rigorous, self-contained research report.' in body['messages'][0]['content']
        calls.append({'synthesis': broken})
        content = RAW if broken else 'not json'
        data = b''.join(('data: ' + json.dumps({'choices': [{'delta': {'content': content[i:i+170]}}]}) + '\n\n').encode() for i in range(0, len(content), 170))
        if not broken:
            data += b'data: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data) + (1000 if broken else 0)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()
        self.close_connection = True


provider = ThreadingHTTPServer(('127.0.0.1', 0), Provider)
threading.Thread(target=provider.serve_forever, daemon=True).start()
studio_db.upsert_chat_thread({'id': 'evidence-10759', 'title': 'Interrupted research evidence', 'modelType': 'base', 'modelId': 'fixture-model', 'createdAt': 1789060000000})
studio_db.upsert_chat_message({'id': 'user-1', 'threadId': 'evidence-10759', 'role': 'user', 'content': [{'type': 'text', 'text': 'Research how to recover an interrupted report.'}], 'createdAt': 1789060000001})
supervisor = worker.ResearchSupervisor(SimpleNamespace(state=SimpleNamespace(server_port=provider.server_port)))
worker.execute_tool = lambda *a, **kw: 'Title: Recovery guide\nURL: https://example.test/recovery\nSnippet: Retain interrupted report output while marking the run failed.'
db.create_run(run_id='run-1', owner_subject='unsloth', thread_id='evidence-10759', user_message_id='user-1', assistant_message_id=None, config={'model': 'fixture-model', 'inferenceRequest': {'model': 'fixture-model'}, 'ragScope': None, 'instructions': '', 'question': 'Research report recovery', 'budgets': {'maxSteps': 1, 'maxSources': 5, 'modelTimeoutSeconds': 30, 'toolTimeoutSeconds': 10}})
planned = db.set_plan('run-1', {'title': 'Research report recovery', 'steps': [{'title': 'Read recovery evidence', 'query': 'report recovery'}]})
db.approve('run-1', planned['planRevision'], planned['planHash'])
asyncio.run(supervisor._process(db.claim_next(supervisor.worker_id)))
provider.shutdown()
assert any(c['synthesis'] for c in calls)
run = db.get_run('run-1')
assert run['status'] == 'failed' and 'peer closed connection' in run['error']
(state / 'fixture.json').write_text(json.dumps({'provider_calls': calls, 'emitted_report_chars': len(REPORT.strip()), 'expected_report': REPORT.strip()}))
print('FIXTURE_READY: synthesis emitted report and HTTP body was interrupted', flush=True)
import main
import uvicorn
main.setup_frontend(main.app, root / 'studio/frontend/dist')
main.app.state.server_port = port
uvicorn.run(main.app, host='127.0.0.1', port=port, log_level='warning')
