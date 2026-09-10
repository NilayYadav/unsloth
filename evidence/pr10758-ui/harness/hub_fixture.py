"""Synthetic remote Hub; Studio and its worker remain unmodified."""
import hashlib
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote

REPO = 'fixture/saved-login-model'
TOKEN = 'hf_fixture_saved_login_only'
COMMIT = 'a' * 40
_header = json.dumps({'fixture': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}, separators=(',', ':')).encode()
_header += b' ' * ((8 - len(_header) % 8) % 8)
FILES = {
    'config.json': json.dumps({'model_type': 'llama', 'architectures': ['LlamaForCausalLM'], 'hidden_size': 1, 'num_hidden_layers': 1, 'num_attention_heads': 1, 'vocab_size': 1}).encode(),
    'model.safetensors': struct.pack('<Q', len(_header)) + _header + struct.pack('<f', 1.0),
    'README.md': b'---\nlibrary_name: transformers\npipeline_tag: text-generation\nlicense: mit\n---\n# Saved login download fixture\n\nA tiny gated repository used to verify saved Hugging Face login. No model inference is performed.\n',
}

def oid(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()

MODEL = {'_id': 'a'*24, 'id': REPO, 'modelId': REPO, 'author': 'fixture', 'sha': COMMIT,
         'private': False, 'gated': 'manual', 'disabled': False,
         'pipeline_tag': 'text-generation', 'library_name': 'transformers',
         'tags': ['transformers', 'safetensors', 'llama', 'text-generation', 'license:mit'],
         'downloads': 1, 'downloadsAllTime': 1, 'likes': 0,
         'createdAt': '2026-01-01T00:00:00.000Z', 'lastModified': '2026-01-01T00:00:00.000Z',
         'config': {'architectures': ['LlamaForCausalLM'], 'model_type': 'llama'},
         'safetensors': {'parameters': {'F32': 1}, 'total': 1},
         'siblings': [{'rfilename': name, 'size': len(data), 'blobId': oid(data)} for name, data in FILES.items()]}

class Fixture:
    def __init__(self):
        self.events = []
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_OPTIONS(self): self.send(b'', 'text/plain')
            def do_HEAD(self): self.respond(True)
            def do_GET(self): self.respond(False)
            def send(self, data, mime='application/json', status=200, head=False, etag=None):
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Headers', '*')
                self.send_header('X-Repo-Commit', COMMIT)
                if etag: self.send_header('ETag', '"'+etag+'"')
                if status == 401:
                    self.send_header('X-Error-Code', 'GatedRepo')
                    self.send_header('X-Error-Message', 'Saved Hugging Face login is required for this gated fixture.')
                self.end_headers()
                if not head: self.wfile.write(data)
            def respond(self, head):
                path = unquote(urlsplit(self.path).path)
                authorized = self.headers.get('Authorization') == 'Bearer '+TOKEN
                browser = self.headers.get('X-Evidence-Client') == 'browser'
                status = 200
                if '/resolve/' in path or '/raw/' in path:
                    name = path.split('/resolve/',1)[-1] if '/resolve/' in path else path.split('/raw/',1)[-1]
                    name = name.split('/',1)[-1]
                    data = FILES.get(name)
                    if data is None:
                        status = 404; data = b'{"error":"file not found"}'
                    elif name != 'README.md' and not authorized:
                        status = 401; data = b'{"error":"Saved Hugging Face login required"}'
                    mime = 'text/plain' if name.endswith('.md') else 'application/octet-stream'
                    fixture.events.append({'method': self.command, 'path': path, 'status': status, 'authorized': authorized, 'browser': browser, 'bytes': len(data) if status == 200 and not head else 0})
                    self.send(data,mime,status,head,oid(FILES[name]) if name in FILES else None); return
                if '/tree/' in path:
                    data = [{'type':'file','oid':oid(blob),'size':len(blob),'path':name} for name,blob in FILES.items()]
                elif path.startswith('/api/models/'+REPO): data = MODEL
                elif path == '/api/models': data = [MODEL]
                elif path.startswith('/api/datasets'): data = []
                elif path.endswith('/README.md'): self.send(FILES['README.md'],'text/plain',head=head);return
                elif path.endswith('/overview'): data = {'name':'fixture','fullname':'Fixture','avatarUrl':None}
                elif path == '/': self.send(b'Fixture Hub','text/plain',head=head);return
                else: data = []
                fixture.events.append({'method':self.command,'path':path,'status':status,'authorized':authorized,'browser':browser,'bytes':0})
                self.send(json.dumps(data).encode(),head=head)
        self.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}'
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
    def close(self): self.server.shutdown();self.server.server_close()

FIXTURES = {}
