import json,secrets,sys
from pathlib import Path
sys.path.insert(0,str(Path.home()/'.agents/skills/pr-ui-evidence/scripts'))
from pr_ui_scenes._common import studio_session,api_post
root=Path(__file__).resolve().parent
side=sys.argv[1]; port=int(sys.argv[2]); home=root/f'home-{side}'
f=home/'.evidence-password'
if not f.exists():
 f.write_text(secrets.token_urlsafe(24)); f.chmod(0o600)
s=studio_session(f'http://127.0.0.1:{port}',home,f.read_text())
model=str(Path.home()/'.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/739965d73654c0ead8020786aa998fc813070087/gemma-4-E2B-it-UD-Q4_K_XL.gguf')
try:
 result=api_post(s,'/api/inference/load',{'model_path':model,'max_seq_length':8192,'n_parallel':1,'speculative_type':'off'},timeout=600)
 print(json.dumps(result,indent=2))
except Exception as e:
 print(type(e).__name__,str(e))
 if hasattr(e,'read'):print(e.read().decode())
 raise
