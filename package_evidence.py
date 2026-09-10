import hashlib,json,shutil,sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.home()/'.agents/skills/pr-repro-ci/scripts'))
from studio_test_kit.compose import hstack_images
before=root/'before-api-monitor.png';after=root/'after-api-monitor.png'
assert hashlib.sha256(before.read_bytes()).digest()!=hashlib.sha256(after.read_bytes()).digest()
facts={side:json.loads((root/f'{side}-ui-facts.json').read_text()) for side in ['before','after']}
assert facts['before']['response']['content'][0]['text']=='NO_IMAGE'
assert facts['after']['response']['content'][0]['text']=='742'
assert '[image]' not in facts['before']['ui_body_text'] and '[image]' in facts['after']['ui_body_text']
assert all(not f['page_errors'] for f in facts.values())
hstack_images(before,after,root/'pr10762-before-after.png',label_left='BEFORE d0dbe9059: image lost / NO_IMAGE',label_right='AFTER dbf0d7aac: image retained / 742')
meta={'pr':'https://github.com/unslothai/unsloth/pull/10762','before_sha':'d0dbe9059efa443c6ad8bd1d51af7e2d9276a2bc','after_sha':'dbf0d7aacf8f733939f2969fe004742f53dfe890','model':'unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL','model_revision':'739965d73654c0ead8020786aa998fc813070087','llama_server':'build 10840 commit 58670d128; AppleClang 21 arm64','context_length':8192,'parallel_slots':1,'scene':'native_tool_image_monitor','facts':facts,'identical_images':False,'tests':{'passed':1346,'skipped':6,'skips':'5 redundant matrix cases; optional mlx_lm unavailable'},'limitations':['Local macOS evidence; no GitHub Actions run.','Direct source app launches, separately built frontends and separate homes/auth/HF/XDG caches; shared read-only Python dependency environment, cached model weights and pinned llama-server binary. Installer is outside this evidence.','Coverage claim concerns changed executable lines and branches originating on those lines, not all backend code.','Automated checks and browser driving do not satisfy independent human testing.']}
(root/'meta.json').write_text(json.dumps(meta,indent=2))
pub=root/'public';pub.mkdir(exist_ok=True)
files=['pr10762-before-after.png','before-api-monitor.png','after-api-monitor.png','tool-capture.png','before-live-matrix.json','after-live-matrix.json','changed-coverage.json','meta.json','test_extended_tool_images.py','capture_ui.py','live_request.py','live_matrix.py','launch.py','load_model.py','finalize_coverage.py','package_evidence.py']
for name in files:shutil.copyfile(root/name,pub/name)
for source,dest in [('tests-preload.log','tests.log'),('extended-tests.log','extended-tests.log')]:
 s=(root/source).read_text().replace(str(root.parent.parent),'<workspace>');(pub/dest).write_text(s)
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pub.iterdir() if p.is_file()}
(pub/'sha256.json').write_text(json.dumps(manifest,indent=2))
print('Validated facts, nonidentical screenshots, and prepared',len(files),'public files.')
