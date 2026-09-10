import copy,json,sys,time
from pathlib import Path
import httpx
root=Path(__file__).resolve().parent
sys.path.insert(0,str(Path.home()/'.agents/skills/pr-ui-evidence/scripts'))
from pr_ui_scenes._common import studio_session
side=sys.argv[1];port=int(sys.argv[2]);home=root/f'home-{side}'
s=studio_session(f'http://127.0.0.1:{port}',home,(home/'.evidence-password').read_text())
original=json.loads((root/'request.json').read_text());cases=[]
for name in ['base64_mixed','image_only','data_url','two_images','history','streaming','top_level_control','count_image','count_text']:
 b=copy.deepcopy(original);parts=b['messages'][2]['content'][0]['content'];image=parts[1]
 if name=='image_only':b['messages'][2]['content'][0]['content']=[image]
 if name=='data_url':image['source']={'type':'url','url':'data:image/webp;base64,'+image['source']['data']}
 if name=='two_images':parts.insert(2,copy.deepcopy(image))
 if name=='history':b['messages'] += [{'role':'assistant','content':'I have inspected the screenshot.'},{'role':'user','content':'Read the same three-digit number from the earlier screenshot again. Reply only with the number, or NO_IMAGE if no image is available.'}]
 if name=='streaming':b['stream']=True
 if name=='top_level_control':b['messages']=[{'role':'user','content':[{'type':'text','text':'Read the three-digit number in this screenshot. Reply only with the number.'},image]}];b.pop('tools')
 if name=='count_text':b['messages']=[{'role':'user','content':'Hello.'}];b.pop('tools')
 path='/v1/messages/count_tokens' if name.startswith('count_') else '/v1/messages'
 with httpx.Client(base_url=s.base_url,headers={'Authorization':'Bearer '+s.access_token},timeout=180) as c:
  start=time.monotonic();r=c.post(path,json=b);elapsed=time.monotonic()-start
 if name=='streaming':
  events=[json.loads(line[6:]) for line in r.text.splitlines() if line.startswith('data: ') and line[6:]!='[DONE]']
  answer=''.join(e.get('delta',{}).get('text','') for e in events)
  result={'events':events}
 else:
  result=r.json();answer=''.join(p.get('text','') for p in result.get('content',[]))
 facts={'case':name,'http_status':r.status_code,'answer':answer,'elapsed_seconds':round(elapsed,3),'response':result}
 cases.append(facts);print(json.dumps({k:v for k,v in facts.items() if k!='response'}),flush=True)
 (root/f'{side}-live-matrix.json').write_text(json.dumps(cases,indent=2))
 if not name.startswith('count_'):
  assert r.status_code==200,facts
  if side=='after' or name=='top_level_control':assert answer.strip()=='742',facts
 if side=='after' and name=='count_image':assert r.status_code==503,facts
 if name=='count_text':assert r.status_code==200 and result['input_tokens']>0,facts
