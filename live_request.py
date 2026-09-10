import base64,json,sys
import httpx
from pathlib import Path
from io import BytesIO
from PIL import Image,ImageDraw,ImageFont
sys.path[:0]=[str(Path.home()/'.agents/skills/pr-ui-evidence/scripts')]
from pr_ui_scenes._common import studio_session,api_post,api_get
root=Path(__file__).resolve().parent
side=sys.argv[1]; port=int(sys.argv[2]); home=root/f'home-{side}'
s=studio_session(f'http://127.0.0.1:{port}',home,(home/'.evidence-password').read_text())
im=Image.new('RGB',(640,480),'#f2d54b');d=ImageDraw.Draw(im)
f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',180)
d.text((320,240),'742',font=f,fill='black',anchor='mm')
im.save(root/'tool-capture.png')
b=BytesIO();im.save(b,format='WEBP',lossless=True)
image={'type':'image','source':{'type':'base64','media_type':'image/webp','data':base64.b64encode(b.getvalue()).decode()}}
body={'max_tokens':128,'temperature':0,'thinking':{'type':'disabled'},'stream':False,'tools':[{'name':'inspect_capture','description':'Read a screenshot.','input_schema':{'type':'object','properties':{}}}], 'messages':[{'role':'user','content':'Read the three-digit number in the screenshot returned by inspect_capture. Reply only with the number. If no image is available, reply NO_IMAGE.'},{'role':'assistant','content':[{'type':'tool_use','id':'toolu_capture','name':'inspect_capture','input':{}}]},{'role':'user','content':[{'type':'tool_result','tool_use_id':'toolu_capture','content':[{'type':'text','text':'Screenshot follows.'},image,{'type':'text','text':'End of screenshot.'}]}]}]}
(root/'request.json').write_text(json.dumps(body,indent=2))
httpx.delete(s.base_url+'/api/inference/monitor',headers={'Authorization':'Bearer '+s.access_token}).raise_for_status()
result=api_post(s,'/v1/messages',body,timeout=180)
(root/f'{side}-response.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
(root/f'{side}-monitor.json').write_text(json.dumps(api_get(s,'/api/inference/monitor'),indent=2))
