"""Real UI and backend; only remote Hugging Face requests use a fixture."""
import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, quote
from playwright.async_api import async_playwright
from pr_ui_scenes._common import api_get
from hub_fixture import FIXTURES, REPO, COMMIT, FILES

async def drive(session, out_dir, label, **kwargs):
    out_dir=Path(out_dir);fixture=FIXTURES[str(session.home)]
    requests=[];errors=[]
    async with async_playwright() as p:
        browser=await p.chromium.launch()
        context=await browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1,record_video_dir=str(out_dir/'video'))
        seed={'unsloth_auth_token':session.access_token,'unsloth_refresh_token':session.refresh_token,'unsloth_onboarding_completed':'true'}
        await context.add_init_script('const evidenceSeed='+json.dumps(seed)+';for(const [k,v] of Object.entries(evidenceSeed))localStorage.setItem(k,v);')
        async def remote(route):
            req=route.request; url=urlsplit(req.url)
            headers={k:v for k,v in req.headers.items() if k.lower() not in ('host','cookie')}
            headers['X-Evidence-Client']='browser'
            response=await route.fetch(url=fixture.endpoint+url.path+('?' + url.query if url.query else ''),headers=headers)
            await route.fulfill(response=response)
        await context.route('https://huggingface.co/**',remote)
        page=await context.new_page()
        page.on('pageerror',lambda exc:errors.append(str(exc)))
        def request(req):
            if req.method=='POST' and urlsplit(req.url).path in ('/api/hub/download','/api/models/download'):
                requests.append({'path':urlsplit(req.url).path,'body':req.post_data_json,'has_explicit_hf_token':any(k.lower()=='x-unsloth-hf-token' for k in req.headers)})
        page.on('request',request)
        try:
            await page.goto(session.base_url+'/hub?model='+quote(REPO,safe=''),wait_until='domcontentloaded')
            await page.wait_for_timeout(6000)
            await page.screenshot(path=str(out_dir/'initial.png'),full_page=True)
            (out_dir/'initial-dom.txt').write_text(await page.locator('body').inner_text())
            button=page.get_by_role('button',name=re.compile(r'^Download(?:\s|$)')).first
            await button.wait_for(state='visible',timeout=45000)
            await button.click()
            for _ in range(90):
                status=await asyncio.to_thread(api_get,session,'/api/hub/download-status?repo_id='+quote(REPO,safe=''))
                if status.get('state') in ('error','complete'):break
                await page.wait_for_timeout(1000)
            else:raise RuntimeError('No terminal download state')
            await page.wait_for_timeout(4000)
            text=await page.locator('body').inner_text()
            (out_dir/'final-dom.txt').write_text(text)
            shot=out_dir/'download-result.png';await page.screenshot(path=str(shot),full_page=True)
            snapshot=session.home/'hf/hub'/('models--'+REPO.replace('/','--'))/'snapshots'/COMMIT
            downloaded={name:len((snapshot/name).read_bytes()) for name in FILES if (snapshot/name).is_file()}
            facts={'state':status['state'],'download_requests':requests,'downloaded_files':downloaded,'downloaded_bytes':sum(downloaded.values()),'complete_snapshot':all((snapshot/name).is_file() and (snapshot/name).read_bytes()==blob for name,blob in FILES.items()),'on_device_visible':'On device' in text,'page_errors':errors,'remote_401s':sum(e['status']==401 and not e['browser'] for e in fixture.events),'authenticated_file_requests':sum('/resolve/' in e['path'] and e['authorized'] and not e['browser'] for e in fixture.events),'viewport':{'width':1440,'height':1000},'browser':browser.version}
            (out_dir/'fixture-events.json').write_text(json.dumps(fixture.events,indent=2))
            assert len(requests)==1 and not requests[0]['has_explicit_hf_token'],requests
            if label=='BEFORE': assert facts['state']=='error' and not facts['complete_snapshot'],facts
            else: assert facts['state']=='complete' and facts['complete_snapshot'] and facts['on_device_visible'],facts
            return [shot],facts
        finally:
            await context.close();await browser.close()
