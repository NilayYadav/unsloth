import asyncio,hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(Path.home()/'.agents/skills/pr-ui-evidence/scripts'),str(Path.home()/'.agents/skills/pr-repro-ci/scripts')]
from pr_ui_scenes._common import studio_session,api_get
from pr_ui_scenes.registry import ScenePlan,REGISTRY
from studio_test_kit.auth import seed_init_script
from playwright.async_api import async_playwright
REGISTRY[10762]=ScenePlan(pr=10762,scene='native_tool_image_monitor',what='Actual Studio API monitor reply to a native Anthropic tool result carrying a screenshot',expect='Before drops the image and cannot read the screenshot; after replies 742. Same image, request, cached Gemma vision weights and llama-server binary.',needs_model=True)
async def main():
 side=sys.argv[1];port=int(sys.argv[2]);home=root/f'home-{side}'
 s=studio_session(f'http://127.0.0.1:{port}',home,(home/'.evidence-password').read_text())
 facts={'plan':REGISTRY[10762].__dict__,'response':json.loads((root/f'{side}-response.json').read_text()),'page_errors':[],'browser':'Chromium','viewport':{'width':1440,'height':1250}}
 async with async_playwright() as p:
  browser=await p.chromium.launch();facts['browser_version']=browser.version
  ctx=await browser.new_context(viewport=facts['viewport'],color_scheme='light')
  await ctx.add_init_script(seed_init_script(s,[])); page=await ctx.new_page()
  page.on('pageerror',lambda e:facts['page_errors'].append(str(e)))
  try:
   await page.goto(s.base_url+'/api-monitor',wait_until='domcontentloaded')
   await page.get_by_label('Search API requests').wait_for(timeout=60000)
   await page.get_by_role('button').filter(has_text='/messages').first.click()
   await page.get_by_role('heading',name='POST /v1/messages').wait_for()
   reply=facts['response']['content'][0]['text']
   await page.get_by_text(reply,exact=True).last.wait_for(timeout=20000)
   await page.get_by_text(reply,exact=True).last.scroll_into_view_if_needed()
   await page.evaluate('document.fonts.ready')
   facts['ui_body_text']=await page.locator('body').inner_text()
   assert reply in facts['ui_body_text']
   await page.screenshot(path=str(root/f'{side}-api-monitor.png'),clip={'x':310,'y':35,'width':1100,'height':1165})
  except BaseException:
   await page.screenshot(path=str(root/f'{side}-debug.png'))
   (root/f'{side}-debug.txt').write_text(await page.locator('body').inner_text());raise
  finally:
   await ctx.close();await browser.close()
 (root/f'{side}-ui-facts.json').write_text(json.dumps(facts,indent=2))
 print(side,'screenshot saved; reply:',reply)
asyncio.run(main())
