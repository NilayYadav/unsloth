import os

from playwright.sync_api import sync_playwright

URL = os.environ["UI_URL"]
PW = os.environ["UI_PW"]
OUT = os.environ["UI_OUT"]
PROJECT = os.environ["UI_PROJECT"]
os.makedirs(OUT, exist_ok = True)


def shot(page, name):
    page.screenshot(path = os.path.join(OUT, f"{name}.png"), full_page = False)
    print("shot", name, page.url)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport = {"width": 1440, "height": 900})
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(f"{URL}/login", wait_until = "domcontentloaded")
    page.locator("input[type=password]").first.fill(PW)
    page.get_by_role("button", name = "Login").click()
    page.wait_for_url("**/chat**", timeout = 60000)
    page.wait_for_timeout(3000)
    for _ in range(3):
        page.keyboard.press("Escape")
    shot(page, "01-chat")

    page.goto(f"{URL}/library", wait_until = "domcontentloaded")
    page.wait_for_timeout(6000)
    for _ in range(2):
        page.keyboard.press("Escape")
    shot(page, "02-library")
    try:
        page.get_by_text("notes-project.txt").first.wait_for(timeout = 15000)
        print("library shows notes-project.txt: yes")
    except Exception:
        print("library shows notes-project.txt: no")
    shot(page, "03-library-after-wait")

    page.goto(f"{URL}/projects", wait_until = "domcontentloaded")
    page.get_by_text(PROJECT).first.wait_for(timeout = 30000)
    page.wait_for_timeout(1500)
    shot(page, "04-projects")
    page.get_by_role("button", name = "Project options").first.click()
    page.get_by_role("menuitem", name = "Delete").first.click()
    dialog = page.get_by_role("dialog")
    dialog.wait_for(timeout = 15000)
    page.wait_for_timeout(800)
    shot(page, "05-delete-dialog")
    dialog.screenshot(path = os.path.join(OUT, "06-delete-dialog-crop.png"))
    print("dialog text:", dialog.inner_text().replace("\n", " | "))
    page.keyboard.press("Escape")
    print("console errors:", errors[:10])
    browser.close()
