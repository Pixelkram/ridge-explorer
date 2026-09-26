"""Verify: map dot -> sidebar preview -> click -> same detail modal as strip images."""
import time
from playwright.sync_api import sync_playwright

OUT = ("/tmp/claude-1000/-home-student-ai-jspace/"
       "090e2078-48fc-4f5b-bdcf-cb67d0789297/scratchpad")

with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1050})
    pg.goto("http://localhost:5173/", wait_until="networkidle")
    pg.get_by_text("Cascade — find & refine boundaries").click()
    pg.wait_for_timeout(300)
    for label, val in [("k", "3"), ("chords", "8"), ("patches", "2"), ("seed", "23")]:
        pg.locator(f'label:has-text("{label}") input').first.fill(val)
    pg.get_by_role("button", name="Start").click()
    print("started", flush=True)

    t0 = time.time()
    clicked = False
    while time.time() - t0 < 360:
        # crossing dots are the clickable circles with pointer cursor
        dots = pg.locator('svg circle[style*="cursor"]')
        if not clicked and dots.count() > 0:
            pg.get_by_text("Cascade — find & refine boundaries").scroll_into_view_if_needed()
            pg.mouse.wheel(0, 350)
            pg.wait_for_timeout(200)
            dots.first.click(force=True)
            pg.wait_for_timeout(400)
            side = pg.locator('img[title*="click to enlarge"]')
            if side.count() > 0:
                clicked = True
                pg.screenshot(path=f"{OUT}/ui_sidebar_a.png")
                print("sidebar preview present; clicking it", flush=True)
                side.first.click()
                pg.wait_for_timeout(600)
                modal = pg.get_by_text("explore from here")
                print(f"MODAL OPEN: {modal.count() > 0}", flush=True)
                pg.screenshot(path=f"{OUT}/ui_sidebar_modal.png")
                break
        pg.wait_for_timeout(1500)
    if not clicked:
        print("never saw a selectable crossing with thumb", flush=True)
    b.close()
print("DONE", flush=True)
