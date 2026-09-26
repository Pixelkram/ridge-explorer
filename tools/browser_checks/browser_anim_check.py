"""Drive the real frontend in headless chromium: start a cascade from the UI,
screenshot the new animations (lock-on reticles, filling patch diamonds, shimmer
grids) as they appear, and prove motion with a frame diff."""
import time

from playwright.sync_api import sync_playwright

OUT = ("/tmp/claude-1000/-home-student-ai-jspace/"
       "090e2078-48fc-4f5b-bdcf-cb67d0789297/scratchpad")


def shot(page, name, note):
    page.screenshot(path=f"{OUT}/{name}")
    print(f"saved {name}: {note}", flush=True)


with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 1050})
    page.goto("http://localhost:5173/", wait_until="networkidle")
    page.get_by_text("Cascade — find & refine boundaries").click()
    page.wait_for_timeout(400)

    # k=3 quick run so all phases fit in ~3 min
    for label, val in [("k", "3"), ("chords", "10"), ("patches", "3"),
                       ("seed", "11")]:
        page.locator(f'label:has-text("{label}") input').first.fill(val)
    page.get_by_role("button", name="Start").click()
    print("started from UI", flush=True)

    svg = page.locator('svg[style*="rgb(13, 13, 32)"], svg').last
    page.get_by_text("Cascade — find & refine boundaries").scroll_into_view_if_needed()

    got = set()
    t0 = time.time()
    while time.time() - t0 < 420:
        el = int(time.time() - t0)
        n_ret = page.locator("circle.cs-march").count()
        n_diam = page.locator("rect.cs-march").count()
        n_shim = page.locator("div.cs-shimmer").count()
        n_pts = page.locator("circle").count()

        if "cloud" not in got and n_pts > 60:
            page.mouse.wheel(0, 400)
            page.wait_for_timeout(300)
            shot(page, "ui_t1_cloud.png", f"t+{el}s chord phase, {n_pts} circles")
            got.add("cloud")
        if "reticle" not in got and n_ret > 0:
            shot(page, "ui_t2_reticles_a.png",
                 f"t+{el}s {n_ret} lock-on reticles visible")
            page.wait_for_timeout(600)
            shot(page, "ui_t2_reticles_b.png", "same view 600ms later (motion diff)")
            got.add("reticle")
        if "shimmer" not in got and n_shim > 0:
            shot(page, "ui_t3_map_patches.png",
                 f"t+{el}s active diamonds={n_diam}")
            sh = page.locator("div.cs-shimmer").first
            sh.scroll_into_view_if_needed()
            page.wait_for_timeout(300)
            shot(page, "ui_t4_shimmer_grid.png",
                 f"t+{el}s {n_shim} shimmer cells in assembling grid")
            got.add("shimmer")
        running = page.get_by_role("button", name="running…").count() > 0
        if not running and len(got) >= 1 and el > 60:
            page.get_by_text("Cascade — find & refine boundaries")\
                .scroll_into_view_if_needed()
            page.mouse.wheel(0, 400)
            page.wait_for_timeout(300)
            shot(page, "ui_t5_done.png", f"t+{el}s run complete")
            break
        page.wait_for_timeout(1500)

    print(f"captured: {sorted(got)}", flush=True)
    browser.close()

# motion proof: reticle frames must differ
try:
    from PIL import Image, ImageChops
    a = Image.open(f"{OUT}/ui_t2_reticles_a.png")
    b = Image.open(f"{OUT}/ui_t2_reticles_b.png")
    diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
    bbox = diff.getbbox()
    px = sum(1 for p in diff.getdata() if sum(p) > 30)
    print(f"MOTION DIFF: bbox={bbox}, changed_px={px}", flush=True)
except Exception as e:
    print("motion diff skipped:", e, flush=True)
print("BROWSER CHECK DONE", flush=True)
