"""Browser smoke test: draw a fresh sketch on the Sketchpad with the mouse, type a prompt, click Generate,
wait for the 3D viewer, save screenshots. usage: python tools/ui_smoke_test.py [url] [out_prefix]"""
import math
import sys

from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
out = sys.argv[2] if len(sys.argv) > 2 else "ui"

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1500, "height": 1300})
    pg.goto(url)
    pg.wait_for_selector("canvas", timeout=60000)
    pg.wait_for_timeout(2000)
    cv = max(pg.query_selector_all("canvas"), key=lambda c: (c.bounding_box() or {"width": 0})["width"])
    bb = cv.bounding_box()
    X = lambda u: bb["x"] + u * bb["width"]
    Y = lambda v: bb["y"] + v * bb["height"]

    def line(pts):
        pg.mouse.move(X(pts[0][0]), Y(pts[0][1]))
        pg.mouse.down()
        for u, v in pts[1:]:
            pg.mouse.move(X(u), Y(v), steps=8)
        pg.mouse.up()

    # a fresh sketch: a teapot-ish kettle (body, lid, spout, handle)
    body = [(0.5 + 0.22 * math.cos(t), 0.6 + 0.18 * math.sin(t)) for t in [i * 2 * math.pi / 40 for i in range(41)]]
    line(body)
    line([(0.38, 0.43), (0.42, 0.36), (0.58, 0.36), (0.62, 0.43)])
    line([(0.5, 0.36), (0.5, 0.32)])
    line([(0.28, 0.6), (0.18, 0.5), (0.14, 0.42)])
    line([(0.29, 0.68), (0.2, 0.55), (0.14, 0.45)])
    line([(0.72, 0.5), (0.84, 0.5), (0.86, 0.62), (0.72, 0.7)])
    pg.get_by_label("Optional prompt (type, material, colour, style)").fill("white porcelain teapot with gold trim")
    pg.screenshot(path=f"{out}_1_drawn.png")
    pg.get_by_role("button", name="Generate 3D").click()
    pg.wait_for_function("() => document.querySelector('a[download]') !== null || document.body.innerText.includes('Backend:') || document.body.innerText.includes('Error')", timeout=1500000)
    pg.wait_for_timeout(6000)
    pg.screenshot(path=f"{out}_2_result.png", full_page=True)
    print(pg.inner_text("body")[-1500:])
    b.close()
