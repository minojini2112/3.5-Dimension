"""Render a GLB from several angles in headless Chromium using <model-viewer> (for README/QA).
usage: python tools/render_views.py model.glb out.png"""
import base64
import sys

from playwright.sync_api import sync_playwright

HTML = """<!doctype html><html><head>
<script type="module" src="https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>body{margin:0;background:#fff;display:flex;flex-wrap:wrap;width:1600px}
model-viewer{width:400px;height:400px;background:#f4f4f6}</style></head><body>
%s</body></html>"""


def main(glb, out):
    data = base64.b64encode(open(glb, "rb").read()).decode()
    src = "data:model/gltf-binary;base64," + data
    views = ["0deg 75deg 2.2m", "90deg 75deg 2.2m", "180deg 75deg 2.2m", "270deg 75deg 2.2m"]
    tags = "".join(f'<model-viewer src="{src}" camera-orbit="{v}" exposure="1.0" shadow-intensity="0.6" '
                   f'environment-image="neutral" interaction-prompt="none"></model-viewer>' for v in views)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        pg = b.new_page(viewport={"width": 1600, "height": 400})
        pg.set_content(HTML % tags)
        pg.wait_for_function("""() => [...document.querySelectorAll('model-viewer')].every(m => m.loaded)""",
                             timeout=120000)
        pg.wait_for_timeout(1500)
        pg.screenshot(path=out)
        b.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
