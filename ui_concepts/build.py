"""Render four static UI concepts to PNG from existing Sketch2Shape outputs. Runs no AI model."""
import base64
import http.server
import os
import shutil
import socketserver
import threading

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "ui_preview", "assets")
A = os.path.join(HERE, "assets")
os.makedirs(A, exist_ok=True)
for f in os.listdir(SRC):
    if not f.endswith(".glb"):
        shutil.copy(os.path.join(SRC, f), A)

VIEWER = """<!doctype html><script type=module src=https://unpkg.com/@google/model-viewer@3.5.0/dist/model-viewer.min.js></script>
<style>html,body{margin:0;background:transparent}model-viewer{width:1200px;height:900px;background:transparent}</style>
<model-viewer id=m src="%s" camera-orbit="%s" exposure=1.15 shadow-intensity=1 environment-image=neutral interaction-prompt=none></model-viewer>"""
STILLS = [("car_side", "car", "0deg 80deg auto"), ("car_34", "car", "35deg 74deg auto"),
          ("chair_34", "chair", "28deg 74deg auto"), ("flower_front", "flower", "0deg 85deg auto")]

BASE = """<!doctype html><html><head><meta charset=utf-8>
<link rel=stylesheet href="https://fonts.googleapis.com/css2?%s&display=swap"><style>
*{box-sizing:border-box}html,body{margin:0;width:1440px;height:900px;overflow:hidden}h1,h2,h3,p{margin:0}%s</style></head><body>%s</body></html>"""

NOIR = ("family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&family=Jost:wght@300;400;500",
"""body{background:#08090b;color:#efe9dc;font-family:Jost,sans-serif;position:relative}
.beam{position:absolute;left:50%;top:-120px;width:1000px;height:1000px;transform:translateX(-50%);background:conic-gradient(from 160deg at 50% 0,transparent 0 12deg,rgba(255,236,200,.13) 20deg,transparent 40deg)}
.floor{position:absolute;left:50%;top:600px;width:880px;height:130px;transform:translateX(-50%);border-radius:50%;background:radial-gradient(closest-side,rgba(255,236,200,.2),transparent)}
nav{position:absolute;inset:36px 64px auto;display:flex;justify-content:space-between;font-size:13px;letter-spacing:.24em;text-transform:uppercase;color:#a79f90}
nav b{color:#efe9dc;font-weight:500}
h1{position:absolute;top:96px;width:100%;text-align:center;font:400 84px/1.05 "Cormorant Garamond",serif}h1 i{color:#d8b878}
.car{position:absolute;left:230px;top:215px;width:980px}
.bar{position:absolute;left:50%;bottom:112px;transform:translateX(-50%);width:760px;display:flex;align-items:center;gap:14px;border:1px solid #4a4538;padding:10px 10px 10px 26px;font-size:17px;color:#a79f90;background:#0e0f12}
.bar span{flex:1}.bar b{background:#d8b878;color:#17130a;padding:14px 30px;font-weight:500;letter-spacing:.04em}
.rail{position:absolute;bottom:44px;width:100%;display:flex;justify-content:center;gap:58px;font-size:13px;letter-spacing:.2em;text-transform:uppercase;color:#a79f90}.rail b{color:#d8b878;font-weight:400}""",
"""<div class=beam></div><div class=floor></div>
<nav><b>Sketch2Shape</b><span>Gallery &nbsp;&nbsp;&nbsp; How it works &nbsp;&nbsp;&nbsp; Studio</span></nav>
<h1>Every object begins <i>as a line.</i></h1><img class=car src=assets/car_side.png>
<div class=bar><span>Drop your sketch here, or draw one</span><b>Turn it into 3D</b></div>
<div class=rail><span><b>I</b> &nbsp;Sketch</span><span><b>II</b> &nbsp;Image</span><span><b>III</b> &nbsp;3D model</span></div>""")

ATELIER = ("family=Italiana&family=EB+Garamond:ital,wght@0,400;0,500;1,400",
"""body{background:#f3efe6;color:#1f2a24;font-family:"EB Garamond",serif;display:grid;grid-template-columns:1fr 1fr}
.page{position:relative;padding:44px 56px;border-right:1px solid #cfc7b5;background-image:linear-gradient(#e7e1d3 1px,transparent 1px);background-size:100% 34px}
.right{background:#1c3a30;color:#f3efe6;padding:44px 56px;position:relative}
.mark{font:400 26px Italiana,serif;letter-spacing:.06em}
h1{font:400 62px/1.08 Italiana,serif;margin-top:34px}
.sk{display:block;width:430px;margin:18px auto 0;mix-blend-mode:multiply}
.note{position:absolute;left:56px;bottom:94px;font-style:italic;font-size:21px;color:#6a6251}
.tools{position:absolute;left:56px;bottom:40px;display:flex;gap:26px;font-size:18px}.tools u{text-underline-offset:4px}
.right small{font-size:17px;letter-spacing:.14em;text-transform:uppercase;color:#c9b27c}
.right h2{font:400 44px/1.1 Italiana,serif;margin-top:10px}
.obj{position:absolute;left:40px;top:150px;width:640px}
.spec{position:absolute;left:56px;right:56px;bottom:112px;display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid #4f6c60;padding-top:14px;font-size:18px}.spec b{display:block;font-weight:500;color:#c9b27c}
.go{position:absolute;left:56px;right:56px;bottom:40px;background:#c9b27c;color:#1c2a22;text-align:center;padding:15px;font-size:21px}""",
"""<div class=page><div class=mark>Sketch2Shape</div><h1>Draw on the left.<br>Hold it on the right.</h1>
<img class=sk src=assets/chair_sketch.png><p class=note>A wooden chair, walnut finish</p><div class=tools><u>Pencil</u><span>Eraser</span><span>Upload a drawing</span></div></div>
<div class=right><small>Your model</small><h2>Wooden chair</h2><img class=obj src=assets/chair_34.png>
<div class=spec><div><b>Format</b>GLB and OBJ</div><div><b>Surface</b>Fully textured</div><div><b>Drag</b>To turn it</div></div><div class=go>Make it three-dimensional</div></div>""")

COMMAND = ("family=Sora:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;500",
"""body{background:#0a1324;color:#e9eefb;font-family:Sora,sans-serif;display:grid;grid-template:64px 1fr 92px/300px 1fr 280px;background-image:linear-gradient(#142340 1px,transparent 1px),linear-gradient(90deg,#142340 1px,transparent 1px);background-size:40px 40px}
header{grid-column:1/-1;display:flex;align-items:center;justify-content:space-between;padding:0 26px;border-bottom:1px solid #2a4068;background:#0d182c;font-weight:700}header i{font:400 13px "IBM Plex Mono",monospace;color:#9fb0cf}
aside{background:#101c33;padding:20px;display:grid;gap:14px;align-content:start;font-size:14px;color:#9fb0cf}
aside h3{color:#e9eefb;font-size:15px;font-weight:600}
.pad{width:100%;background:#f5f7fc;border-radius:8px;display:block}
.in{border:1px solid #2a4068;border-radius:8px;padding:11px 12px;color:#e9eefb}
.seg{display:grid;grid-template-columns:1fr 1fr;gap:6px;text-align:center}.seg span{border:1px solid #2a4068;border-radius:8px;padding:9px 0}.seg .on{border-color:#ffc46b;color:#e9eefb}
.go{background:#ffc46b;color:#241703;font-weight:600;text-align:center;border-radius:8px;padding:14px;font-size:15px}
main{position:relative}main img{position:absolute;left:0;top:70px;width:860px}
.dim{position:absolute;left:74px;right:70px;top:610px;border-top:1px solid #6fb7ff;font:400 13px "IBM Plex Mono",monospace;color:#6fb7ff;text-align:center;padding-top:6px}
ol{list-style:none;padding:0;margin:0;display:grid;gap:12px}li::before{content:"";display:inline-block;width:11px;height:11px;border-radius:50%;background:#ffc46b;margin-right:10px}li.w::before{background:none;box-shadow:inset 0 0 0 2px #ffc46b}
footer{grid-column:1/-1;display:flex;align-items:center;gap:10px;padding:0 26px;border-top:1px solid #2a4068;background:#0d182c}footer img{height:66px;border-radius:6px;filter:invert(1)}footer img.c{filter:none}footer span{color:#ffc46b;font-size:22px}""",
"""<header>Sketch2Shape<i>Studio</i></header>
<aside><h3>1. Your sketch</h3><img class=pad src=assets/car_sketch.png><h3>2. Describe it</h3><div class=in>red sports car</div><h3>3. Quality</h3><div class=seg><span class=on>Best</span><span>Fast</span></div><div class=go>Generate 3D model</div></aside>
<main><img src=assets/car_34.png><div class=dim>Drag to spin &nbsp; Scroll to zoom</div></main>
<aside><h3>Progress</h3><ol><li>Reading your sketch</li><li>Creating the image</li><li>Isolating the object</li><li>Building the 3D shape</li><li class=w>Painting the texture</li></ol><h3>Download</h3><div class=in>model.glb</div><div class=in>model_obj.zip</div></aside>
<footer><img src=assets/car_sketch.png><span>›</span><img class=c src=assets/car_image.jpg><span>›</span><img class=c src=assets/car_34.png style="background:#172744"></footer>""")

WIZARD = ("family=DM+Serif+Display:ital@0;1&family=Manrope:wght@400;500;700",
"""body{background:#5b1a2b;color:#f9ede4;font-family:Manrope,sans-serif;position:relative}
nav{position:absolute;inset:34px 60px auto;display:flex;justify-content:space-between;font-weight:500}
.count{position:absolute;left:60px;top:150px;font-size:18px;letter-spacing:.04em;color:#e9b9a4}
h1{position:absolute;left:60px;top:186px;width:690px;font:400 104px/.98 "DM Serif Display",serif}h1 i{color:#f3c69b}
.choices{position:absolute;left:60px;bottom:64px;display:grid;grid-template-columns:repeat(2,300px);gap:14px}
.choices div{background:#f9ede4;color:#3e0f1c;padding:22px;font-size:20px;font-weight:700}.choices small{display:block;font-weight:400;font-size:15px;margin-top:4px}
.choices .alt{background:none;color:#f9ede4;box-shadow:inset 0 0 0 2px #f9ede4}
.film{position:absolute;right:0;top:0;bottom:0;width:640px;background:#f9ede4;display:grid;grid-template-rows:1fr 1fr 1.5fr}
.film>div{position:relative;display:grid;place-items:center;border-bottom:2px solid #5b1a2b;overflow:hidden;color:#3e0f1c}
.film img{max-height:100%;max-width:100%;object-fit:contain}.film .tall img{width:640px;max-height:none}.film .a{mix-blend-mode:multiply;height:100%}
.film span{position:absolute;left:18px;top:14px;font-weight:700;font-size:15px;background:#f9ede4;padding:2px 8px}""",
"""<nav><b>Sketch2Shape</b><span>Gallery</span></nav><p class=count>Step 1 of 3</p><h1>What would you like to <i>make?</i></h1>
<div class=choices><div>Draw it here<small>Use your mouse or pen</small></div><div>Upload a sketch<small>Photo of paper works</small></div><div class=alt>Try a sample<small>Car, chair or flower</small></div><div class=alt>See past models<small>Open the gallery</small></div></div>
<div class=film><div><span>Your sketch</span><img class=a src=assets/flower_sketch.png></div><div><img src=assets/flower_image.jpg style="width:100%;height:100%;object-fit:cover"><span>The image</span></div><div class=tall><span>Your 3D model</span><img src=assets/car_side.png></div></div>""")

CONCEPTS = {"1_noir_gallery": NOIR, "2_atelier_sketchbook": ATELIER, "3_blueprint_studio": COMMAND, "4_guided_steps": WIZARD}

os.chdir(HERE)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


srv = socketserver.TCPServer(("127.0.0.1", 8932), Quiet)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    for name, model, orbit in STILLS:
        src = "data:model/gltf-binary;base64," + base64.b64encode(open(os.path.join(SRC, model + ".glb"), "rb").read()).decode()
        g = b.new_page(viewport={"width": 1200, "height": 900})
        g.set_content(VIEWER % (src, orbit))
        g.wait_for_function("()=>document.getElementById('m').loaded", timeout=150000)
        g.wait_for_timeout(2500)
        g.screenshot(path=f"assets/{name}.png", omit_background=True)
        g.close()
        print("still", name, flush=True)
    for name, (fonts, css, body) in CONCEPTS.items():
        with open(name + ".html", "w", encoding="utf-8") as f:
            f.write(BASE % (fonts, css, body))
        g = b.new_page(viewport={"width": 1440, "height": 900})
        g.goto(f"http://127.0.0.1:8932/{name}.html", wait_until="networkidle")
        g.wait_for_timeout(800)
        g.screenshot(path=name + ".png")
        g.close()
        print("concept", name, flush=True)
    b.close()
