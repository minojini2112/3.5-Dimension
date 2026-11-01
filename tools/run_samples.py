"""Run the full pipeline on every sample sketch, render 4 views of each GLB and build a gallery + table."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw

from s2s import pipeline
from tools.render_views import main as render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [
    ("mug", "ceramic coffee mug, glossy blue glaze"),
    ("lamp", "table lamp with cream fabric shade and brass stand"),
    ("car", "red sports car"),
    ("chair", "wooden chair"),
    ("vase", ""),  # no prompt: tests prompt-free path
    ("chair_side", "wooden chair"),  # front sketch + side-view sketch
]


def tile(case_dir, sketch_path, out):
    W = 300
    ims = [Image.open(sketch_path).convert("RGB"), Image.open(os.path.join(case_dir, "2_candidate0.png")),
           Image.open(os.path.join(case_dir, "3_foreground.png")).convert("RGBA")]
    v = Image.open(os.path.join(case_dir, "views.png")).convert("RGB")
    row = Image.new("RGB", (W * 3 + int(v.width * W / v.height), W), "white")
    for i, im in enumerate(ims):
        im = im.resize((W, W))
        if im.mode == "RGBA":
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im); im = bg.convert("RGB")
        row.paste(im, (i * W, 0))
    row.paste(v.resize((int(v.width * W / v.height), W)), (W * 3, 0))
    row.save(out)
    return row


if __name__ == "__main__":
    only = sys.argv[1:]
    out_root = os.path.join(ROOT, "outputs", "samples")
    rows, table = [], []
    for name, prompt in CASES:
        if only and name not in only:
            continue
        side = None
        if name.endswith("_side"):
            side = Image.open(os.path.join(ROOT, "samples", f"{name}.png"))
            name = name[:-5]
        sk = os.path.join(ROOT, "samples", f"{name}.png")
        tag = name + ("+side" if side is not None else "")
        r = pipeline.run(Image.open(sk), prompt, side_sketch=side, out_root=os.path.join(out_root, tag))
        render(r.glb, os.path.join(r.out_dir, "views.png"))
        rows.append(tile(r.out_dir, sk, os.path.join(r.out_dir, "summary.png")))
        s = r.summary()
        table.append(dict(case=tag, shape_prior=s["shape_prior"], prompt=prompt or "(none)", backend=s["backend"], iou=s["fidelity_iou"],
                          faces=s["validation"]["glb"]["faces"], textured=s["validation"]["glb"]["texture"],
                          valid=s["validation"]["glb"]["ok"], timings=s["timings_s"], dir=r.out_dir))
        print(json.dumps(table[-1], indent=1))
    if rows:
        g = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for r in rows:
            g.paste(r, (0, y)); y += r.height
        g.save(os.path.join(out_root, "gallery.png"))
    json.dump(table, open(os.path.join(out_root, f"results-{int(time.time())}.json"), "w"), indent=2)
