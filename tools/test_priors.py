"""Re-run only the 3D stage on saved foregrounds with different shape priors, render 4 views each."""
import glob, os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
from s2s import reconstruct, export
from tools.render_views import main as render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cases = dict(mug="ceramic coffee mug, glossy blue glaze", lamp="table lamp with cream fabric shade and brass stand",
             car="red sports car", chair="wooden chair", vase="")
out = sys.argv[1]
modes = sys.argv[2].split(",")
names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(cases)
os.makedirs(out, exist_ok=True)
for n in names:
    fg = Image.open(sorted(glob.glob(f"{ROOT}/outputs/samples/{n}/*/3_foreground.png"))[-1])
    for m in modes:
        mesh, t = reconstruct.image_to_mesh(fg, prior=m, prompt=cases[n], texture_res=512)
        mesh = export.normalise(export.upright(mesh))
        p = f"{out}/{n}_{m}.glb"
        mesh.export(p)
        render(p, p.replace(".glb", ".png"))
        print(n, m, t.get("prior_mode"), {k: round(v, 2) for k, v in t.items() if isinstance(v, float)}, flush=True)
