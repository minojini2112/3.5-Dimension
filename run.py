"""CLI: python run.py sketch.png --prompt "red sports car" [--seed 42] [--backend auto|neural|geometric]"""
import argparse
import json

from PIL import Image

from s2s import pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sketch")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--control-scale", type=float, default=0.9)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--mc-resolution", type=int, default=256)
    ap.add_argument("--texture-res", type=int, default=1024)
    ap.add_argument("--backend", default="auto", choices=["auto", "neural", "geometric"])
    ap.add_argument("--quality", default="hq", choices=["hq", "fast"],
                    help="hq=Hunyuan3D-2mini, fast=TripoSR")
    ap.add_argument("--shape-prior", default="auto", choices=["auto", "none", "round", "mirror-depth", "mirror-lr"])
    ap.add_argument("--side", default=None, help="optional side-view sketch (object's right side, front to the left)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-setup", action="store_true", help="skip the automatic model check/download")
    a = ap.parse_args()
    if not a.no_setup:
        from s2s.bootstrap import ensure_all
        ensure_all(need_hq=(a.quality == "hq"))
    r = pipeline.run(Image.open(a.sketch), a.prompt, seed=a.seed, steps=a.steps, control_scale=a.control_scale,
                     guidance=a.guidance, n_candidates=a.candidates, mc_resolution=a.mc_resolution,
                     texture_res=a.texture_res, backend=a.backend, out_root=a.out,
                     shape_prior=a.shape_prior, side_sketch=Image.open(a.side) if a.side else None,
                     quality=a.quality)
    print(json.dumps(r.summary(), indent=2, default=str))
