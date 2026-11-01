"""Sketch2Shape Gradio app: draw or upload a sketch, add an optional prompt, get a textured 3D model."""
import argparse
import json
import os
import traceback

import gradio as gr
import numpy as np
from PIL import Image

from s2s import pipeline
from s2s.preprocess import SketchError

EXAMPLES = [[f"samples/{n}.png", p] for n, p in [
    ("mug", "ceramic coffee mug, glossy blue glaze"), ("lamp", "table lamp with cream fabric shade and brass stand"),
    ("car", "red sports car"), ("chair", "wooden chair"), ("vase", "")]]


def _ink_pixels(rgba: np.ndarray) -> int:
    return int(((rgba[..., 3] > 32) & (rgba[..., :3].min(axis=-1) < 200)).sum())


def _has_strokes(drawn) -> bool:
    """True only if the Gradio Sketchpad has real ink (not an untouched / blank editor)."""
    if drawn is None:
        return False
    if isinstance(drawn, dict):
        layers = drawn.get("layers") or []
        for layer in layers:
            if layer is None:
                continue
            a = np.asarray(layer)
            if a.ndim == 3 and a.shape[-1] == 4 and _ink_pixels(a) > 30:
                return True
        comp = drawn.get("composite")
        if comp is None:
            return False
        a = np.asarray(comp)
    else:
        a = np.asarray(drawn)
    if a.size == 0:
        return False
    if a.ndim == 3 and a.shape[-1] == 4:
        return _ink_pixels(a) > 30
    # RGB: dark strokes on light paper covering a small but non-zero fraction
    g = a[..., :3].min(axis=-1) if a.ndim == 3 else a
    dark = (g < 200).mean()
    return bool(0.0005 < dark < 0.45)


def _pick_input(drawn, uploaded):
    # Prefer the sketchpad when it has ink. A leftover Upload-tab image must NOT override a new drawing
    # (that made every Generate replay the previous bottle).
    if _has_strokes(drawn):
        return drawn
    if uploaded is not None:
        return uploaded
    raise SketchError("Draw a sketch on the canvas or upload a sketch image first.")


def _optional(drawn, uploaded):
    """Side view is optional — blank sketchpads must not count (they corrupt 3D depth)."""
    if _has_strokes(drawn):
        return drawn
    if uploaded is not None:
        return uploaded
    return None




def generate(drawn, uploaded, side_drawn, side_uploaded, prompt, shape_prior, quality, seed, steps, adherence, n_cand,
             mc_res, tex_res, backend, upright, ensemble, retry, progress=gr.Progress()):
    try:
        sketch = _pick_input(drawn, uploaded)
        side = _optional(side_drawn, side_uploaded)
        r = pipeline.run(sketch, prompt or "", seed=int(seed), steps=int(steps), control_scale=float(adherence),
                         n_candidates=int(n_cand), mc_resolution=int(mc_res), texture_res=int(tex_res),
                         backend=backend, upright=upright, shape_prior=shape_prior, side_sketch=side,
                         ensemble=bool(ensemble), auto_retry=1 if retry else 0, quality=quality or "hq",
                         progress=progress)
    except SketchError as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {type(e).__name__}: {e}")
    s = r.summary()
    rows = [[k, f"{v:.2f}"] for k, v in s["timings_s"].items()]
    v = s["validation"]["glb"]
    info = (f"**Prompt used:** {s['prompt_used'] or '(none)'}  \n**Backend:** {s['backend']}  \n**Shape prior used:** {s['shape_prior']}  \n**Hardware:** {s['hardware']}  \n"
            f"**Sketch fidelity (silhouette IoU):** {s['fidelity_iou'] if s['fidelity_iou'] is not None else 'n/a'}  \n"
            f"**Mesh:** {v['vertices']:,} vertices, {v['faces']:,} faces, "
            f"{'UV texture' if v['texture'] else 'vertex colours' if v['vertex_colors'] else 'no colour'}, "
            f"valid: {'yes' if v['ok'] else 'NO'}")
    rep = s.get("shape_prior_report") or {}
    if rep.get("candidates"):
        info += f"  \n**Object (CLIP):** {rep.get('clip')}  \n**Symmetry scores:** " + ", ".join(
            f"{m} {v['score']:.2f}" for m, v in rep["candidates"].items())
    if s["warnings"]:
        info += "\n\n" + "\n".join(f"⚠️ {w}" for w in s["warnings"])
    gal = [(r.line_map, "1. cleaned sketch")]
    gal += [(c, f"2. generated #{i}") for i, c in enumerate(r.candidates)]
    if r.foreground is not None:
        gal.append((r.foreground, "3. foreground (input to 3D)"))
    return r.glb, [r.glb, r.obj_zip], gal, info, rows, json.dumps(s["validation"], indent=2)


def build():
    with gr.Blocks(title="Sketch2Shape", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Sketch2Shape\nDraw or upload a single-view sketch, optionally describe it, and get a "
                    "textured 3D model (.glb / .obj). Pipeline: preprocessing → ControlNet image synthesis → "
                    "background removal → **Hunyuan3D-2mini (HQ)** or TripoSR (fast) → UV texture bake → validation.")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Draw"):
                        drawn = gr.Sketchpad(type="numpy", image_mode="RGBA", canvas_size=(512, 512),
                                             brush=gr.Brush(colors=["#000000"], default_size=4, color_mode="fixed"),
                                             label="Sketch (brush, eraser, clear)")
                    with gr.Tab("Upload"):
                        uploaded = gr.Image(type="pil", label="Sketch image (PNG/JPG, photo of paper is fine)")
                gr.Markdown("If the canvas has strokes, **Draw wins** over a leftover Upload (clear Upload if you "
                            "only want the file).")
                prompt = gr.Textbox(label="Optional prompt (type, material, colour, style)",
                                    placeholder="e.g. wooden rocking chair, walnut finish")
                quality = gr.Radio(
                    ["hq", "fast"], value="hq",
                    label="3D quality",
                    info="hq = Hunyuan3D-2mini (better shape). fast = TripoSR + shape priors (lighter).")
                shape_prior = gr.Radio(["auto", "none", "round", "mirror-depth", "mirror-lr"], value="auto",
                                       label="Hidden-side shape prior (fast / TripoSR only)",
                                       info="round = vase/mug/lamp (lathe); mirror-depth = side view of a "
                                            "car/animal; mirror-lr = front view of a chair/sofa; auto picks one")

                with gr.Accordion("Optional: side-view sketch (off unless you draw one)", open=False):
                    gr.Markdown("Not needed: hidden sides are inferred automatically. Draw the object seen from "
                                "its **right side** (front pointing left) only if you want to force its depth.")
                    side_drawn = gr.Sketchpad(type="numpy", image_mode="RGBA", canvas_size=(512, 512),
                                              brush=gr.Brush(colors=["#000000"], default_size=4, color_mode="fixed"),
                                              label="Side-view sketch")
                    side_uploaded = gr.Image(type="pil", label="...or upload a side-view sketch")
                with gr.Accordion("Advanced", open=False):
                    ensemble = gr.Checkbox(True, label="Flip ensemble (average 2 reconstructions, +15 s)")
                    retry = gr.Checkbox(True, label="Auto-retry if the image drifts from the sketch")
                    seed = gr.Number(42, label="Seed", precision=0)
                    adherence = gr.Slider(0.4, 1.5, 0.9, step=0.05, label="Sketch adherence (ControlNet scale)")
                    steps = gr.Slider(4, 12, 6, step=1, label="Diffusion steps (LCM)")
                    n_cand = gr.Slider(1, 4, 1, step=1, label="Candidates (best silhouette match is used)")
                    mc_res = gr.Slider(128, 384, 256, step=32, label="Marching-cubes resolution")
                    tex_res = gr.Dropdown([512, 1024, 2048], value=1024, label="Texture resolution")
                    backend = gr.Radio(["auto", "neural", "geometric"], value="auto",
                                       label="Backend (auto = neural with geometric fallback)")
                    upright = gr.Checkbox(True, label="Auto-straighten (remove camera tilt)")
                btn = gr.Button("Generate 3D", variant="primary")
                gr.Examples(EXAMPLES, inputs=[uploaded, prompt])
            with gr.Column(scale=1):
                viewer = gr.Model3D(label="3D preview (drag to orbit)", clear_color=[0.95, 0.95, 0.96, 1.0],
                                    height=480)
                files = gr.File(label="Download (.glb, .obj zip)", file_count="multiple")
                info = gr.Markdown()
                stages = gr.Gallery(label="Intermediate stages", columns=4, height=200)
                timings = gr.Dataframe(headers=["stage", "seconds"], label="Timings")
                with gr.Accordion("Validation report", open=False):
                    valid = gr.Code(language="json")
        btn.click(generate, [drawn, uploaded, side_drawn, side_uploaded, prompt, shape_prior, quality, seed, steps,
                             adherence, n_cand, mc_res, tex_res, backend, upright, ensemble, retry],
                  [viewer, files, stages, info, timings, valid])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--preload", action="store_true", help="load models at startup")
    ap.add_argument("--no-setup", action="store_true", help="skip the automatic model check/download")
    a = ap.parse_args()
    if not a.no_setup:
        from s2s.bootstrap import ensure_all
        ensure_all()  # downloads any missing model / source on first run
    if a.preload:  # only the fast path benefits: image generation and HQ run in their own processes
        from s2s import reconstruct as rc
        rc.load()
    build().queue(max_size=4).launch(server_name=a.host, server_port=a.port, share=a.share,
                                     allowed_paths=[os.path.abspath("outputs")])
