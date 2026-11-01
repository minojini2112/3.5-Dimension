"""Image understanding used to repair the hidden side automatically (no extra user input):
  * CLIP zero-shot: geometry / symmetry language + a broad object vocabulary (averaged)
  * Depth Anything V2 Small: relative depth of the visible surface, used to score candidate shapes

CLIP is *not* a closed allow-list of judge images. Unknown objects still land on a symmetry mode via
geometry prompts; object words only stabilize common classes. Both models are small and fast on CPU."""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

from . import config, memory

# Geometry-first (works for categories never listed below).
GEOM_PROMPTS = {
    "round": [
        "a 3d object that is a surface of revolution around a vertical axis",
        "a rotationally symmetric upright vessel with a roughly circular top view",
        "an object whose horizontal cross-sections are circles or rings",
    ],
    "mirror-depth": [
        "a 3d object seen from the side with front-back bilateral symmetry",
        "a long object in profile whose far side mirrors the near side",
        "a side-view silhouette of a vehicle, animal, or tool",
    ],
    "mirror-lr": [
        "a 3d object seen from the front with left-right bilateral symmetry",
        "a front-facing symmetric structure or piece of furniture",
        "an object whose left half mirrors its right half",
    ],
    "none": [
        "an irregular asymmetric 3d object with no clear symmetry",
        "a freeform organic shape without bilateral or rotational symmetry",
    ],
}

# Soft anchors only — stabilizes common man-made shapes; not required for an object to work.
OBJECT_ANCHORS = {
    "round": ["vase", "bottle", "cup", "mug", "drinking glass", "bowl", "jar", "table lamp", "candle",
              "tin can", "bucket", "teapot", "kettle", "flower pot", "trophy", "chess piece", "ball",
              "hat", "bell", "lantern", "wine glass", "cake", "spinning top", "rocket", "cylinder",
              "urn", "goblet", "flask", "barrel", "cone", "pillar"],
    "mirror-depth": ["car", "truck", "bus", "airplane", "boat", "bicycle", "motorcycle", "train", "dog",
                     "cat", "horse", "bird", "fish", "shoe", "guitar", "sword", "tank", "tractor",
                     "dinosaur", "whale", "sneaker", "rifle"],
    "mirror-lr": ["chair", "armchair", "sofa", "table", "bed", "cabinet", "desk", "house", "robot",
                  "person", "face", "building", "backpack", "television", "bench", "stool", "shelf",
                  "wardrobe", "gate", "window"],
    "none": ["rock", "tree", "abstract sculpture", "crumpled paper", "pile of stones", "cloud",
             "coral", "driftwood"],
}

_MODES = ("round", "mirror-depth", "mirror-lr", "none")
_GEOM_LABELS = [(m, t) for m, ts in GEOM_PROMPTS.items() for t in ts]
_OBJ_LABELS = [(m, f"a 3d render of a {c}") for m, cs in OBJECT_ANCHORS.items() for c in cs]
_CLIP = None
_DEPTH = None


def _clip():
    global _CLIP
    if _CLIP is None:
        from transformers import CLIPModel, CLIPProcessor
        path = os.path.join(config.WEIGHTS, "clip-vit-b-32")
        model = CLIPModel.from_pretrained(path).eval()
        proc = CLIPProcessor.from_pretrained(path)
        with torch.no_grad():
            texts = [t for _, t in _GEOM_LABELS] + [t for _, t in _OBJ_LABELS]
            t = proc(text=texts, return_tensors="pt", padding=True)
            tf = model.get_text_features(**t)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        n_g = len(_GEOM_LABELS)
        _CLIP = (model, proc, tf[:n_g], tf[n_g:])
    return _CLIP


def _mode_probs(sim: np.ndarray, labels) -> dict:
    p = torch.softmax(torch.tensor(sim) * 100.0, dim=-1).numpy()
    probs = {m: 0.0 for m in _MODES}
    for (m, _), v in zip(labels, p):
        probs[m] += float(v)
    return probs, p


def symmetry_probs(img: Image.Image) -> dict:
    """Blend geometry-CLIP and object-CLIP into one symmetry distribution."""
    model, proc, tf_g, tf_o = _clip()
    with torch.no_grad():
        x = proc(images=img.convert("RGB"), return_tensors="pt")
        f = model.get_image_features(**x)
        f = f / f.norm(dim=-1, keepdim=True)
        g = (f @ tf_g.T)[0].numpy()
        o = (f @ tf_o.T)[0].numpy()
    pg, _ = _mode_probs(g, _GEOM_LABELS)
    po, p_o = _mode_probs(o, _OBJ_LABELS)
    # Equal blend: geometry covers unknowns; objects stabilize everyday sketches.
    probs = {m: 0.5 * pg[m] + 0.5 * po[m] for m in _MODES}
    top_obj_i = int(np.argmax(p_o))
    top_mode = max(probs, key=probs.get)
    short = {"round": "rotational", "mirror-depth": "front-back", "mirror-lr": "left-right", "none": "asymmetric"}
    if float(p_o.max()) >= 0.05:
        label = _OBJ_LABELS[top_obj_i][1].removeprefix("a 3d render of a ")
    else:
        label = short[top_mode]
    return dict(probs=probs, label=label, label_p=float(max(probs.values())),
                probs_geom=pg, probs_obj=po)


def _depth_model():
    global _DEPTH
    if _DEPTH is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        path = os.path.join(config.WEIGHTS, "depth-anything-v2-small")
        _DEPTH = (AutoModelForDepthEstimation.from_pretrained(path).eval(), AutoImageProcessor.from_pretrained(path))
    return _DEPTH


def relative_depth(img: Image.Image) -> np.ndarray:
    """Relative inverse depth (larger = closer), same size as img."""
    model, proc = _depth_model()
    with torch.no_grad():
        x = proc(images=img.convert("RGB"), return_tensors="pt")
        d = model(**x).predicted_depth[None]
        d = torch.nn.functional.interpolate(d, size=img.size[::-1], mode="bicubic", align_corners=False)
    return d[0, 0].numpy()


def available() -> bool:
    return all(os.path.exists(os.path.join(config.WEIGHTS, n, "model.safetensors"))
               for n in ("clip-vit-b-32", "depth-anything-v2-small"))


def unload():
    global _CLIP, _DEPTH
    _CLIP = None
    _DEPTH = None
    memory.free_torch()


_EXTRA_NOUNS = ("flower", "rose", "tulip", "tree", "plant", "leaf", "bottle", "chair", "table", "sofa", "lamp", "mug",
                "cup", "vase", "car", "truck", "boat", "airplane", "house", "castle", "bridge", "robot", "toy",
                "teddy bear", "shoe", "hat", "bag", "key", "sword", "guitar", "fish", "bird", "cat", "dog", "star",
                "heart", "sun", "moon", "apple", "cake")
_HUES = (("red", 0, 10), ("orange", 10, 22), ("yellow", 22, 35), ("green", 35, 85), ("blue", 85, 130),
         ("purple", 130, 150), ("pink", 150, 170), ("red", 170, 181))


def _vocabulary():
    """Label list and unit text embeddings (2 templates averaged). Slow once, then read from weights/."""
    from torchvision.models import ResNet50_Weights
    words = sorted({c.lower() for c in ResNet50_Weights.IMAGENET1K_V1.meta["categories"]} | set(_EXTRA_NOUNS))
    path = os.path.join(config.WEIGHTS, "vocab_embeddings.npy")
    if os.path.exists(path):
        emb = np.load(path)
        if len(emb) == len(words):
            return words, torch.from_numpy(emb)
    model, proc, _, _ = _clip()
    parts = []
    with torch.no_grad():
        for i in range(0, len(words), 128):
            batch = words[i:i + 128]
            feats = 0
            for template in ("a drawing of a {}", "a 3d render of a {}"):
                t = proc(text=[template.format(c) for c in batch], return_tensors="pt", padding=True)
                f = model.get_text_features(**t)
                feats = feats + f / f.norm(dim=-1, keepdim=True)
            parts.append(feats / feats.norm(dim=-1, keepdim=True))
    emb = torch.cat(parts)
    np.save(path, emb.numpy())
    return words, emb


def colour_words(rgb: np.ndarray) -> list:
    """Names of the dominant saturated colours in the input: only coloured sketches and photos have any."""
    import cv2
    if rgb.ndim != 3:
        return []
    h, s, v = cv2.split(cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2HSV))
    hue = h[(s > 90) & (v > 70)]
    if len(hue) < 0.01 * h.size:
        return []
    shares = {}
    for name, lo, hi in _HUES:
        shares[name] = shares.get(name, 0) + float(((hue >= lo) & (hue < hi)).mean())
    return [n for n, p in sorted(shares.items(), key=lambda kv: -kv[1])[:2] if p > 0.3]


def name_drawing(line_map: Image.Image, rgb: np.ndarray) -> str:
    """Name a drawing using CLIP over a large vocabulary. '' when no label is confident enough."""
    words, emb = _vocabulary()
    model, proc, _, _ = _clip()
    with torch.no_grad():
        f = model.get_image_features(**proc(images=line_map.convert("RGB"), return_tensors="pt"))
        f = f / f.norm(dim=-1, keepdim=True)
        probs = (model.logit_scale.exp() * f @ emb.T).softmax(-1)[0]
    p, i = probs.max(0)
    print(f"[understand] drawing looks like {words[int(i)]} ({float(p):.2f})", flush=True)
    return " ".join(colour_words(rgb) + [words[int(i)]]) if float(p) > 0.12 else ""


def name_isolated(line_map: Image.Image, rgb: np.ndarray, out_dir: str) -> str:
    """name_drawing() in a child process: its memory has to be gone before the 3D stage starts."""
    lm, col, out = (os.path.join(out_dir, n) for n in ("name_input.png", "name_colour.png", "name.json"))
    line_map.save(lm)
    Image.fromarray(rgb).save(col)
    p = subprocess.run([sys.executable, "-m", "s2s.understand", lm, col, out], cwd=config.ROOT, timeout=600)
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"naming process exited with code {p.returncode}")
    with open(out, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    lm_path, col_path, out_path = sys.argv[1:4]
    result = name_drawing(Image.open(lm_path), np.array(Image.open(col_path)))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
