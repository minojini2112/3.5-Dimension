"""Shape priors that repair the hidden (unseen) side of a single-view reconstruction.

TripoSR only sees the front, so depth and the back are guesses. Most man-made objects are either
  * round   (vase, bottle, mug, lamp, bowl ...): a surface of revolution around the vertical axis
  * mirror-depth (car, boat, animal seen from the side): the far half mirrors the near half
  * mirror-lr (chair, sofa, building seen from the front): left half mirrors the right half
We edit TripoSR's density volume (not the mesh) in a camera-aligned frame, so the result stays one clean
surface after marching cubes. Parts that stick out of a round body (mug handle, teapot spout) are kept from
the original field. An optional side-view sketch carves the volume so its depth matches the user's intent.

All work is NumPy on a 256^3 grid: about 1 to 3 s on a laptop CPU."""
import re

import numpy as np
from scipy.ndimage import median_filter

ROUND_WORDS = r"vase|bottle|cup|mug|glass|goblet|bowl|pot|jar|lamp|candle|can\b|bucket|barrel|kettle|teapot|" \
              r"tumbler|flask|urn|trophy|pawn|chess|column|pillar|cake|bell|planter|egg|ball|sphere|globe|" \
              r"lantern|cone|cylinder|top hat|hat|tower|lighthouse|rocket|spinning top|bulb|perfume"
SIDE_WORDS = r"car|truck|bus|van|vehicle|train|tram|boat|ship|yacht|submarine|airplane|aeroplane|plane|jet|" \
             r"bike|bicycle|motorcycle|scooter|tank|dog|cat|horse|cow|fish|shark|whale|dinosaur|bird|shoe|" \
             r"sneaker|boot|gun|rifle|guitar|sword|knife|axe|hammer|tractor|locomotive"

BIG = 1e4


def _rot_tsr_to_view():
    """Same rotations reconstruct.py applies to the mesh: TripoSR frame -> (x right, y up, z to camera)."""
    def R(axis, a):
        c, s = np.cos(a), np.sin(a)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    M = R("y", np.pi) @ R("y", np.pi / 2) @ R("x", -np.pi / 2)
    return np.round(M).astype(int)


R_TV = _rot_tsr_to_view()  # view = R_TV @ tsr  (a signed permutation)


def tsr_to_view(vol: np.ndarray) -> np.ndarray:
    """Re-index a cubic volume sampled on a symmetric grid from TripoSR axes to view axes."""
    perm = [int(np.nonzero(R_TV[a])[0][0]) for a in range(3)]
    v = np.transpose(vol, perm)
    for a in range(3):
        if R_TV[a, perm[a]] < 0:
            v = np.flip(v, a)
    return np.ascontiguousarray(v)


def view_points_to_tsr(p: np.ndarray) -> np.ndarray:
    return p @ R_TV  # R^T applied to row vectors


def _row_extent(row: np.ndarray, c: int):
    """Walk outward from index c along a boolean row until the first gap; return (left, right) reach."""
    n = len(row)
    if not row[c]:
        return 0, 0
    l = c
    while l > 0 and row[l - 1]:
        l -= 1
    r = c
    while r < n - 1 and row[r + 1]:
        r += 1
    return c - l, r - c


def analyse(F: np.ndarray) -> dict:
    """F: view-frame field (x, y, z), >0 inside. Returns silhouette statistics used for auto mode."""
    occ = F > 0
    P = occ.any(2)                       # front silhouette (x, y)
    xs = np.nonzero(P.any(1))[0]
    ys = np.nonzero(P.any(0))[0]
    if len(xs) == 0:
        return dict(empty=True)
    cols = []
    for y in ys:  # centre of the widest contiguous run per row (ignores handles / spouts)
        row = P[:, y].astype(np.int8)
        d = np.diff(np.concatenate([[0], row, [0]]))
        st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        i = int(np.argmax(en - st))
        cols.append((st[i] + en[i] - 1) / 2)
    x0 = int(round(np.median(cols)))
    flipped = np.zeros_like(P)
    src = 2 * x0 - np.arange(P.shape[0])
    ok = (src >= 0) & (src < P.shape[0])
    flipped[ok] = P[src[ok]]
    inter, uni = np.logical_and(P, flipped).sum(), np.logical_or(P, flipped).sum()
    sym_lr = float(inter / max(uni, 1))
    axis_fill = float(P[x0, ys].mean())
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    zs = np.nonzero(occ.any((0, 1)))[0]
    depth = zs.max() - zs.min() + 1
    return dict(empty=False, x0=x0, sym_lr=sym_lr, axis_fill=axis_fill, width=int(width), height=int(height),
                depth=int(depth), y_range=(int(ys.min()), int(ys.max())))


def choose_mode(stats: dict, prompt: str = "") -> str:
    """Fallback when CLIP/depth cues are unavailable — geometry only, no object list required."""
    if stats.get("empty"):
        return "none"
    p = (prompt or "").lower()
    slab = _slabness(stats)
    if re.search(ROUND_WORDS, p) and stats["sym_lr"] > 0.7:
        return "round"
    if re.search(SIDE_WORDS, p) and stats["width"] > 1.15 * stats["height"]:
        return "mirror-depth"
    # Thin + LR-symmetric → lathe (classic TripoSR front-view collapse of upright vessels)
    if slab > 0.35 and stats["sym_lr"] > 0.7:
        return "round"
    if slab > 0.35 and stats["width"] > 1.15 * stats["height"]:
        return "mirror-depth"
    if stats["sym_lr"] > 0.9 and stats["axis_fill"] > 0.97:
        return "round"
    if stats["sym_lr"] > 0.85:
        return "mirror-lr"
    return "none"


def _depth_centre(occ: np.ndarray, x0: int) -> int:
    band = occ[max(0, x0 - 3):x0 + 4].any(0)   # (y, z) occupancy near the axis
    zs = [np.nonzero(r)[0].mean() for r in band if r.any()]
    if not zs:
        zs = [np.nonzero(occ.any((0, 1)))[0].mean()]
    return int(round(np.median(zs)))


def apply_round(F: np.ndarray, x0: int, k: float = 2.0) -> np.ndarray:
    n = F.shape[0]
    occ = F > 0
    P = occ.any(2)
    z0 = _depth_centre(occ, x0)
    r_out = np.zeros(n)
    r_in = np.zeros(n)
    for y in range(n):
        if not P[:, y].any():
            continue
        l, r = _row_extent(P[:, y], x0)
        # the shorter side: a handle/spout attached on one side must not inflate the body
        r_out[y] = min(l, r) + 0.5 if P[x0, y] else 0
        # hollow rows (cup/vase opening): the slice through the axis is empty at the centre
        sl = occ[:, y, z0]
        if not sl[x0] and r_out[y] > 4:
            left = np.nonzero(sl[:x0][::-1])[0]
            right = np.nonzero(sl[x0 + 1:])[0]
            if len(left) and len(right):
                ri = (left[0] + right[0]) / 2.0
                if ri < r_out[y] - 2:
                    r_in[y] = ri
    r_out = median_filter(r_out, size=5)
    r_in = median_filter(r_in, size=5)
    xs = np.arange(n)[:, None] - x0
    zs = np.arange(n)[None, :] - z0
    d = np.sqrt(xs ** 2 + zs ** 2)                     # (x, z) radial distance
    rev = np.empty_like(F)
    for y in range(n):
        if r_out[y] <= 0:
            rev[:, y, :] = -BIG
            continue
        f = r_out[y] - d
        if r_in[y] > 0:
            f = np.minimum(f, d - r_in[y])
        rev[:, y, :] = f * k
    # keep what sticks out of the round body when seen from the front (handles, spouts)
    import cv2
    far = np.abs(np.arange(n)[:, None] - x0) > (r_out[None, :] + 2)       # (x, y) clearly outside the body
    E = (P & far).astype(np.uint8)
    E = cv2.morphologyEx(E, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # drop edge slivers, keep handles
    E = cv2.dilate(E, np.ones((7, 7), np.uint8)) > 0                     # grow back so it re-attaches
    extras = np.where(E[:, :, None], F, -BIG)
    return np.maximum(rev, extras)


def apply_mirror_depth(F: np.ndarray) -> np.ndarray:
    occ = F > 0
    zs = np.nonzero(occ.any((0, 1)))[0]
    zc = (zs.min() + zs.max()) / 2.0
    n = F.shape[2]
    z = np.arange(n)
    src = np.round(2 * zc - z).astype(int)
    out = F.copy()
    back = z < zc                                    # far side (away from camera) is the hidden one
    ok = back & (src >= 0) & (src < n)
    out[:, :, ok] = F[:, :, src[ok]]
    out[:, :, back & ~ok] = -BIG
    return out


def apply_mirror_lr(F: np.ndarray, x0: int) -> np.ndarray:
    n = F.shape[0]
    src = 2 * x0 - np.arange(n)
    ok = (src >= 0) & (src < n)
    mir = np.full_like(F, -BIG)
    mir[ok] = F[src[ok]]
    return 0.5 * (F + mir)


def apply_side_sketch(F: np.ndarray, side_mask: np.ndarray, x0: int, k: float = 2.0) -> np.ndarray:
    """Carve/extend depth with a side-view silhouette (as seen from the object's right, front to the left).
    side_mask: bool (H, W) image. Scaled so its height matches the reconstruction's height."""
    import cv2
    occ = F > 0
    n = F.shape[0]
    ys = np.nonzero(occ.any((0, 2)))[0]
    zs = np.nonzero(occ.any((0, 1)))[0]
    if len(ys) == 0:
        return F
    h_obj = ys.max() - ys.min() + 1
    rows, cols = np.nonzero(side_mask)
    # single strokes (legs, poles) have no area: give them a plausible thickness (~2.5 % of the height)
    rad_px = max(2, int(0.025 * (rows.max() - rows.min() + 1)))
    side_mask = cv2.dilate(side_mask.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * rad_px + 1, 2 * rad_px + 1))) > 0
    rows, cols = np.nonzero(side_mask)
    m = side_mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1].astype(np.uint8)
    scale = h_obj / m.shape[0]
    w_new = max(1, int(round(m.shape[1] * scale)))
    m = cv2.resize(m, (w_new, h_obj), interpolation=cv2.INTER_NEAREST) > 0
    zc = int(round((zs.min() + zs.max()) / 2))
    S = np.zeros((n, n), bool)                       # (y, z) in volume index space
    z_start = zc - w_new // 2
    for j in range(w_new):
        # image column j (left = object's front = +z, towards camera)
        z = z_start + (w_new - 1 - j)
        if 0 <= z < n:
            col = m[:, j][::-1]                     # image rows go down; volume y goes up
            S[ys.min():ys.min() + h_obj, z] = col
    # signed distance of the side silhouette, extruded along x
    S8 = S.astype(np.uint8)
    dist_in = cv2.distanceTransform(S8, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(1 - S8, cv2.DIST_L2, 3)
    sd = (dist_in - dist_out) * k                    # (y, z), >0 inside
    hull = np.broadcast_to(sd[None], F.shape)
    front = occ.any(2)                               # (x, y) front silhouette
    front8 = front.astype(np.uint8)
    fd = (cv2.distanceTransform(front8, cv2.DIST_L2, 3) - cv2.distanceTransform(1 - front8, cv2.DIST_L2, 3)) * k
    hull = np.minimum(hull, fd[:, :, None])          # 2-view visual hull
    # The two drawn outlines are the user's intent: the result is the 2-view visual hull, with TripoSR's
    # surface kept where it is already inside it (so curved detail survives instead of a pure box).
    return np.minimum(hull, np.maximum(F, hull - 1.5 * k))


def image_to_grid(arr: np.ndarray, img_mask: np.ndarray, P0: np.ndarray) -> np.ndarray:
    """Resample an image-space map into the (x, y) grid of the volume by aligning the object's bounding boxes."""
    import cv2
    rr, cc = np.nonzero(img_mask)
    xs, ys = np.nonzero(P0)
    if len(rr) == 0 or len(xs) == 0:
        return np.zeros(P0.shape, np.float32)
    crop = arr[rr.min():rr.max() + 1, cc.min():cc.max() + 1].astype(np.float32)
    w, h = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    crop = cv2.resize(crop, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)[::-1].T   # rows down -> y up, [x, y]
    out = np.full(P0.shape, np.nan, np.float32)
    out[xs.min():xs.min() + w, ys.min():ys.min() + h] = crop
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 50:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def _front_depth(F: np.ndarray):
    occ = F > 0
    has = occ.any(2)
    zf = np.where(has, F.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2), -1).astype(np.float32)
    return has, zf


def _build(F_view, mode, stats):
    occ = F_view > 0
    if mode == "round":
        stats["z_mirror"] = _depth_centre(occ, stats["x0"])
        return apply_round(F_view, stats["x0"])
    if mode == "mirror-depth":
        zs = np.nonzero(occ.any((0, 1)))[0]
        stats["z_mirror"] = (zs.min() + zs.max()) / 2.0
        return apply_mirror_depth(F_view)
    if mode == "mirror-lr":
        return apply_mirror_lr(F_view, stats["x0"])
    return F_view


def _slabness(stats: dict) -> float:
    """How paper-thin TripoSR's volume is vs the silhouette (0 = plausible thickness, 1 = flat extrusion).

    Single-view NeRF often collapses unseen depth into a wall of the front outline. That wall matches the
    silhouette perfectly, so silhouette-only scoring wrongly prefers `none`. Slabness is object-agnostic."""
    ref = max(int(stats.get("width", 1)), int(stats.get("height", 1)), 1)
    depth = float(stats.get("depth", 0))
    return float(np.clip(1.0 - depth / (0.85 * ref), 0.0, 1.0))


def _thickness_fit(mode: str, stats: dict, slab: float) -> float:
    """Object-agnostic vote for which repair should restore plausible 3D thickness."""
    sym = float(stats.get("sym_lr", 0.0))
    wide = float(stats["width"]) > 1.15 * float(stats["height"])
    tall = float(stats["height"]) > 1.15 * float(stats["width"])
    if mode == "none":
        # Raw TripoSR: punish flat extrusions (they look correct from the front, wrong from the side).
        return -slab
    if mode == "round":
        # Lathe: helps when the outline is fairly LR-symmetric and the volume is too thin.
        return slab * max(sym, 0.0) * (0.7 + 0.3 * float(tall or not wide))
    if mode == "mirror-depth":
        # Side-view bilateral: helps when the silhouette is wide and the volume is thin in depth.
        return slab * (0.85 if wide else 0.25)
    if mode == "mirror-lr":
        # Front bilateral: usually already has some depth; slight preference when not a slab.
        return 0.2 * (1.0 - slab) * max(sym - 0.5, 0.0)
    return 0.0


def select_mode(F_view: np.ndarray, stats: dict, cues: dict, prompt: str = ""):
    """Score every plausible symmetry and keep the best, using object-agnostic signals:
       front silhouette IoU, Depth Anything rank-correlation, CLIP *geometry* prior (+ soft prompt
       keywords), and a slabness/thickness term that stops flat TripoSR extrusions from winning."""
    P0 = (F_view > 0).any(2)
    prior = dict(cues.get("probs") or {})
    p = (prompt or "").lower()
    # Prompt keywords are a *soft* hint only (user may leave prompt empty; judge images vary).
    if re.search(ROUND_WORDS, p):
        prior["round"] = prior.get("round", 0) + 0.25
    if re.search(SIDE_WORDS, p):
        prior["mirror-depth"] = prior.get("mirror-depth", 0) + 0.25
    tot = sum(prior.values()) or 1.0
    prior = {k: v / tot for k, v in prior.items()}
    Dg = cues.get("depth_grid")
    slab = _slabness(stats)
    stats = dict(stats)
    stats["slabness"] = round(slab, 3)

    cands = ["none"]
    # Allow round even with a hole on the axis (cups) or when the volume is suspiciously flat.
    if (stats["sym_lr"] > 0.55 and stats["axis_fill"] > 0.45) or (slab > 0.35 and stats["sym_lr"] > 0.55):
        cands.append("round")
    if stats["sym_lr"] > 0.65:
        cands.append("mirror-lr")
    cands.append("mirror-depth")

    report, best = {}, None
    for m in cands:
        st = dict(stats)
        Fm = _build(F_view, m, st)
        has, zf = _front_depth(Fm)
        uni = np.logical_or(has, P0).sum()
        sil = float(np.logical_and(has, P0).sum() / max(uni, 1))
        corr = 0.0
        if Dg is not None:
            R = has & P0 & np.isfinite(Dg)
            corr = _spearman(zf[R], Dg[R])
        thick = _thickness_fit(m, stats, slab)
        # When TripoSR already has thickness, lean on silhouette+depth+CLIP.
        # When it collapsed to a slab, thickness dominates so `none` cannot win on silhouette alone.
        w_sil, w_dep, w_pri, w_th = (0.28, 0.22, 0.30, 0.20) if slab < 0.25 else (0.22, 0.18, 0.25, 0.35)
        score = (w_sil * sil + w_dep * max(corr, 0.0) + w_pri * prior.get(m, 0.0) + w_th * thick)
        report[m] = dict(silhouette=round(sil, 3), depth_corr=round(corr, 3),
                         prior=round(prior.get(m, 0.0), 3), thickness=round(thick, 3),
                         score=round(score, 3))
        if best is None or score > best[0]:
            best = (score, m, Fm, st)
    return best[1], best[2], best[3], report



def apply(F_view: np.ndarray, mode: str = "auto", prompt: str = "", side_mask: np.ndarray = None,
          cues: dict = None):
    stats = analyse(F_view)
    if stats.get("empty"):
        return F_view, "none", stats
    if mode == "auto" and cues:
        mode, out, stats, stats["candidates"] = select_mode(F_view, stats, cues, prompt)
    else:
        if mode == "auto":
            mode = choose_mode(stats, prompt)
        out = _build(F_view, mode, stats)
    if side_mask is not None and side_mask.any():
        out = apply_side_sketch(out, side_mask, stats["x0"])
        mode += "+side-sketch"
    return out, mode, stats
