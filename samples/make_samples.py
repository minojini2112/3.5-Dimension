"""Generate hand-drawn-looking test sketches (wobbly strokes, uneven pressure) for reproducible testing."""
import os
import cv2
import numpy as np

rng = np.random.default_rng(0)
OUT = os.path.dirname(os.path.abspath(__file__))


def wobble(pts, amp=2.0, n=6):
    pts = np.asarray(pts, np.float32)
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, n, endpoint=False):
            dense.append(a + (b - a) * t)
    dense.append(pts[-1])
    dense = np.array(dense)
    noise = np.cumsum(rng.normal(0, amp * 0.4, dense.shape), 0)
    noise -= np.linspace(noise[0], noise[-1], len(noise))
    return dense + noise * 0.5 + rng.normal(0, amp * 0.3, dense.shape)


def stroke(img, pts, closed=False, th=3, amp=2.0):
    p = list(pts) + ([pts[0]] if closed else [])
    w = wobble(p, amp).astype(np.int32)
    cv2.polylines(img, [w], False, 30, th + int(rng.integers(0, 2)), cv2.LINE_AA)


def ellipse_pts(c, ax, a0=0, a1=360, n=40):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([c[0] + ax[0] * np.cos(t), c[1] + ax[1] * np.sin(t)], -1)


def canvas():
    return np.full((768, 768), 250, np.uint8)


def mug():
    im = canvas()
    stroke(im, ellipse_pts((360, 220), (160, 45)), True)
    stroke(im, [(200, 220), (215, 600)]); stroke(im, [(520, 220), (505, 600)])
    stroke(im, ellipse_pts((360, 600), (145, 40), 0, 180))
    stroke(im, ellipse_pts((560, 400), (90, 110), -90, 90))
    stroke(im, ellipse_pts((545, 400), (50, 70), -90, 90))
    return im


def lamp():
    im = canvas()
    stroke(im, [(250, 330), (300, 140), (470, 140), (520, 330)], True)
    stroke(im, [(385, 330), (385, 600)]); stroke(im, [(370, 330), (370, 600)])
    stroke(im, ellipse_pts((380, 620), (130, 30)), True)
    return im


def car():
    im = canvas()
    body = [(80, 470), (90, 390), (230, 360), (300, 280), (480, 280), (570, 360), (690, 380), (700, 470)]
    stroke(im, body + [(620, 470)])
    stroke(im, [(250, 470), (480, 470)]); stroke(im, [(80, 470), (130, 470)])
    stroke(im, ellipse_pts((190, 470), (60, 60)), True); stroke(im, ellipse_pts((560, 470), (60, 60)), True)
    stroke(im, [(320, 300), (290, 355), (390, 355), (390, 300)], True)
    stroke(im, [(410, 300), (410, 355), (530, 355), (470, 300)], True)
    return im


def chair():
    im = canvas()
    stroke(im, [(260, 120), (480, 120), (480, 400), (260, 400)], True)
    stroke(im, [(220, 400), (520, 400), (560, 460), (190, 460)], True)
    stroke(im, [(210, 460), (210, 680)]); stroke(im, [(540, 460), (540, 680)])
    stroke(im, [(270, 460), (275, 620)]); stroke(im, [(480, 460), (475, 620)])
    return im


def vase():
    im = canvas()
    left = [(330, 120), (320, 200), (260, 300), (230, 420), (260, 560), (320, 640)]
    right = [(770 - x, y) for x, y in left]
    stroke(im, left); stroke(im, right)
    stroke(im, ellipse_pts((385, 120), (55, 15)), True)
    stroke(im, [(320, 640), (450, 640)])
    return im


if __name__ == "__main__":
    for name, fn in dict(mug=mug, lamp=lamp, car=car, chair=chair, vase=vase).items():
        cv2.imwrite(os.path.join(OUT, f"{name}.png"), fn())
        print("wrote", name)


def chair_side():
    """Side view of the chair (front of the chair on the left), for the optional side-view input."""
    im = canvas()
    stroke(im, [(470, 120), (520, 120), (520, 400), (470, 400)], True)       # back rest, edge-on
    stroke(im, [(220, 400), (520, 400), (520, 460), (220, 460)], True)       # seat
    stroke(im, [(230, 460), (230, 680)]); stroke(im, [(505, 460), (505, 680)])
    return im


if __name__ == "__main__":
    cv2.imwrite(os.path.join(OUT, "chair_side.png"), chair_side())
