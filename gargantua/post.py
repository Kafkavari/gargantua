"""Tone mapping and camera/film imperfections (CPU, NumPy + OpenCV)."""
import numpy as np
import cv2


def aces(x):
    """ACES filmic curve (Narkowicz fit), clipped to [0, 1]."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def tonemap_simple(img, exposure):
    """Plain bloom + ACES + gamma (presets without `film`)."""
    img = img * exposure
    b = np.zeros_like(img)
    Hh = img.shape[0]
    for s, w in ((0.004, 0.5), (0.015, 0.35), (0.05, 0.25)):
        b += w * cv2.GaussianBlur(np.maximum(img - 0.6, 0).astype(np.float32), (0, 0), s * Hh)
    img = img + b
    m = aces(img)
    return (np.clip(m, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)


def tonemap_film(img, prm, frame):
    """Camera/film imperfections on the HDR image: wide glare, red halation, a veil of scattered light,
    chromatic aberration, vignette, ACES, then light-scaled grain (stays black on black pixels).
    `frame` seeds the grain, so every frame gets different grain."""
    H, W = img.shape[:2]
    x = (img * prm["exposure"]).astype(np.float32)
    bright = np.maximum(x - prm["bloom_thr"], 0)
    b = np.zeros_like(x)
    for s, w in ((0.003, 0.55), (0.012, 0.4), (0.04, 0.3), (0.12, 0.22)):
        b += w * cv2.GaussianBlur(bright, (0, 0), s * H)
    hal = cv2.GaussianBlur(np.maximum(x - 0.7, 0), (0, 0), 0.007 * H) * np.array([1.0, 0.33, 0.12], np.float32)
    veil = cv2.GaussianBlur(x, (0, 0), 0.2 * H)
    x = x + prm["bloom"] * b + prm["halation"] * hal + prm["scatter"] * veil
    # chromatic aberration: red scaled outwards, blue inwards (centre = image centre)
    ca = prm["ca"]
    if ca > 0:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W / 2, H / 2
        for k, s in ((0, 1 + ca), (2, 1 - ca)):
            mx = cx + (xx - cx) / s
            my = cy + (yy - cy) / s
            x[..., k] = cv2.remap(x[..., k], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 * 0.6 + ((yy - H / 2) / (H / 2)) ** 2)
    x = x * (1 - prm["vignette"] * np.clip(rr - 0.4, 0, 1) ** 2)[..., None]
    m = aces(x) ** (1 / 2.2)
    # grain: different every frame, scaled with brightness
    rng = np.random.default_rng(1000 + frame)
    g = rng.normal(0, 1, (H, W, 1)).astype(np.float32) * 0.75 + rng.normal(0, 1, (H, W, 3)).astype(np.float32) * 0.25
    g = cv2.GaussianBlur(g, (0, 0), 0.6)
    m = m + prm["grain"] * g * np.sqrt(np.clip(m.mean(-1, keepdims=True), 0, 1))
    return (np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)


def tonemap(img, prm, frame=0):
    """HDR float image -> uint8 sRGB-ish image, using the film chain when the preset asks for it."""
    if not prm.get("film", False):
        return tonemap_simple(img, prm["exposure"])
    return tonemap_film(img, prm, frame)
