"""Shared helpers for the render scripts."""
import numpy as np
import cupy as cp

from . import build_ray_map, paint, backdrop_stars, tonemap, VARS

ASPECT = 3120 / 1440   # phone wallpaper, height / width


def image_height(width):
    return int(width * ASPECT)


def make_prm(preset, loop=True):
    """Copy of a preset; loop=True blends the time phases so that t=1 equals t=0."""
    return dict(VARS[preset], loop=loop)


def build_bands(prm, W, H, n_bands):
    """Ray maps for n_bands horizontal bands. With a single band everything stays on the GPU;
    with several, each band is moved to host memory and uploaded again when painted."""
    bounds = np.linspace(0, H, n_bands + 1).astype(int)
    bands = []
    for b in range(n_bands):
        rm = build_ray_map(W, H, prm, bounds[b], bounds[b + 1])
        bd = backdrop_stars(rm, prm, H)
        if n_bands > 1:
            rm = {k: cp.asnumpy(v) for k, v in rm.items()}
            bd = cp.asnumpy(bd)
            cp.get_default_memory_pool().free_all_blocks()
        bands.append((rm, bd))
    return bounds, bands


def paint_frame(prm, fields, bounds, bands, z, W, H):
    """Paint all bands at loop time z and return the stitched float HDR image (H, W, 3)."""
    if len(bands) == 1:
        rm, bd = bands[0]
        return cp.asnumpy(paint(rm, prm, z, fields, bd, W, H))
    img = np.zeros((H, W, 3), np.float32)
    for b, (rm_h, bd_h) in enumerate(bands):
        rm = {k: cp.asarray(v) for k, v in rm_h.items()}
        img[bounds[b]:bounds[b + 1]] = cp.asnumpy(paint(rm, prm, z, fields, cp.asarray(bd_h), W, H))
        del rm
        cp.get_default_memory_pool().free_all_blocks()
    return img


def finish_frame(img, prm, frame):
    return tonemap(img.astype(np.float64), prm, frame)
