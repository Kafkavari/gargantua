"""Geodesic ray map for a Schwarzschild black hole (CuPy).

Units: G = c = M = 1, event horizon at r = 2. The camera is fixed, so the ray map
(where every pixel's ray crosses the disk volume, and how it ends) is computed once
and reused for every frame; only the shading changes per frame.

Light follows null geodesics of the Schwarzschild metric. With angular momentum
h = |x cross v| the acceleration of a ray is
    x'' = -1.5 h^2 x / r^5
which is integrated with a leapfrog scheme (adaptive step, finer inside the disk).
"""
import time
import numpy as np
import cupy as cp

# Vertical offsets (in units of the local disk half-thickness) of the sheets used for
# in-cloud star lookups.
LAYERS = (-1.3, -0.65, 0.0, 0.65, 1.3)


def thickness(r, prm):
    """Disk half-thickness at cylindrical radius r."""
    return prm["thick_slope"] * r + prm["thick_base"]


def camera_position(prm):
    e = np.radians(prm["tilt"])
    return np.array([0.0, prm["distance"] * np.sin(e), prm["distance"] * np.cos(e)])


def camera_rays(W, H, prm):
    """Unit ray directions for a W x H image, plus the camera position.

    The camera sits in the x=0 plane looking at the origin; `tilt` is the elevation
    above the disk plane in degrees, `center_y` places the hole vertically on screen
    (0 = top, 1 = bottom), `roll` rotates the frame about the view axis.
    """
    pos = camera_position(prm)
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    t = np.tan(np.radians(prm["fov"]) / 2)
    v = (prm["center_y"] - (np.arange(H) + 0.5) / H) * 2 * t
    u = ((np.arange(W) + 0.5) / W - 0.5) * 2 * t * W / H
    U, V = np.meshgrid(u, v)
    ro = np.radians(prm.get("roll", 0.0))
    U, V = U * np.cos(ro) - V * np.sin(ro), U * np.sin(ro) + V * np.cos(ro)
    d = fwd[None, None] + U[..., None] * right + V[..., None] * up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return pos, d.reshape(-1, 3)


def build_ray_map(W, H, prm, y0=0, y1=None):
    """Trace every ray; return the sample/crossing arrays used by the painter.

    y0..y1 restrict the work to a band of image rows (used for full-resolution
    renders that do not fit in GPU memory at once).
    """
    t0 = time.time()
    pos, d = camera_rays(W, H, prm)
    d = d[y0 * W:(H if y1 is None else y1) * W]
    N = len(d)
    posg = cp.asarray(pos)
    x = cp.repeat(posg[None], N, 0)
    v = cp.asarray(d, dtype=cp.float64)
    h2 = cp.sum(cp.cross(x, v) ** 2, axis=1)       # angular momentum squared
    active = cp.arange(N)
    r_out, Y = prm["r_out"], prm["y_cap"]
    mid = cp.zeros(N, cp.int8)                     # mid-plane crossings on the camera side
    count = cp.zeros(N, cp.int32)                  # samples stored per ray
    parts = []      # (ray, r, phi, yn, ds, secondary, back, iteration)
    L2 = len(LAYERS) * 2
    lr = cp.full((N, L2), cp.nan, cp.float32)      # sheet crossings: radius
    lp = cp.zeros((N, L2), cp.float32)             # ... azimuth
    lk = cp.zeros((N, L2), cp.int32)               # ... number of samples before it
    lm = cp.zeros((N, L2), cp.float32)             # ... distance from the camera
    ls = cp.zeros((N, L2), cp.int8)                # ... behind the hole (secondary image)
    outcome = cp.zeros(N, cp.int8)                 # 0 running, 1 horizon, 2 escaped
    escape = cp.zeros((N, 3), cp.float32)          # escape direction
    rs = cp.random.RandomState(42)
    for it in range(20000):
        if len(active) == 0:
            break
        xa, va, ha = x[active], v[active], h2[active]
        r = cp.linalg.norm(xa, axis=1)
        rc = cp.sqrt(xa[:, 0] ** 2 + xa[:, 2] ** 2)
        h = thickness(rc, prm)
        inside = (cp.abs(xa[:, 1]) < Y * h) & (rc > 2.06) & (rc < r_out)
        dt = cp.clip(prm["step_k"] * r, 0.01, 25.0)
        vy = cp.maximum(cp.abs(va[:, 1]), 0.02)
        dt = cp.where(inside, cp.minimum(cp.minimum(dt, prm["inner_step"] * r), 0.45 * h / vy), dt)
        near = (~inside) & (cp.abs(xa[:, 1]) < 3 * Y * h) & (rc < r_out)
        dt = cp.where(near, cp.minimum(dt, cp.maximum(0.5 * (cp.abs(xa[:, 1]) - Y * h) / vy, 0.3 * h / vy)), dt)
        # jittered step: a fixed fine noise instead of visible banding
        dt = cp.where(inside, dt * (0.6 + 0.8 * rs.random_sample(len(dt))), dt)
        a = -1.5 * ha[:, None] * xa / (r ** 5)[:, None]
        vh = va + 0.5 * dt[:, None] * a
        xn = xa + dt[:, None] * vh
        rn = cp.linalg.norm(xn, axis=1)
        an = -1.5 * ha[:, None] * xn / (rn ** 5)[:, None]
        vn = vh + 0.5 * dt[:, None] * an
        # store volume samples while the ray is inside the disk
        gi = cp.nonzero(inside & (count[active] < prm["max_samples"]))[0]
        if len(gi):
            idx = active[gi]
            m = 0.5 * (xa[gi] + xn[gi])
            rm = cp.sqrt(m[:, 0] ** 2 + m[:, 2] ** 2)
            back = (m[:, 0] * pos[0] + m[:, 2] * pos[2]) < 0
            parts.append((idx.astype(cp.int32), rm.astype(cp.float32), cp.arctan2(m[:, 2], m[:, 0]).astype(cp.float32),
                          (m[:, 1] / thickness(rm, prm)).astype(cp.float32), dt[gi].astype(cp.float32),
                          cp.where(mid[idx] >= 1, (m[:, 0] * pos[0] + m[:, 2] * pos[2]) / np.hypot(pos[0], pos[2]), 99.0).astype(cp.float32),
                          back.astype(cp.int8), cp.full(len(gi), it, cp.int32)))
            count[idx] += 1
        # star sheets: record the first two crossings of each sheet
        rcn = cp.sqrt(xn[:, 0] ** 2 + xn[:, 2] ** 2)
        hn = thickness(rcn, prm)
        for j, c in enumerate(LAYERS):
            f0 = xa[:, 1] - c * h
            f1 = xn[:, 1] - c * hn
            g = cp.nonzero(f0 * f1 < 0)[0]
            if len(g) == 0:
                continue
            idx = active[g]
            for s in range(2):
                sl = j * 2 + s
                empty = cp.isnan(lr[idx, sl])
                gb, ib = g[empty], idx[empty]
                f = f0[gb] / (f0[gb] - f1[gb])
                p = xa[gb] + f[:, None] * (xn[gb] - xa[gb])
                lr[ib, sl] = cp.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2)
                lp[ib, sl] = cp.arctan2(p[:, 2], p[:, 0])
                lk[ib, sl] = count[ib]
                lm[ib, sl] = cp.linalg.norm(p - posg, axis=1)
                ls[ib, sl] = ((p[:, 0] * pos[0] + p[:, 2] * pos[2]) < 0) & (mid[ib] >= 1)
                g, idx = g[~empty], idx[~empty]
                if len(g) == 0:
                    break
        o0 = (xa[:, 1] * xn[:, 1] < 0) & ((xa[:, 0] * pos[0] + xa[:, 2] * pos[2]) > 0)
        mid[active[o0]] = cp.minimum(mid[active[o0]] + 1, 100).astype(cp.int8)
        x[active], v[active] = xn, vn
        horizon = rn < 2.05
        esc = (rn > r_out * 1.15) & (cp.sum(xn * vn, axis=1) > 0)
        outcome[active[horizon]] = 1
        outcome[active[esc]] = 2
        escape[active[esc]] = (vn[esc] / cp.linalg.norm(vn[esc], axis=1, keepdims=True)).astype(cp.float32)
        active = active[~(horizon | esc)]
    # order samples by ray (samples of one ray keep their step order)
    cat = [cp.concatenate([p[i] for p in parts]) for i in range(8)]
    o = cp.lexsort(cp.stack([cat[7], cat[0]]))
    ray, sr, sp, sy, sd, sec, sback, _ = [c[o] for c in cat]
    start = cp.zeros(N + 1, cp.int64)
    start[1:] = cp.cumsum(count)
    print(f"ray map {W}x{H}: {time.time() - t0:.1f} s, samples {len(sr) / 1e6:.1f} M", flush=True)
    return dict(ray=ray, r=sr, phi=sp, yn=sy, ds=sd, sec=sec, back=sback, count=count, start=start,
                lay_r=lr, lay_phi=lp, lay_k=lk, lay_m=lm, lay_s=ls, outcome=outcome, escape=escape,
                impact=cp.sqrt(h2).astype(cp.float32))
