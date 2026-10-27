"""Periodic noise fields (clouds, shadow, patches) and procedural value noise (CuPy/NumPy).

Fields are sampled in (u, phi, y) with u = ln(r), so they are periodic in the azimuth
phi and stretched along the orbit; they are generated with FFT-filtered noise.
"""
import os
import time
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import map_coordinates

# log-radius range covered by the fields (r = 4 .. 480)
U0, U1 = np.log(4.0), np.log(480.0)


def cloud_field(NU, NP, NY, slope, stretch, seed, thick_slope):
    """Gaussian random field with power-law spectrum k^-slope, periodic in all axes.
    `stretch` elongates structure along the orbit; the y axis is in physical units."""
    rng = np.random.default_rng(seed)
    n = rng.normal(size=(NU, NP, NY)).astype(np.float32)
    F = np.fft.rfftn(n)
    ku = 2 * np.pi * np.fft.fftfreq(NU, d=(U1 - U0) / NU)
    kp = 2 * np.pi * np.fft.fftfreq(NP, d=2 * np.pi / NP)
    ky = 2 * np.pi * np.fft.rfftfreq(NY, d=5.0 * thick_slope / NY)   # y/h axis in physical scale (thickness = slope*r)
    K = np.sqrt(ku[:, None, None] ** 2 + (kp[None, :, None] * stretch) ** 2 + ky[None, None, :] ** 2)
    A = np.where(K > 2.0, np.maximum(K, 1e-6) ** -slope, 0.0)
    f = np.fft.irfftn(F * A, s=n.shape, axes=(0, 1, 2)).astype(np.float32)
    return (f - f.mean()) / f.std()


def shadow_field(coarse, fine, threshold, kappa):
    """Optical depth accumulated inward-to-outward along u: how much light a cloud
    column still receives (cheap self-shadowing)."""
    cover = np.clip(coarse * 0.8 + 0.35 * fine, -3, 3)
    t = np.clip((cover - (threshold - 0.4)) / 1.0, 0, 1)
    D = t * t * (3 - 2 * t)
    du = (U1 - U0) / coarse.shape[0]
    return np.exp(-kappa * np.cumsum(D, axis=0) * du).astype(np.float32)


def build_fields(prm, cache_dir=None):
    """Return (coarse, fine, shadow, patch) fields on the GPU. The expensive FFT fields
    are cached as .npz in cache_dir when given."""
    p = os.path.join(cache_dir, f"clouds_{prm['ny']}.npz") if cache_dir else None
    if p and os.path.exists(p):
        z = np.load(p)
        a, b, patch = z["a"], z["b"], z["patch"]
    else:
        t0 = time.time()
        a = cloud_field(384, 1536, prm["ny"], 1.3, 1.4, 1, prm["thick_slope"])
        b = cloud_field(384, 1536, prm["ny"], 0.8, 1.0, 2, prm["thick_slope"])
        patch = cloud_field(48, 96, 8, 2.4, 1.0, 5, prm["thick_slope"] * 4)
        if p:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(p, a=a, b=b, patch=patch)
        print(f"cloud fields {time.time() - t0:.1f} s", flush=True)
    shadow = shadow_field(a, b, prm["threshold"], prm["kappa"])
    return tuple(cp.asarray(z) for z in (a, b, shadow, patch))


def sample_field(field, u, phi, yn):
    """Trilinear lookup, wrapping in u and phi."""
    NU, NP, NY = field.shape
    cu = (u - U0) / (U1 - U0) * NU
    cph = cp.mod(phi, 2 * np.pi) / (2 * np.pi) * NP
    cy = (cp.clip(yn, -2.5, 2.5) + 2.5) / 5.0 * (NY - 1)
    return map_coordinates(field, cp.stack([cu, cph, cy]), order=1, mode="grid-wrap")


def _hash3(i, j, k):
    a = cp.sin(i * 127.1 + j * 311.7 + k * 74.7) * 43758.5453
    return a - cp.floor(a)


def value_noise(u, phi, yn, Pu, Pp, Ky):
    """Periodic 3D value noise with Pu x Pp lattice cells over the (u, phi) domain."""
    x = (u - U0) / (U1 - U0) * Pu
    y = cp.mod(phi, 2 * np.pi) / (2 * np.pi) * Pp
    z = yn * Ky
    xi, yi, zi = cp.floor(x), cp.floor(y), cp.floor(z)
    fx, fy, fz = x - xi, y - yi, z - zi
    fx, fy, fz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    xi1, yi1 = cp.mod(xi + 1, Pu), cp.mod(yi + 1, Pp)
    xi, yi = cp.mod(xi, Pu), cp.mod(yi, Pp)
    L = lambda a, b, t: a + (b - a) * t
    c00 = L(_hash3(xi, yi, zi), _hash3(xi1, yi, zi), fx)
    c10 = L(_hash3(xi, yi1, zi), _hash3(xi1, yi1, zi), fx)
    c01 = L(_hash3(xi, yi, zi + 1), _hash3(xi1, yi, zi + 1), fx)
    c11 = L(_hash3(xi, yi1, zi + 1), _hash3(xi1, yi1, zi + 1), fx)
    return L(L(c00, c10, fy), L(c01, c11, fy), fz)


def erosion(u, phi, yn, ds, prm):
    """Fine-scale erosion of the cloud edges. Octaves fade in only where the ray step
    is small enough to resolve them (avoids aliasing at low resolution)."""
    r = cp.exp(u)
    hu = r * (U1 - U0) / 384
    t = 0.0
    for octave, amp, ky in ((3, 1.4, prm["erosion_y"]), (7, 0.8, prm["erosion_y"] * 2.3), (15, 0.45, prm["erosion_y"] * 5)):
        w = cp.clip(hu / octave / cp.maximum(ds, 1e-6) - 0.5, 0, 1)
        t = t + (value_noise(u, phi, yn, 384 * octave, 1536 * octave, ky) - 0.5) * amp * w
    return t
