"""Volume shading: density and emission of the disk, stars, and the per-frame painter (CuPy).

Everything here is artistic rather than physical: the ray *paths* come from the geodesic
integration in raytrace.py, but colours, brightness, the Doppler term and the cloud
structure are tuned by eye. Time `z` runs over one loop, z in [0, 1); every moving
feature uses an integer number of cycles per loop so the video repeats seamlessly.
"""
import numpy as np
import cupy as cp

from .fields import U0, U1, sample_field, erosion
from .raytrace import thickness, camera_position

B_CRIT = 3 * np.sqrt(3)   # critical impact parameter of the photon ring


def smooth(a, b, x):
    """Smoothstep from a to b."""
    t = cp.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def blackbody(T):
    """Approximate linear RGB of a blackbody at T kelvin (Tanner Helland fit), 1000..40000 K."""
    t = T / 100.0
    r = cp.where(t <= 66, 255.0, 329.698727446 * cp.power(cp.maximum(t - 60, 1e-3), -0.1332047592))
    g = cp.where(t <= 66, 99.4708025861 * cp.log(cp.maximum(t, 1e-3)) - 161.1195681661,
                 288.1221695283 * cp.power(cp.maximum(t - 60, 1e-3), -0.0755148492))
    b = cp.where(t >= 66, 255.0, cp.where(t <= 19, 0.0, 138.5177312231 * cp.log(cp.maximum(t - 10, 1e-3)) - 305.0447927307))
    c = cp.clip(cp.stack([r, g, b], -1), 0, 255) / 255.0
    return c ** 2.2


# ---------------- density / emission ----------------
def _phase(r, phi, yn, ds, prm, z, fields):
    """Field lookups at one time phase: n1, n2, erosion, arms, shadow, patches, bands, dust, heat noise."""
    coarse, fine, shadow, patch = fields
    r_in, m = prm["r_in"], prm["arms"]
    u = cp.log(r)
    ph = phi - 2 * np.pi * prm["spin"] * (r / r_in) ** -1.5 * z      # Keplerian-like differential rotation
    ua = u + prm["inflow"] * z
    if prm.get("shimmer", 0) > 0:
        # heat shimmer near the hole: texture lookup wobbles (integer frequencies in z -> seamless loop)
        env = 1 - smooth(r_in * 1.2, prm["r_flow"] * 1.5, r)
        f1, f2 = prm.get("shimmer_f", (3, -2))
        wob = cp.sin(2 * np.pi * (f1 * z + r / (r_in * 0.55)) + 3 * phi) + 0.6 * cp.sin(2 * np.pi * (f2 * z + r / (r_in * 0.3)) - 5 * phi + 1.7)
        ph = ph + prm["shimmer"] * env * wob * (r_in / r)
        ua = ua + prm["shimmer"] * 0.4 * env * wob
    n1 = sample_field(coarse, ua, ph, yn)
    n2 = sample_field(fine, ua, ph, yn)
    er = erosion(ua, ph, yn, ds, prm)
    tp = np.tan(np.radians(prm["arm_pitch"]))
    arm = (0.5 + 0.5 * cp.cos(m * ph + m * u / tp + 0.6 * n1)) ** 3
    # flow region: bright arms with dark dust lanes in between
    psi_a = m * ph + m * ua / np.tan(np.radians(prm["flow_pitch"])) + 0.8 * n1
    band = (0.5 + 0.5 * cp.cos(psi_a)) ** 2
    dust = (0.5 + 0.5 * cp.cos(psi_a + np.pi * 0.85)) ** 6 * smooth(-0.3, 0.8, n2)
    # heat exchange: a second gas layer counter-rotating and flowing inward; mixes with n1 into hot/cold tongues
    if prm.get("heat", False):
        ph2 = phi + 2 * np.pi * prm["counter_spin"] * (r / r_in) ** -1.0 * z + 0.9 * n1
        ua2 = u - prm["counter_inflow"] * z + 0.25 * n2
        heat_n = 0.65 * sample_field(coarse, ua2 * 1.0 + 0.37, ph2 * 1.0 + 1.3, yn * 0.7) + 0.45 * sample_field(fine, ua2 + 0.11, ph2 + 2.1, yn)
    else:
        heat_n = cp.zeros_like(r)
    return n1, n2, er, arm, sample_field(shadow, ua, ph, yn), sample_field(patch, ua, ph, yn), band, dust, heat_n


def density_emission(r, phi, yn, ds, prm, z, fields):
    """Extinction `sig` and emitted colour `em` at samples (r, phi, yn) at loop time z."""
    r_in, r_flow = prm["r_in"], prm["r_flow"]
    A = _phase(r, phi, yn, ds, prm, z, fields)
    if prm.get("loop", False):
        # loop: phases z and z-1 are blended at field level, variance-preserving (no blurry double image)
        B = _phase(r, phi, yn, ds, prm, z - 1.0, fields)
        w = z
        nrm = 1 / np.sqrt((1 - w) ** 2 + w ** 2)
        n1, n2, er = [(a * (1 - w) + b * w) * nrm for a, b in zip(A[:3], B[:3])]
        arm, shad, patch_v, band, dust = [a * (1 - w) + b * w for a, b in zip(A[3:8], B[3:8])]
        heat_n = (A[8] * (1 - w) + B[8] * w) * nrm
    else:
        n1, n2, er, arm, shad, patch_v, band, dust, heat_n = A
    vert = cp.exp(-yn ** 2 * prm["vert_exp"])
    w_in = 1 - smooth(r_flow * 0.9, r_flow * 2.0, r)
    w_fl = smooth(r_flow * 0.6, r_flow * 1.4, r) * (1 - smooth(r_flow * 3, r_flow * 7, r))
    w_gx = smooth(r_flow * 2.5, r_flow * 6, r) * (1 - smooth(prm["r_out"] * 0.75, prm["r_out"], r))
    x = r / r_in
    prof = cp.clip(1 - cp.sqrt(1 / cp.maximum(x, 1.0001)), 0, 1) * x ** -2.2 / 0.0603
    cover = cp.clip(n1 * 0.8 + 0.35 * n2 + prm["erosion"] * er, -3, 3)
    cloud = smooth(prm["threshold"] - 0.4, prm["threshold"] + 0.6, cover)
    cloud_g = smooth(prm["threshold"] - 0.2, prm["threshold"] + 0.8, cover + 1.4 * arm - 0.7)
    # waves travelling outward now and then (wave_n per loop, two independent waves); irregular rings
    wave = cp.zeros_like(r)
    for k0 in (0.0, 0.47):
        f = cp.mod(z * prm["wave_n"] + k0, 1.0)
        rp = r_in * 0.5 + f * (r_flow * 1.6 - r_in * 0.5)
        env = cp.sin(np.pi * f) ** 1.5
        rr = r * (1 + 0.07 * cp.sin(3 * phi + 1.3 + k0 * 9) + 0.04 * cp.sin(5 * phi - 0.7))
        d = rr - rp
        width = r_in * 0.35 * (0.6 + f)
        wave = wave + env * cp.exp(-(d / width) ** 2) * cp.cos(d / width * 3.0)   # signed: crest +, trough -
    # haze reaches down to just outside the horizon: rays below the shadow cross the disk plane at r ~ 2.1-2.7
    haze_zone = smooth(2.1, r_in * 0.6, r) * (1 - smooth(r_in * 0.9, r_in * 1.3, r))
    wave = wave * (1 - smooth(r_flow * 1.0, r_flow * 1.6, r))
    haze_d = haze_zone * (0.4 + 0.6 * cloud) * prm["haze_density"]
    # arm_front > 0: the flow region shows distinct galaxy arms (bright arms, empty between) instead of dust clouds
    front = prm.get("arm_front", 0.0)
    arm_mask = smooth(prm.get("arm_thr0", 0.2), prm.get("arm_thr1", 0.7), band)
    cloud_a = cloud * (1 - front + front * arm_mask * 1.3)
    dens = vert * (haze_d + w_in * (0.12 + 1.1 * cloud) * prof * 1.2 + w_fl * (cloud_a * (0.5 + 0.8 * band) + prm["dust"] * dust) + w_gx * cloud_g * 0.8)
    scale = 1 / (0.15 * r + 1)
    sig = dens * prm["sigma"] * scale
    light = (r_flow / cp.maximum(r, r_in)) ** prm["light_exp"] * (0.04 + 0.96 * shad)
    patch = prm["patch_min"] + (1 - prm["patch_min"]) * smooth(-0.5, 0.7, patch_v)
    light = light * patch
    heat = prm.get("heat", False)
    if heat:
        # temperature contrast gives the sense of heat; the base temperature falls with r, modulated by exp(mixing)
        T_base = cp.maximum(prm["T0"] * x ** -prm["T_exp"], prm["T_flow"] * (r_flow / r) ** 0.7)
        mix = cp.clip(0.7 * heat_n + 0.3 * n1 + 0.25 * cp.clip(patch_v, -1.5, 1.5), -2.5, 2.5)
        T_mod = cp.exp(prm["T_wave"] * mix)
        T_gas = cp.clip(T_base * T_mod, 900, 30000)
        if prm.get("T_inner_min"):
            T_gas = cp.where(r < r_in * 1.6, cp.maximum(T_gas, prm["T_inner_min"] * (1 - smooth(r_in * 1.2, r_in * 1.6, r)) + 900), T_gas)
        heat_bright = cp.clip(T_mod, 0.3, 3.0) ** prm["T_bright_exp"]
        if prm.get("gradient") is not None:
            # ONE disk, ONE colour gradient: cream-white centre -> amber -> orange -> rust -> dark brown at the rim.
            # The gradient position is bent by noise; a mask sharpens band edges in some regions.
            stops = cp.asarray(prm["gradient"])
            t = (cp.log(r) - np.log(2.2)) / (np.log(prm["grad_R"]) - np.log(2.2))
            t = cp.clip(t + prm["grad_warp"] * (0.55 * cp.clip(heat_n, -2.5, 2.5) + 0.45 * cp.clip(n1, -2.5, 2.5)), 0, 1)
            K = prm["grad_band"]
            f = t * K
            i0 = cp.floor(f)
            fr = f - i0
            sharp = smooth(0.0, 0.9, 0.8 * cp.clip(patch_v, -2, 2) + 0.4 * cp.clip(n2, -2, 2))
            w = 0.5 - (0.5 - 0.04) * sharp
            fr = smooth(0.5 - w, 0.5 + w, fr)
            tb = cp.clip((i0 + fr) / K, 0, 1)
            ns = stops.shape[0] - 1
            j = cp.clip(cp.floor(tb * ns), 0, ns - 1).astype(cp.int32)
            frac = tb * ns - j
            gcol = stops[j] * (1 - frac)[:, None] + stops[j + 1] * frac[:, None]
            gpar = cp.clip(heat_bright, 0.7, 1.6)
            c_in = gcol * gpar[:, None]
            if prm.get("nebula", 0) > 0:
                # minimal coloured nebulae in the brown dust zone: pink (H-alpha), teal (OIII), violet in between.
                # Large-scale cloud (patch field) + medium tufts (n1); no fine noise -> cloud, not grain.
                nb = 0.85 * cp.clip(patch_v, -3, 3) + 0.35 * cp.clip(n1, -3, 3) + 0.15 * cp.clip(heat_n, -3, 3)
                mask = smooth(0.4, 0.55, tb) * smooth(prm["neb_thr"], prm["neb_thr"] + 0.8, nb) * prm["nebula"]
                sec = 0.5 + 0.5 * cp.sin(2.0 * phi + 1.6 * cp.clip(n1, -2, 2) - 1.2 * cp.clip(heat_n, -2, 2))
                pink, violet, teal = [cp.asarray(c) for c in prm["neb_colors"]]
                ncol = cp.where((sec < 0.5)[:, None], teal[None] + (violet - teal)[None] * (2 * sec)[:, None], violet[None] + (pink - violet)[None] * (2 * sec - 1)[:, None])
                # self-emitting (dim dust hid the colour change); the camera-facing half gets a boost
                facing = 1 + prm["neb_front"] * smooth(0.0, 0.6, cp.sin(phi))
                neb_em = ncol * (mask * prm["neb_bright"] * facing)[:, None]
            # one brightness profile: decreases continuously from the centre outwards (no "two stacked disks")
            prof = cp.clip((2.4 / r) ** prm["grad_bright_exp"], 0, 1.5) * smooth(2.05, 2.4, r) * prm["grad_bright"]
        else:
            c_in = blackbody(T_gas) * heat_bright[:, None]
        c_fl = c_in
    else:
        c_in = blackbody(cp.clip(prm["T0"] * x ** -0.5, 1500, 30000))
        c_fl = blackbody(cp.clip(5200 * (r_flow / r) ** 0.45, 1500, 9000))
    c_arm = blackbody(cp.full_like(r, 10500.0))
    c_hii = cp.asarray([1.0, 0.28, 0.45])
    hii = w_gx * arm * smooth(1.3, 2.2, n2) * vert
    arm_stars = w_gx * vert * (0.15 + arm) * (0.5 + 0.5 * cp.clip(n2, -1, 2)) * cp.exp(-r / prm["scale"])
    em = (c_in * (w_in * prof * (0.08 + 1.4 * cloud) * vert * prm["inner_bright"])[:, None]
          + c_fl * (w_fl * cloud_a * vert * light * prm["flow_bright"] * (0.15 + 1.6 * band) * (1 + front * 1.2 * arm_mask) * (1 - 0.8 * dust))[:, None]
          + (c_in if prm.get("gradient") is not None else blackbody(cp.full_like(r, prm.get("arm_front_T", 10500.0)))) * (w_fl * front * arm_mask * vert * (0.3 + light) * prm.get("arm_light_front", 0.5))[:, None]
          + blackbody(cp.clip(5200 * (r_flow / r) ** 0.45, 1500, 9000)) * (w_gx * cloud_g * vert * light * 0.6)[:, None]   # galaxy gas: same colour as preset B
          + c_arm * (arm_stars * prm["arm_star_glow"])[:, None]
          + c_hii[None] * (hii * prm["hii"])[:, None])
    if heat and prm.get("gradient") is not None and prm.get("nebula", 0) > 0:
        em = em + neb_em * vert[:, None]
    em = em * cp.sqrt(patch)[:, None]
    # measured Doppler: the side turning towards the camera (world phi ~ 0, +x) is bright and white, the other dim and warm; strong inside
    dop = prm["doppler"] * cp.cos(phi) * cp.clip((r_in / r) ** 0.5, 0, 1) * (1 - smooth(r_flow * 3, r_flow * 7, r))
    em = em * ((1 + dop) ** 3)[:, None] * cp.stack([1 - 0.25 * dop, cp.ones_like(dop), 1 + 0.35 * dop], 1)
    core = cp.exp(-((r - r_in * 1.15) / (r_in * 0.4)) ** 2) * smooth(r_in * 0.5, r_in * 0.95, r) * vert * (0.5 + 0.8 * cloud) * prm["core"]
    # inner-gap haze: soft transition between the disk's inner edge and the shadow (no hard gap)
    haze = haze_zone * vert * (0.4 + 0.6 * cloud) * prm["haze"]
    if heat:
        if prm.get("gradient") is not None:
            # haze uses the same gradient; no brightness peak of its own, it stays the inner end of the disk
            em = em + cp.asarray(prm["gradient"][0])[None] * (core * prm["grad_core"])[:, None] + c_in * (haze * prm["grad_haze"])[:, None]
        else:
            veins = smooth(0.35, 0.95, cloud) ** 1.5            # glowing veins inside, dark between
            haze_rgb = blackbody(cp.clip(prm["haze_T"] * T_mod, 1200, 30000))
            haze_l = haze_rgb.mean(1, keepdims=True)
            haze_rgb = cp.clip(haze_l + (haze_rgb - haze_l) * prm["haze_sat"], 0, None)       # saturation
            em = em + blackbody(cp.full_like(r, prm["core_T"])) * core[:, None] + haze_rgb * (haze * heat_bright * (0.12 + 1.9 * veins))[:, None]
    else:
        em = em + blackbody(cp.full_like(r, prm["haze_T"])) * (core + haze)[:, None]
    # outward glow around the hole: warm light that does not block rays
    glow = cp.exp(-cp.maximum(r - r_in * 0.9, 0) / (r_in * prm["glow_r"])) * smooth(2.1, r_in * 0.8, r) * cp.exp(-yn ** 2 * 0.35) * prm["glow"]
    em = em + blackbody(cp.full_like(r, prm["glow_T"])) * glow[:, None]
    # wave: haze swells and brightens at the crest, thins in the trough; iridescent tint shifts (magenta / cyan / gold)
    ph_c = r / (r_in * 1.6) + 0.25 * n1
    c_s = cp.cos(2 * np.pi * ph_c)
    tint = cp.stack([0.6 * c_s, -0.35 * cp.abs(c_s) + 0.1, -0.6 * c_s], 1)   # warm magenta <-> cold cyan (no green)
    if not heat:
        em = em * cp.clip(1 + prm["wave_bright"] * wave, 0.2, 4)[:, None] * (1 + prm["wave_tint"] * cp.abs(wave)[:, None] * tint)
        sig = sig * cp.clip(1 + prm["wave_density"] * wave, 0.3, 3)
    else:
        em = em + escaping_light(r, phi, vert, z, prm) + flare_clumps(r, phi, vert, z, prm)
    return sig, em * scale[:, None]


def escaping_light(r, phi, vert, z, prm):
    """Arcs of light breaking away from the hole's edge: they rise, redden and fade (gravitational
    redshift look), slow down and fall back. Irregular fragments, not full rings. Each arc rises an
    integer number of times per loop -> seamless."""
    r_in = prm["r_in"]
    total = cp.zeros((len(r), 3))
    rng = np.random.default_rng(31)
    for j in range(prm["arc_n"]):
        n = int(rng.integers(1, prm.get("arc_nmax", 2) + 1))
        o = rng.random()
        th0 = rng.random() * 2 * np.pi
        span = 0.5 + 0.9 * rng.random()
        r_max = r_in * (1.4 + 1.6 * rng.random())
        f = cp.mod(z * n + o, 1.0)
        h = cp.sin(np.pi * f)                       # 0 -> 1 -> 0: rise and fall back
        rr = 2.15 + (r_max - 2.15) * h
        dth = cp.angle(cp.exp(1j * (phi - th0 - 0.6 * np.sin(2 * np.pi * (z * n + o)))))   # arc centre sways slightly
        window = cp.exp(-(dth / span) ** 2)
        width = r_in * (0.12 + 0.25 * h)
        ring = cp.exp(-((r - rr) / width) ** 2) * window * vert
        pal_src = prm["gradient"][:3] if prm.get("gradient") is not None else None
        if pal_src is not None:
            p0, p1, p2 = [cp.asarray(c) for c in pal_src]
            hh = h[:, None] if hasattr(h, "shape") and h.ndim else h
            col = cp.where(hh < 0.5, p0[None] + (p1 - p0)[None] * (2 * hh), p1[None] + (p2 - p1)[None] * (2 * hh - 1))
        else:
            col = blackbody(cp.full_like(r, 1.0) * 14000 * (1 - 0.72 * h))   # reddens as it rises
        bright = prm["arc_bright"] * (1 - 0.6 * h) * cp.sin(0.5 * np.pi * cp.clip(f * 8, 0, 1))  # soft fade-in at break-off (quarter period)
        total = total + col * (ring * bright)[:, None]
    return total


def flare_clumps(r, phi, vert, z, prm):
    """Bright clumps orbiting in the inner disk; each makes exactly one revolution per loop and
    pulses at an integer frequency."""
    r_in = prm["r_in"]
    total = cp.zeros((len(r), 3))
    rng = np.random.default_rng(57)
    for j in range(prm["flare_n"]):
        rb = r_in * (1.15 + 1.3 * rng.random())
        th = 2 * np.pi * (z + rng.random())
        size = r_in * (0.25 + 0.3 * rng.random())
        dth = cp.angle(cp.exp(1j * (phi - th)))
        d2 = ((r - rb) / (size * 0.6)) ** 2 + (dth * rb / (size * 2.2)) ** 2     # stretched along the orbit
        pulse = 0.55 + 0.45 * np.sin(2 * np.pi * (prm.get("flare_pulse_f", 2) * z + rng.random()))
        pk = prm["gradient"] if prm.get("gradient") is not None else None
        color = cp.asarray(pk[1])[None] if pk is not None else blackbody(cp.full_like(r, 16000.0))
        total = total + color * (cp.exp(-d2) * vert * prm["flare_bright"] * pulse)[:, None]
    return total


# ---------------- stars (cell hash) ----------------
def _hash2(i, j, s):
    a = cp.sin(i * 12.9898 + j * 78.233 + s * 37.719) * 43758.5453
    return a - cp.floor(a)


def polar_stars(rq, phq, foot, prm, seed, z, q=12, jinv=None):
    """Stars in polar cells: ring j (width c) holds n_j = q*k cells, so the pattern repeats q times
    around each ring. A ring turns an integer multiple of 1/q revolution per loop -> at z=1 the
    pattern is identical (no loop blending needed, stars never double up and separate).
    Probability and colour come from the world position (the displayed position)."""
    c = prm["cell"]
    tp = np.tan(np.radians(prm.get("arm_pitch_front", prm["arm_pitch"])))
    m = prm["arms"]
    jq = cp.floor(rq / c)
    total = cp.zeros((len(rq), 3))
    for dj in (-1, 0, 1):
        jj = cp.maximum(jq + dj, 0)
        rj = (jj + 0.5) * c
        n = q * cp.maximum(1, cp.round(2 * np.pi * rj / (c * q)))
        turns = cp.round(prm["spin"] * (rj / prm["r_in"]) ** -1.5 * q) / q      # turns per loop (multiple of 1/q)
        rot = 2 * np.pi * turns * z
        ai = cp.floor(cp.mod(phq - rot, 2 * np.pi) / (2 * np.pi) * n)
        for di in (-1, 0, 1):
            i = ai + di
            ip = cp.mod(i, n / q)
            h1 = _hash2(jj, ip, seed)
            h2 = _hash2(jj, ip, seed + 1.7)
            sa = (i + h1) * 2 * np.pi / n + rot
            rs = (jj + h2) * c + 1e-6
            sx, sz = rs * cp.cos(sa), rs * cp.sin(sa)
            arm = (0.5 + 0.5 * cp.cos(m * sa + m * cp.log(rs) / tp)) ** 3
            odds = (0.12 + 0.88 * arm) * smooth(prm["r_flow"] * 0.8, prm["r_flow"] * 1.6, rs) * (1 - smooth(prm["r_out"] * 0.6, prm["r_out"], rs))
            odds = cp.maximum(odds, prm["inner_stars"] * smooth(prm["r_in"] * 0.55, prm["r_in"] * 0.9, rs) * (1 - smooth(prm["r_flow"] * 1.2, prm["r_flow"] * 2.5, rs)))
            odds = odds * (1 + prm.get("arm_star_front", 0.0) * arm * smooth(prm["r_flow"] * 0.6, prm["r_flow"] * 1.2, rs) * (1 - smooth(prm["r_flow"] * 4, prm["r_flow"] * 8, rs)))
            exists = _hash2(jj, ip, seed + 3.1) < odds * prm["star_prob"]
            p = _hash2(jj, ip, seed + 5.3)
            bright = (0.03 + 0.25 * (1 / (1 - 0.995 * p) ** 0.55 - 1)) * exists
            young = _hash2(jj, ip, seed + 7.9) < arm * 0.8
            T = cp.where(young, 8000 + 12000 * _hash2(jj, ip, seed + 9.1), 3500 + 3000 * _hash2(jj, ip, seed + 9.1))
            xq, zq = rq * cp.cos(phq), rq * cp.sin(phq)
            dx, dz = xq - sx, zq - sz
            if jinv is not None:
                # disk -> screen (pixel) distance: lensing magnification does not stretch the star, it stays round on screen
                a, b, c2, d, valid = jinv
                px_, py_ = a * dx + b * dz, c2 * dx + d * dz
                d2n = cp.where(valid, (px_ * px_ + py_ * py_) / prm["star_px"] ** 2, (dx * dx + dz * dz) / foot ** 2)
            else:
                d2n = (dx * dx + dz * dz) / foot ** 2
            halo = prm["star_halo"] * (1 - smooth(prm["r_flow"] * 1.0, prm["r_flow"] * 2.0, rs))
            total = total + blackbody(T) * (bright * (cp.exp(-d2n) + halo * cp.exp(-d2n / prm["halo_r"] ** 2)))[:, None]
    return total


def inverse_jacobian(lr, lp, W):
    """Per-pixel disk->screen map (2x2 inverse Jacobian) for one star sheet, from neighbouring pixels.
    Forward difference; backward where the neighbour is invalid. Invalid / extreme jumps -> valid=False
    (the isotropic footprint is used instead)."""
    X = (lr * cp.cos(lp)).reshape(-1, W)
    Z = (lr * cp.sin(lp)).reshape(-1, W)

    def diff(A, axis):
        fwd = cp.roll(A, -1, axis) - A
        bwd = A - cp.roll(A, 1, axis)
        if axis == 1:
            fwd[:, -1] = cp.nan
            bwd[:, 0] = cp.nan
        else:
            fwd[-1, :] = cp.nan
            bwd[0, :] = cp.nan
        return cp.where(cp.isnan(fwd), bwd, fwd)
    Xx, Xy, Zx, Zy = diff(X, 1), diff(X, 0), diff(Z, 1), diff(Z, 0)
    det = Xx * Zy - Xy * Zx
    size = cp.sqrt(cp.abs(det))
    valid = ~cp.isnan(det) & (cp.abs(det) > 1e-12)
    # jumps (caustics / sheet changes): ignore neighbour differences far above the local area scale
    valid = valid & (cp.maximum(cp.hypot(Xx, Zx), cp.hypot(Xy, Zy)) < 12 * size)
    det = cp.where(valid, det, 1.0)
    a, b, c2, d = Zy / det, -Xy / det, -Zx / det, Xx / det
    return [v.ravel() for v in (a, b, c2, d, valid)]


# ---------------- painter ----------------
def paint(rm, prm, z, fields, backdrop, W, H):
    """Paint one frame (linear HDR, shape (H, W, 3)) at loop time z in [0, 1)."""
    N = len(rm["count"])
    cam = camera_position(prm)
    camg = cp.asarray(cam)
    px_angle = np.radians(prm["fov"]) / H
    M = len(rm["r"])
    tau = cp.zeros(M)
    emis = cp.zeros((M, 3))
    tau_front = cp.zeros(M)
    CHUNK = 6_000_000
    for a in range(0, M, CHUNK):
        sl = slice(a, min(M, a + CHUNK))
        r = rm["r"][sl].astype(cp.float64)
        ph = rm["phi"][sl].astype(cp.float64)
        yn = rm["yn"][sl].astype(cp.float64)
        ds = rm["ds"][sl].astype(cp.float64)
        sig, em = density_emission(r, ph, yn, ds, prm, z, fields)
        pc = cp.stack([r * cp.cos(ph), yn * thickness(r, prm), r * cp.sin(ph)], 1)
        near = cp.clip((cp.linalg.norm(pc - camg, axis=1) - prm["near0"]) / prm["near1"], 0.03, 1.0)
        # secondary image: fades smoothly behind the hole (a binary threshold left a hard line)
        sk = 1 - (1 - prm["secondary"]) * smooth(0.0, -prm["secondary_fade"], rm["sec"][sl].astype(cp.float64))
        sig = sig * near * sk
        em = em * (near * sk)[:, None]
        tau[sl] = sig * ds
        emis[sl] = em * ds[:, None]
        tau_front[sl] = cp.where(rm["back"][sl] == 0, tau[sl], 0.0)
        del r, ph, yn, ds, sig, em, pc
    # front to back along each ray: T_before = exp(-(sum of tau of earlier samples of that ray))
    cs = cp.cumsum(tau)
    start = rm["start"]
    ray = rm["ray"]
    before = cs - tau - cp.where(start[ray] > 0, cs[cp.maximum(start[ray] - 1, 0)], 0.0)
    T_before = cp.exp(-before)
    color = cp.zeros((N, 3))
    for k in range(3):
        color[:, k] = cp.bincount(ray, weights=T_before * emis[:, k], minlength=N) * prm["gain"]
    tau_sum = cp.bincount(ray, weights=tau, minlength=N)
    T = cp.exp(-tau_sum)
    T_front = cp.exp(-cp.bincount(ray, weights=tau_front, minlength=N))

    def T_at(kidx):
        # transmittance before the ray's kidx-th sample (the full transmittance if kidx >= count)
        full = kidx >= rm["count"]
        j = cp.minimum(start[:-1] + kidx, max(M - 1, 0))
        return cp.where(full, T, T_before[j])
    # stars inside the cloud
    for s in range(rm["lay_k"].shape[1]):
        gi = cp.nonzero(~cp.isnan(rm["lay_r"][:, s]))[0]
        if len(gi) == 0:
            continue
        r = rm["lay_r"][gi, s].astype(cp.float64)
        ph = rm["lay_phi"][gi, s].astype(cp.float64)
        foot = cp.maximum(rm["lay_m"][gi, s] * px_angle * 0.7, 1e-4)
        jinv = [v[gi] for v in inverse_jacobian(rm["lay_r"][:, s].astype(cp.float64), rm["lay_phi"][:, s].astype(cp.float64), W)]
        y = polar_stars(r, ph, foot, prm, 10.0 + s // 2 * 3.3, z, jinv=jinv)
        w = cp.clip((rm["lay_m"][gi, s] - 10.0) / 40.0, 0.12, 1.0)
        w = w * cp.where(rm["lay_s"][gi, s] >= 1, prm["secondary"], 1.0)
        Tk = T_at(rm["lay_k"][:, s])[gi]
        color[gi] += y * (Tk * w * prm["star_cloud"])[:, None]   # stars are pixel-sized on screen -> no resolution factor needed
    # atmosphere veil: thin hazy light just outside the shadow (skipped when veil = 0)
    if prm["veil"] <= 0:
        color += T[:, None] * sky_stars_select(rm, prm, z, backdrop, H) * prm["stars"]
        return color.reshape(-1, W, 3)
    b = rm["impact"].astype(cp.float64)
    yy, xx = cp.mgrid[0:H, 0:W]
    ang = cp.arctan2(-(yy.ravel() - prm["center_y"] * H), xx.ravel() - W / 2)
    wob = 0.0
    for n, ph0, a in ((3, 0.4, 0.5), (7, 1.9, 0.35), (13, 4.1, 0.25), (29, 2.2, 0.15)):
        wob = wob + a * cp.sin(n * ang + ph0 + 2 * np.pi * z * (1 if n % 2 else -1))
    wob = 0.55 + 0.45 * wob / 1.25
    db = b - B_CRIT * 1.002
    veil = cp.where(db > 0, cp.exp(-db / prm["veil_width"]) * (0.35 + 0.65 * wob), 0.0)
    veil = veil + cp.exp(-(db / (prm["veil_width"] * 0.25)) ** 2) * 0.6     # sharp line at the photon ring
    if prm["veil"] > 0:
        color += (T_front ** 8 * veil * prm["veil"])[:, None] * blackbody(cp.full(N, prm["veil_T"]))
    # background stars (precomputed, static)
    color += T[:, None] * sky_stars_select(rm, prm, z, backdrop, H) * prm["stars"]
    return color.reshape(-1, W, 3)


def sky_stars_select(rm, prm, z, backdrop, H):
    if not prm.get("sky_warp", False):
        return backdrop
    out = cp.zeros_like(backdrop)
    esc = cp.nonzero(rm["outcome"] == 2)[0]
    if len(esc):
        out[esc] = sky_stars(rm["escape"][esc].astype(cp.float64), np.radians(prm["fov"]) / H, prm, z)
    return out


def sky_stars(d, px_angle, prm, z):
    """Sky stars (GPU, 3D cell hash on the sphere). The direction is perturbed first: a small heat shimmer
    and a slow swirl around the hole (stars close to the hole feel its pull). All z frequencies are integers."""
    cam = camera_position(prm)
    h = -cam / np.linalg.norm(cam)
    hg = cp.asarray(h)
    A = prm["sky_shimmer"] * px_angle
    w = cp.stack([cp.sin(2 * np.pi * z + 41 * d[:, 1] + 7 * d[:, 0] + 1.3 * cp.sin(2 * np.pi * z + 23 * d[:, 2])),
                  cp.sin(-2 * np.pi * z + 37 * d[:, 0] + 5 * d[:, 2]),
                  cp.sin(2 * np.pi * z + 31 * d[:, 2] + 11 * d[:, 1] + 0.7)], 1)
    d = d + A * w
    # swirl: rotation about the direction of the hole, angle falls off with angular distance; slow back-and-forth
    cos_t = cp.clip(d @ hg, -1, 1)
    theta = cp.arccos(cos_t)
    alpha = prm["sky_swirl"] * cp.sin(2 * np.pi * z) * cp.exp(-theta / prm["sky_swirl_r"])
    k = hg[None, :]
    d = d * cp.cos(alpha)[:, None] + cp.cross(cp.broadcast_to(k, d.shape), d) * cp.sin(alpha)[:, None] + k * (d @ hg)[:, None] * (1 - cp.cos(alpha))[:, None]
    d = d / cp.linalg.norm(d, axis=1, keepdims=True)
    K = 1 / prm["sky_cell"]
    c0 = cp.floor(d * K)
    total = cp.zeros((len(d), 3))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = c0 + cp.asarray([dx, dy, dz], dtype=cp.float64)
                hx = _hash2(c[:, 0], c[:, 1] + c[:, 2] * 0.37, 3.3)
                exists = hx < prm["sky_prob"]
                j = cp.stack([_hash2(c[:, 0], c[:, 1], c[:, 2] + 1.1), _hash2(c[:, 1], c[:, 2], c[:, 0] + 2.2), _hash2(c[:, 2], c[:, 0], c[:, 1] + 3.3)], 1)
                s = (c + j) / K
                s = s / cp.linalg.norm(s, axis=1, keepdims=True)
                d2 = cp.sum((d - s) ** 2, axis=1)
                p = _hash2(c[:, 0] + c[:, 2], c[:, 1], 5.5)
                bright = (0.03 + 0.15 * (1 / (1 - 0.995 * p) ** 0.5 - 1)) * exists
                T = 3500 + 8000 * _hash2(c[:, 1], c[:, 0] + c[:, 2], 7.7)
                total = total + blackbody(T) * (bright * cp.exp(-d2 / (px_angle * prm["star_px"]) ** 2))[:, None]
    return total


def backdrop_stars(rm, prm, H):
    """Static star field seen through the escape directions (kd-tree on the CPU), used by
    presets without `sky_warp`."""
    from scipy.spatial import cKDTree
    esc = cp.asnumpy(rm["outcome"] == 2)
    out = np.zeros((len(esc), 3))
    px_angle = np.radians(prm["fov"]) / H
    dirs = cp.asnumpy(rm["escape"])[esc].astype(np.float64)
    count, seed = 60000, 7
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(count, 3))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    bright = rng.pareto(2.2, count) * 0.15 + 0.03
    temp = rng.uniform(3500, 11000, count)
    dist, idx = cKDTree(s).query(dirs, k=1)
    w = np.exp(-(dist / (px_angle * 0.6)) ** 2)
    out[esc] = cp.asnumpy(blackbody(cp.asarray(temp[idx]))) * (bright[idx] * w)[:, None]
    return cp.asarray(out)
