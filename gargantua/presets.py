"""Parameter presets. "C" is the final look; "A" and "B" are earlier steps it builds on.

Parameter glossary (grouped)
  geometry   r_in / r_out / r_flow: inner disk edge, outer edge, start of the outer flow region;
             thick_slope, thick_base: half-thickness = slope * r + base; y_cap: sampled extent in thicknesses
  camera     fov (vertical, deg), distance, tilt (elevation over the disk, deg), center_y, roll (deg)
  motion     spin / inflow: rotation and inward drift per loop; wave_n: outward waves per loop
  look       arms, arm_pitch, flow_pitch: spiral arm count and pitch; threshold, sigma, gain: cloud cover, opacity, exposure
  heat       heat=True enables the temperature model (C); gradient: colour stops from centre to rim
  film       film=True enables the camera/film chain in post.py (C)
"""
import numpy as np


def _gradient(hexes):
    """'#RRGGBB' strings -> list of linear RGB triples (gamma 2.2)."""
    return [list(np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]) ** 2.2) for h in hexes]


VARS = {
    # A: baseline look
    "A": dict(r_in=6, r_out=420, r_flow=18, arms=2, arm_pitch=28, fov=40, distance=60, tilt=7, center_y=0.58, roll=-7.0, doppler=0.32,
              thick_slope=0.04, thick_base=0.35, y_cap=2.6, step_k=0.06, inner_step=0.05, max_samples=256, ny=48,
              vert_exp=0.85, flow_pitch=18, dust=3.0, haze=13.75, haze_density=10.0, wave_n=1, wave_density=0.8,
              wave_bright=1.4, wave_tint=0.4,
              threshold=0.75, sigma=7.0, secondary=0.1, secondary_fade=7.0, spin=0.6, inflow=-0.25, T0=6500,
              inner_bright=4.5, flow_bright=1.6, arm_star_glow=0.9, hii=0.6, scale=110,
              gain=0.6, exposure=1.1, light_exp=1.8, kappa=9.0, near0=20, near1=60,
              erosion=0.6, erosion_y=4.0, patch_min=0.15, core=1.6,
              cell=0.9, star_prob=0.35, star_cloud=2.6, inner_stars=1.3, star_halo=0.025, halo_r=7.0, stars=0.7,
              veil=0.0, veil_width=0.13, veil_T=7500,
              haze_T=6300.0, glow=1.0, star_px=0.85, glow_r=1.3, glow_T=5200.0),
}

# B: same as A, with the front band shaped like a galaxy arm
VARS["B"] = dict(VARS["A"], arm_front=1.0, flow_pitch=32, arm_pitch_front=32, arm_star_front=6.0, dust=0.4,
                 arm_thr0=0.45, arm_thr1=0.85, arm_light_front=0.55, arm_front_T=6800.0)

# C: B plus the heat model, one colour gradient, and film/camera imperfections
VARS["C"] = dict(VARS["B"], heat=True, T0=12000.0, T_exp=1.8, T_flow=2600.0, T_wave=0.45, T_bright_exp=2.0,
                 counter_spin=0.18, counter_inflow=0.15, arm_star_glow=1.7, core_T=20000.0, haze_T=3300.0,
                 haze=6.0, glow=0.6, glow_T=3600.0, shimmer=0.06, arc_n=7, arc_bright=3.0, flare_n=5, flare_bright=4.0,
                 hii=1.5, arm_front_T=9000.0, arm_light_front=0.25, dust=1.2, gain=0.32, star_cloud=1.0,
                 spin=0.3, inflow=-0.12, shimmer_f=(1, -1), arc_nmax=1, flare_pulse_f=1, haze_sat=1.8,
                 gradient=_gradient(("#FFF1D6", "#FFD49A", "#F0A050", "#A8552A", "#3E2015")), grad_R=80.0, grad_warp=0.07,
                 grad_band=5, grad_bright=3.0, grad_bright_exp=1.2, grad_haze=9.0, grad_core=1.5,
                 nebula=0.55, neb_thr=0.85, neb_bright=2.156, neb_front=2.5, neb_colors=_gradient(("#E0457B", "#7A4FC9", "#3FB5B0")),
                 T_inner_min=3200.0, core=1.92,
                 sky_warp=True, sky_shimmer=1.2, sky_swirl=0.003, sky_swirl_r=0.12, sky_cell=0.0145, sky_prob=0.5,
                 film=True, bloom=0.5, bloom_thr=1.4, halation=0.35, scatter=0.01, ca=0.0008, vignette=0.35, grain=0.035)
