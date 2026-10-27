"""Gargantua: a Schwarzschild black hole ray tracer that renders looping wallpapers."""
from .presets import VARS
from .raytrace import build_ray_map
from .fields import build_fields
from .shading import paint, backdrop_stars
from .post import tonemap

__all__ = ["VARS", "build_ray_map", "build_fields", "paint", "backdrop_stars", "tonemap"]
