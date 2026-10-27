"""Render a low-resolution preview loop (PNG frames + MP4 via ffmpeg).

    python scripts/render_loop_preview.py --preset C --width 360 --seconds 8 --fps 24
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image
from gargantua.fields import build_fields
from gargantua.render import make_prm, image_height, build_bands, paint_frame, finish_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="C", choices=["A", "B", "C"])
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default=None, help="output directory (default output/preview_<preset>_<width>)")
    ap.add_argument("--cache", default="output/cache")
    args = ap.parse_args()

    W = args.width
    H = image_height(W)
    out = args.out or os.path.join("output", f"preview_{args.preset}_{W}")
    os.makedirs(out, exist_ok=True)
    prm = make_prm(args.preset, loop=True)
    fields = build_fields(prm, args.cache)
    bounds, bands = build_bands(prm, W, H, 1)
    NF = int(round(args.seconds * args.fps))
    t0 = time.time()
    for f in range(NF):
        p = os.path.join(out, f"k{f:04d}.png")
        if os.path.exists(p):
            continue
        img = paint_frame(prm, fields, bounds, bands, f / NF, W, H)
        Image.fromarray(finish_frame(img, prm, f)).save(p)
        if f % 24 == 0:
            print(f"{f}/{NF}  {time.time() - t0:.0f} s", flush=True)
    mp4 = out.rstrip("/\\") + ".mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", os.path.join(out, "k%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", mp4], check=True)
    print("done", mp4, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
