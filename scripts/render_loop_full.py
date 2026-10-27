"""Render the full-resolution loop (PNG frames + H.264 MP4).

    python scripts/render_loop_full.py --preset C --width 1440 --seconds 8 --fps 24 --strips 12

The image is split into horizontal bands; each band's ray map is built once, kept in host
memory, and uploaded to the GPU for every frame. More strips = less VRAM, a bit slower.
Existing frame files are skipped, so an interrupted run can be resumed.
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
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--strips", type=int, default=12)
    ap.add_argument("--out", default="output", help="output directory")
    ap.add_argument("--cache", default="output/cache")
    args = ap.parse_args()

    W = args.width
    H = image_height(W)
    frames_dir = os.path.join(args.out, f"frames_{args.preset}_{W}")
    os.makedirs(frames_dir, exist_ok=True)
    prm = make_prm(args.preset, loop=True)
    fields = build_fields(prm, args.cache)
    t0 = time.time()
    bounds, bands = build_bands(prm, W, H, args.strips)
    print(f"ray maps ready {time.time() - t0:.0f} s", flush=True)
    NF = int(round(args.seconds * args.fps))
    for f in range(NF):
        p = os.path.join(frames_dir, f"k{f:04d}.png")
        if os.path.exists(p):
            continue
        img = paint_frame(prm, fields, bounds, bands, f / NF, W, H)
        Image.fromarray(finish_frame(img, prm, f)).save(p)
        if f % 12 == 0:
            print(f"{f}/{NF}  {(time.time() - t0) / 60:.1f} min", flush=True)
    mp4 = os.path.join(args.out, f"gargantua_{args.preset}_{W}x{H}_{int(args.seconds)}s.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", os.path.join(frames_dir, "k%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "14", "-preset", "slow", "-profile:v", "high",
                    "-level", "5.1", "-movflags", "+faststart", mp4], check=True)
    print("done", mp4, f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
