"""Render a single frame.

    python scripts/render_still.py --preset C --width 540 --time 0.3 --out output/still.png

--time is the loop phase in [0, 1). --strips splits the image into horizontal bands to
limit GPU memory (use 8-12 for 1440 px wide).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image
from gargantua.fields import build_fields
from gargantua.render import make_prm, image_height, build_bands, paint_frame, finish_frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="C", choices=["A", "B", "C"])
    ap.add_argument("--width", type=int, default=540, help="image width in pixels (height = width * 3120/1440)")
    ap.add_argument("--time", type=float, default=0.0, help="loop phase in [0, 1)")
    ap.add_argument("--frame", type=int, default=0, help="frame number; seeds the film grain")
    ap.add_argument("--strips", type=int, default=1, help="number of horizontal bands (GPU memory)")
    ap.add_argument("--no-loop", action="store_true", help="disable the seamless-loop phase blend")
    ap.add_argument("--out", default="output/still.png")
    ap.add_argument("--cache", default="output/cache", help="cache directory for the cloud fields")
    args = ap.parse_args()

    W = args.width
    H = image_height(W)
    prm = make_prm(args.preset, loop=not args.no_loop)
    fields = build_fields(prm, args.cache)
    bounds, bands = build_bands(prm, W, H, args.strips)
    t0 = time.time()
    img = paint_frame(prm, fields, bounds, bands, args.time, W, H)
    out = finish_frame(img, prm, args.frame)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Image.fromarray(out).save(args.out)
    print(f"paint {time.time() - t0:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
