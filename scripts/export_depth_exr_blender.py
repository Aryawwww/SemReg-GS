"""Export Blender EXR depth passes to a compressed NumPy archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--view-names", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = arguments()
    arrays = {}
    summary = {}
    for name in args.view_names:
        path = args.views.resolve() / name / "depth.exr"
        if not path.is_file():
            raise FileNotFoundError(path)
        image = oiio.ImageInput.open(str(path))
        if image is None:
            raise RuntimeError(f"OpenImageIO could not open {path}: {oiio.geterror()}")
        spec = image.spec()
        pixels = np.asarray(image.read_image(oiio.FLOAT), dtype=np.float32)
        image.close()
        height, width = int(spec.height), int(spec.width)
        if pixels.size != height * width * int(spec.nchannels):
            raise RuntimeError(f"Unexpected EXR buffer shape for {path}: {pixels.shape}")
        pixels = pixels.reshape(height, width, int(spec.nchannels))
        depth = pixels[..., 0].copy()
        finite = np.isfinite(depth) & (depth > 0)
        arrays[name] = depth
        summary[name] = {
            "shape": [height, width],
            "finite_positive_pixels": int(finite.sum()),
            "minimum": float(depth[finite].min()) if finite.any() else None,
            "maximum": float(depth[finite].max()) if finite.any() else None,
        }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(json.dumps({"output": str(output), "views": summary}))


if __name__ == "__main__":
    main()
