import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def resolve_dirs(input_path: Path, output_dir: Optional[str]):
    if input_path.is_file():
        raise ValueError("Input path must be a directory.")

    semantic_dir = input_path / "semantic" if (input_path / "semantic").is_dir() else input_path
    if not semantic_dir.is_dir():
        raise FileNotFoundError(f"Could not find semantic directory under {input_path}")

    base_dir = semantic_dir.parent
    renders_dir = base_dir / "renders"
    gt_dir = base_dir / "gt"

    vis_dir = Path(output_dir) if output_dir else base_dir / "semantic_vis"
    compare_dir = base_dir / "semantic_compare"
    vis_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    return semantic_dir, renders_dir, gt_dir, vis_dir, compare_dir


def list_semantic_files(semantic_dir: Path):
    files = sorted(semantic_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No semantic .npy files found in {semantic_dir}")
    return files


def load_feature_map(path: Path):
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D feature map in {path}, got shape {arr.shape}")

    # Support both HWC and CHW layouts.
    if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
        arr = np.transpose(arr, (1, 2, 0))

    return arr.astype(np.float32, copy=False)


def sample_features(files, max_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    per_file = max(256, max_samples // max(len(files), 1))
    chunks = []

    for path in files:
        feat_map = load_feature_map(path)
        feat = feat_map.reshape(-1, feat_map.shape[-1])
        feat = feat[np.isfinite(feat).all(axis=1)]
        if feat.size == 0:
            continue
        take = min(per_file, feat.shape[0])
        idx = rng.choice(feat.shape[0], size=take, replace=False)
        chunks.append(feat[idx])

    if not chunks:
        raise RuntimeError("No valid semantic features were found for PCA fitting.")

    sampled = np.concatenate(chunks, axis=0)
    if sampled.shape[0] > max_samples:
        idx = rng.choice(sampled.shape[0], size=max_samples, replace=False)
        sampled = sampled[idx]
    return sampled


def fit_pca(samples: np.ndarray):
    mean = samples.mean(axis=0, keepdims=True)
    centered = samples - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:3].T
    projected = centered @ basis
    lower = np.percentile(projected, 1, axis=0)
    upper = np.percentile(projected, 99, axis=0)
    span = np.clip(upper - lower, 1e-6, None)
    return mean, basis, lower, span


def project_to_rgb(feature_map: np.ndarray, mean, basis, lower, span):
    h, w, c = feature_map.shape
    flat = feature_map.reshape(-1, c)
    proj = (flat - mean) @ basis
    proj = (proj - lower) / span
    proj = np.clip(proj, 0.0, 1.0)
    rgb = (proj.reshape(h, w, 3) * 255.0).astype(np.uint8)
    return rgb


def load_image_if_exists(path: Path):
    if not path.is_file():
        return None
    return Image.open(path).convert("RGB")


def hstack_images(images):
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize semantic latent maps with PCA pseudo-coloring.")
    parser.add_argument("input", type=str, help="Path to an `ours_xxx` directory or a `semantic` directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save semantic visualization PNGs.")
    parser.add_argument("--max_samples", type=int, default=200000, help="Maximum number of sampled pixels for PCA.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pixel sampling.")
    parser.add_argument("--no_compare", action="store_true", help="Do not save side-by-side comparisons with render/gt.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    semantic_dir, renders_dir, gt_dir, vis_dir, compare_dir = resolve_dirs(input_path, args.output_dir)
    semantic_files = list_semantic_files(semantic_dir)

    samples = sample_features(semantic_files, args.max_samples, args.seed)
    mean, basis, lower, span = fit_pca(samples)

    for semantic_path in semantic_files:
        feature_map = load_feature_map(semantic_path)
        rgb = project_to_rgb(feature_map, mean, basis, lower, span)
        rgb_image = Image.fromarray(rgb, mode="RGB")

        out_png = vis_dir / f"{semantic_path.stem}.png"
        rgb_image.save(out_png)

        if args.no_compare:
            continue

        render_image = load_image_if_exists(renders_dir / f"{semantic_path.stem}.png")
        gt_image = load_image_if_exists(gt_dir / f"{semantic_path.stem}.png")
        panels = [img for img in [render_image, rgb_image, gt_image] if img is not None]
        if len(panels) >= 2:
            hstack_images(panels).save(compare_dir / f"{semantic_path.stem}.png")

    print(f"Saved semantic visualizations to {vis_dir}")
    if not args.no_compare:
        print(f"Saved semantic comparison images to {compare_dir}")


if __name__ == "__main__":
    main()
