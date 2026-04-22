import os
import glob
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import open_clip

from model import FeatureAutoEncoder


# -----------------------------
# Utils
# -----------------------------
IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]


def ensure_segment_anything_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sam_root = repo_root / "third_party" / "segment-anything"
    sam_root_str = str(sam_root)
    if sam_root.exists() and sam_root_str not in sys.path:
        sys.path.insert(0, sam_root_str)


def list_images(image_dir: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found under {image_dir}")
    return files


def stem(path: str) -> str:
    return Path(path).stem


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_np(feat: np.ndarray, eps: float = 1e-6):
    norm = np.linalg.norm(feat, axis=-1, keepdims=True)
    return feat / np.clip(norm, eps, None)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# -----------------------------
# Region proposals:
#   1) SAM if installed
#   2) fallback to multi-scale grid regions
# -----------------------------
def build_sam_generator(sam_ckpt: str, sam_type: str = "vit_h"):
    ensure_segment_anything_on_path()
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    sam.cuda().eval()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=24,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    return mask_generator


def sam_masks_for_image(mask_generator, image_pil: Image.Image) -> List[Dict]:
    image_np = np.array(image_pil.convert("RGB"))
    masks = mask_generator.generate(image_np)
    return masks


def fallback_multiscale_regions(image_pil: Image.Image) -> List[Dict]:
    """
    不依赖 SAM 的兜底方案。
    输出格式尽量模拟 SAM：每个 region 带 bbox / segmentation / area。
    """
    img = np.array(image_pil.convert("RGB"))
    h, w = img.shape[:2]

    masks = []
    scales = [2, 3, 4]
    for s in scales:
        cell_h = h // s
        cell_w = w // s
        for iy in range(s):
            for ix in range(s):
                y0 = iy * cell_h
                x0 = ix * cell_w
                y1 = h if iy == s - 1 else (iy + 1) * cell_h
                x1 = w if ix == s - 1 else (ix + 1) * cell_w

                seg = np.zeros((h, w), dtype=np.uint8)
                seg[y0:y1, x0:x1] = 1
                area = int(seg.sum())
                masks.append({
                    "segmentation": seg.astype(bool),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": area,
                    "predicted_iou": 0.5,
                    "stability_score": 0.5,
                })

    # 加一个全图区域
    seg = np.ones((h, w), dtype=bool)
    masks.append({
        "segmentation": seg,
        "bbox": [0, 0, w, h],
        "area": int(h * w),
        "predicted_iou": 1.0,
        "stability_score": 1.0,
    })
    return masks


def filter_masks(masks: List[Dict], image_hw: Tuple[int, int], max_regions: int = 32) -> List[Dict]:
    h, w = image_hw
    img_area = h * w
    kept = []
    for m in masks:
        area = m["area"]
        ratio = area / max(img_area, 1)
        if ratio < 0.002:
            continue
        if ratio > 0.90:
            continue
        kept.append(m)

    # 按稳定度 + 适中面积排序
    def score(m):
        ratio = m["area"] / img_area
        area_pref = 1.0 - abs(ratio - 0.12)
        return float(m.get("predicted_iou", 0.5)) + float(m.get("stability_score", 0.5)) + area_pref

    kept = sorted(kept, key=score, reverse=True)[:max_regions]
    return kept


# -----------------------------
# CLIP feature extractor
# -----------------------------
class CLIPRegionEncoder:
    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k", device: str = "cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_pil_batch(self, images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(im) for im in images], dim=0).to(self.device)
        with torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
            feat = self.model.encode_image(batch)
            feat = F.normalize(feat.float(), dim=-1)
        return feat


def crop_region_with_mask(image: Image.Image, seg: np.ndarray, bbox: List[int]) -> Image.Image:
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h

    img_np = np.array(image.convert("RGB"))
    crop = img_np[y:y2, x:x2].copy()
    crop_mask = seg[y:y2, x:x2]

    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return image.copy()

    # 背景置灰，尽量保留区域语义
    bg = np.full_like(crop, 127)
    crop = np.where(crop_mask[..., None], crop, bg)

    return Image.fromarray(crop)


@torch.no_grad()
def build_dense_feature_map(
    image: Image.Image,
    region_encoder: CLIPRegionEncoder,
    masks: List[Dict],
    out_hw: Tuple[int, int] = (64, 64),
    add_global_feature: bool = True,
) -> np.ndarray:
    """
    返回 [Hf, Wf, 512] float16
    """
    image = image.convert("RGB")
    img_np = np.array(image)
    H, W = img_np.shape[:2]
    Hf, Wf = out_hw

    # 先编码全图
    global_feat = region_encoder.encode_pil_batch([image])[0].detach().cpu().numpy()

    # 区域 crop
    crops = []
    resized_masks = []
    weights = []

    for m in masks:
        seg = m["segmentation"].astype(bool)
        bbox = m["bbox"]
        crop = crop_region_with_mask(image, seg, bbox)
        crops.append(crop)

        seg_small = Image.fromarray((seg.astype(np.uint8) * 255)).resize((Wf, Hf), Image.NEAREST)
        seg_small = (np.array(seg_small) > 127).astype(np.float32)
        resized_masks.append(seg_small)

        area_ratio = m["area"] / max(H * W, 1)
        # 偏向中尺度区域
        weight = 1.0 - abs(area_ratio - 0.12)
        weight = max(weight, 0.1)
        weights.append(weight)

    if len(crops) > 0:
        region_feats = region_encoder.encode_pil_batch(crops).detach().cpu().numpy()  # [R, 512]
    else:
        region_feats = np.zeros((0, global_feat.shape[0]), dtype=np.float32)

    feat_map = np.zeros((Hf, Wf, global_feat.shape[0]), dtype=np.float32)
    cnt_map = np.zeros((Hf, Wf, 1), dtype=np.float32)

    for seg_small, feat, w in zip(resized_masks, region_feats, weights):
        seg_small = seg_small[..., None]
        feat_map += seg_small * feat[None, None, :] * w
        cnt_map += seg_small * w

    if add_global_feature:
        feat_map += global_feat[None, None, :]
        cnt_map += 1.0

    feat_map = feat_map / np.clip(cnt_map, 1e-6, None)
    feat_map = normalize_np(feat_map).astype(np.float16)
    return feat_map


# -----------------------------
# Compression
# -----------------------------
def load_autoencoder(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = FeatureAutoEncoder(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
        encoder_hidden=cfg["encoder_hidden"],
        decoder_hidden=cfg["decoder_hidden"],
        normalize_input=cfg.get("normalize_input", True),
        normalize_latent=cfg.get("normalize_latent", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def compress_feature_map(model: FeatureAutoEncoder, feat_map: np.ndarray, device: str, chunk: int = 65536):
    shape = feat_map.shape
    x = feat_map.reshape(-1, shape[-1]).astype(np.float32)
    outs = []

    for i in range(0, len(x), chunk):
        xx = torch.from_numpy(x[i:i+chunk]).to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            z = model.encode(xx).float()
        outs.append(z.cpu().numpy())

    z = np.concatenate(outs, axis=0)
    z = z.reshape(shape[0], shape[1], -1).astype(np.float16)
    return z


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["extract", "compress"], required=True)
    parser.add_argument("--scene_root", type=str, required=True)

    parser.add_argument("--image_dirname", type=str, default="images")
    parser.add_argument("--raw_dirname", type=str, default="language_features_raw")
    parser.add_argument("--compressed_dirname", type=str, default="language_features_dim8")

    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--feature_hw", nargs=2, type=int, default=[64, 64])

    parser.add_argument("--use_sam", action="store_true")
    parser.add_argument("--sam_ckpt", type=str, default="")
    parser.add_argument("--sam_type", type=str, default="vit_h")
    parser.add_argument("--max_regions", type=int, default=32)

    parser.add_argument("--ae_ckpt", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene_root = Path(args.scene_root)
    image_dir = scene_root / args.image_dirname
    raw_dir = scene_root / args.raw_dirname
    compressed_dir = scene_root / args.compressed_dirname

    ensure_dir(raw_dir)
    ensure_dir(compressed_dir)

    image_files = list_images(str(image_dir))

    if args.mode == "extract":
        region_encoder = CLIPRegionEncoder(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=device
        )

        sam_generator = None
        if args.use_sam:
            if args.sam_ckpt == "":
                raise ValueError("--use_sam was set but --sam_ckpt is empty.")
            try:
                sam_generator = build_sam_generator(args.sam_ckpt, args.sam_type)
                print("[Info] SAM loaded successfully.")
            except Exception as e:
                print(f"[Warn] SAM unavailable, fallback to multiscale regions. Reason: {e}")
                sam_generator = None

        meta = {
            "clip_model": args.clip_model,
            "clip_pretrained": args.clip_pretrained,
            "feature_hw": args.feature_hw,
            "sam_enabled": sam_generator is not None,
            "max_regions": args.max_regions,
        }

        for img_fp in tqdm(image_files, desc="Extracting raw language features"):
            name = stem(img_fp)
            out_fp = raw_dir / f"{name}.npy"
            if out_fp.exists():
                continue

            image = Image.open(img_fp).convert("RGB")
            H, W = np.array(image).shape[:2]

            if sam_generator is not None:
                masks = sam_masks_for_image(sam_generator, image)
            else:
                masks = fallback_multiscale_regions(image)

            masks = filter_masks(masks, (H, W), max_regions=args.max_regions)

            feat_map = build_dense_feature_map(
                image=image,
                region_encoder=region_encoder,
                masks=masks,
                out_hw=tuple(args.feature_hw),
                add_global_feature=True,
            )
            np.save(out_fp, feat_map)

        save_json(meta, str(scene_root / "language_feature_meta.json"))
        print(f"[Done] Raw features saved to {raw_dir}")

    elif args.mode == "compress":
        if args.ae_ckpt == "":
            raise ValueError("--ae_ckpt is required in compress mode")

        model, cfg = load_autoencoder(args.ae_ckpt, device=device)
        latent_dim = cfg["latent_dim"]

        # 自动修正输出目录名
        compressed_dir = scene_root / f"language_features_dim{latent_dim}"
        ensure_dir(compressed_dir)

        raw_files = sorted(glob.glob(str(raw_dir / "*.npy")))
        if len(raw_files) == 0:
            raise FileNotFoundError(f"No raw feature maps found in {raw_dir}")

        for raw_fp in tqdm(raw_files, desc="Compressing raw language features"):
            name = stem(raw_fp)
            out_fp = compressed_dir / f"{name}.npy"
            if out_fp.exists():
                continue

            feat_map = np.load(raw_fp)
            z = compress_feature_map(model, feat_map, device=device)
            np.save(out_fp, z)

        save_json(
            {
                "ae_ckpt": args.ae_ckpt,
                "latent_dim": latent_dim,
                "raw_dir": str(raw_dir),
                "compressed_dir": str(compressed_dir),
            },
            str(scene_root / "language_feature_compression_meta.json")
        )
        print(f"[Done] Compressed features saved to {compressed_dir}")


if __name__ == "__main__":
    main()
