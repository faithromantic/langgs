import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip


IMG_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
LEVEL_NAMES = ("default", "s", "m", "l")


def list_images(image_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(Path(p) for p in glob.glob(str(image_dir / f"*{ext}")))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return files


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class CLIPRegionEncoder:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        device: str,
        batch_size: int,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device,
        )
        self.model.eval()

    @torch.no_grad()
    def encode(self, images: List[Image.Image]) -> np.ndarray:
        outputs = []
        for start in range(0, len(images), self.batch_size):
            batch = torch.stack(
                [self.preprocess(image.convert("RGB")) for image in images[start:start + self.batch_size]],
                dim=0,
            ).to(self.device)
            with torch.autocast(device_type="cuda", enabled=(self.device == "cuda")):
                features = self.model.encode_image(batch).float()
                features = F.normalize(features, dim=-1)
            outputs.append(features.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float16)


def build_sam_generator(sam_ckpt: str, sam_type: str):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    sam.to("cuda").eval()
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )


def fallback_grid_masks(height: int, width: int) -> Dict[str, List[Dict]]:
    level_to_grid = {
        "default": 4,
        "s": 8,
        "m": 4,
        "l": 2,
    }
    levels: Dict[str, List[Dict]] = {}
    for level, grid in level_to_grid.items():
        masks = []
        cell_h = max(height // grid, 1)
        cell_w = max(width // grid, 1)
        for gy in range(grid):
            for gx in range(grid):
                y0 = gy * cell_h
                x0 = gx * cell_w
                y1 = height if gy == grid - 1 else min((gy + 1) * cell_h, height)
                x1 = width if gx == grid - 1 else min((gx + 1) * cell_w, width)
                seg = np.zeros((height, width), dtype=bool)
                seg[y0:y1, x0:x1] = True
                masks.append(
                    {
                        "segmentation": seg,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": int(seg.sum()),
                    }
                )
        levels[level] = masks
    return levels


def split_sam_masks_by_level(masks: List[Dict], height: int, width: int, max_masks: int) -> Dict[str, List[Dict]]:
    img_area = max(height * width, 1)
    masks = sorted(
        masks,
        key=lambda m: float(m.get("predicted_iou", 0.0)) * float(m.get("stability_score", 0.0)),
        reverse=True,
    )

    levels = {level: [] for level in LEVEL_NAMES}
    for mask in masks:
        area_ratio = float(mask["area"]) / img_area
        if area_ratio < 0.001 or area_ratio > 0.95:
            continue
        if area_ratio < 0.03:
            levels["s"].append(mask)
        elif area_ratio < 0.18:
            levels["m"].append(mask)
        else:
            levels["l"].append(mask)
        levels["default"].append(mask)

    fallback = fallback_grid_masks(height, width)
    for level in LEVEL_NAMES:
        levels[level] = levels[level][:max_masks]
        if not levels[level]:
            levels[level] = fallback[level]
    return levels


def crop_with_mask(image: Image.Image, mask: Dict) -> Image.Image:
    x, y, w, h = [int(v) for v in mask["bbox"]]
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + max(w, 1), image.width)
    y2 = min(y + max(h, 1), image.height)

    image_np = np.asarray(image.convert("RGB"))
    crop = image_np[y1:y2, x1:x2].copy()
    crop_mask = mask["segmentation"][y1:y2, x1:x2].astype(bool)
    if crop.size == 0 or crop_mask.size == 0:
        return image.copy()

    background = np.full_like(crop, 127)
    crop = np.where(crop_mask[..., None], crop, background)
    return Image.fromarray(crop)


def build_seg_maps_and_crops(
    image: Image.Image,
    level_masks: Dict[str, List[Dict]],
) -> Tuple[np.ndarray, List[Image.Image]]:
    height, width = image.height, image.width
    seg_maps = []
    crops: List[Image.Image] = []
    feature_offset = 0

    for level in LEVEL_NAMES:
        masks = level_masks[level]
        seg_map = -np.ones((height, width), dtype=np.int32)
        for local_idx, mask in enumerate(masks):
            seg_map[mask["segmentation"].astype(bool)] = feature_offset + local_idx
            crops.append(crop_with_mask(image, mask))
        feature_offset += len(masks)
        seg_maps.append(seg_map)

    return np.stack(seg_maps, axis=0), crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LangSplat-style CLIP/SAM language features.")
    parser.add_argument("--dataset", "--scene_root", dest="dataset", type=str, required=True)
    parser.add_argument("--image_dirname", type=str, default="images")
    parser.add_argument("--output_dirname", type=str, default="language_features")
    parser.add_argument("--clip_model", type=str, default="ViT-B-16")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s34b_b88k")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_sam", action="store_true")
    parser.add_argument("--sam_ckpt", type=str, default="")
    parser.add_argument("--sam_type", type=str, default="vit_h")
    parser.add_argument("--max_masks_per_level", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    image_dir = dataset / args.image_dirname
    output_dir = dataset / args.output_dirname
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = CLIPRegionEncoder(args.clip_model, args.clip_pretrained, device, args.batch_size)

    sam_generator = None
    if args.use_sam:
        if not args.sam_ckpt:
            raise ValueError("--use_sam requires --sam_ckpt")
        sam_generator = build_sam_generator(args.sam_ckpt, args.sam_type)

    for image_path in tqdm(list_images(image_dir), desc="Extracting language features"):
        out_prefix = output_dir / image_path.stem
        seg_path = Path(f"{out_prefix}_s.npy")
        feat_path = Path(f"{out_prefix}_f.npy")
        if not args.overwrite and seg_path.exists() and feat_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        if sam_generator is not None:
            masks = sam_generator.generate(np.asarray(image))
            level_masks = split_sam_masks_by_level(masks, image.height, image.width, args.max_masks_per_level)
        else:
            level_masks = fallback_grid_masks(image.height, image.width)

        seg_maps, crops = build_seg_maps_and_crops(image, level_masks)
        features = encoder.encode(crops)

        np.save(seg_path, seg_maps)
        np.save(feat_path, features)


if __name__ == "__main__":
    main()
