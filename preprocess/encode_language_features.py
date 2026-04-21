import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import FeatureAutoEncoder


def load_autoencoder(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = FeatureAutoEncoder(
        input_dim=config.get("input_dim", 512),
        latent_dim=config.get("latent_dim", 3),
        encoder_hidden=config.get("encoder_hidden", [256, 128, 32]),
        decoder_hidden=config.get("decoder_hidden", [32, 128, 256]),
        normalize_input=config.get("normalize_input", True),
        normalize_latent=config.get("normalize_latent", False),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model


def list_feature_files(input_dir: Path):
    files = sorted(glob.glob(str(input_dir / "*_f.npy")))
    if not files:
        raise FileNotFoundError(f"No *_f.npy files found in {input_dir}")
    return [Path(path) for path in files]


@torch.no_grad()
def encode_features(model, features: np.ndarray, device: str, chunk_size: int):
    features = features.astype(np.float32, copy=False)
    outputs = []
    for start in range(0, features.shape[0], chunk_size):
        batch = torch.from_numpy(features[start:start + chunk_size]).to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            latent = model.encode(batch).float()
        outputs.append(latent.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode 512-d language features to dim3 features.")
    parser.add_argument("--dataset", "--scene_root", dest="dataset", type=str, required=True)
    parser.add_argument("--input_dirname", type=str, default="language_features")
    parser.add_argument("--output_dirname", type=str, default="language_features_dim3")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    input_dir = dataset / args.input_dirname
    output_dir = dataset / args.output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(__file__).resolve().parent / "checkpoints" / "ae_dim3.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_autoencoder(checkpoint_path, device)

    for feature_path in tqdm(list_feature_files(input_dir), desc="Encoding language features"):
        stem = feature_path.name[:-6]
        seg_path = input_dir / f"{stem}_s.npy"
        out_feature_path = output_dir / f"{stem}_f.npy"
        out_seg_path = output_dir / f"{stem}_s.npy"

        if not args.overwrite and out_feature_path.exists() and out_seg_path.exists():
            continue

        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation map: {seg_path}")

        features = np.load(feature_path)
        if features.ndim != 2:
            raise ValueError(f"Expected [N, 512] features in {feature_path}, got shape {features.shape}")

        encoded = encode_features(model, features, device, args.chunk_size)
        np.save(out_feature_path, encoded)
        shutil.copyfile(seg_path, out_seg_path)


if __name__ == "__main__":
    main()
