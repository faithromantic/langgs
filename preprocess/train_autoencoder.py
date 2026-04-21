import os
import glob
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from model import FeatureAutoEncoder, reconstruction_loss, to_config_dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_feature_files(raw_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(raw_dir, "*_f.npy")))
    if len(files) == 0:
        raise FileNotFoundError(f"No *_f.npy files found in {raw_dir}")
    return files


def sample_from_feature_map(
    arr: np.ndarray,
    max_samples_per_file: int,
    min_norm: float = 1e-6,
) -> np.ndarray:
    """
    arr: [H, W, C] or [N, C]
    """
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    elif arr.ndim != 2:
        raise ValueError(f"Unexpected feature shape: {arr.shape}")

    norms = np.linalg.norm(arr, axis=-1)
    valid = norms > min_norm
    arr = arr[valid]

    if len(arr) == 0:
        return np.zeros((0, 512), dtype=np.float32)

    if len(arr) > max_samples_per_file:
        idx = np.random.choice(len(arr), size=max_samples_per_file, replace=False)
        arr = arr[idx]

    return arr.astype(np.float32)


def collect_training_features(
    raw_dir: str,
    cache_path: str,
    max_samples_per_file: int = 20000,
):
    files = list_feature_files(raw_dir)
    chunks = []

    print(f"[Info] Collecting training features from {len(files)} files...")
    for fp in tqdm(files):
        arr = np.load(fp)
        sampled = sample_from_feature_map(arr, max_samples_per_file=max_samples_per_file)
        if sampled.shape[0] > 0:
            chunks.append(sampled)

    if len(chunks) == 0:
        raise RuntimeError("No valid features collected for autoencoder training.")

    feats = np.concatenate(chunks, axis=0).astype(np.float32)
    np.save(cache_path, feats)
    print(f"[Info] Saved sampled features to {cache_path}, shape={feats.shape}")


class FeatureDataset(Dataset):
    def __init__(self, feat_npy: str):
        self.data = np.load(feat_npy, mmap_mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.data[idx], dtype=np.float32))
        return x


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=(num_workers > 0),
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    use_amp: bool,
    mse_weight: float,
    cosine_weight: float,
):
    model.train()
    meters = {"loss": 0.0, "mse": 0.0, "cosine": 0.0}
    count = 0

    for x in loader:
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _, x_hat = model(x)
            loss_dict = reconstruction_loss(
                x, x_hat,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight
            )
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.shape[0]
        count += bs
        for k in meters:
            meters[k] += float(loss_dict[k].detach().item()) * bs

    for k in meters:
        meters[k] /= max(count, 1)
    return meters


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
    use_amp: bool,
    mse_weight: float,
    cosine_weight: float,
):
    model.eval()
    meters = {"loss": 0.0, "mse": 0.0, "cosine": 0.0}
    count = 0

    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _, x_hat = model(x)
            loss_dict = reconstruction_loss(
                x, x_hat,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight
            )

        bs = x.shape[0]
        count += bs
        for k in meters:
            meters[k] += float(loss_dict[k].detach().item()) * bs

    for k in meters:
        meters[k] /= max(count, 1)
    return meters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "--scene_root", dest="scene_root", type=str, required=True)
    parser.add_argument("--raw_dirname", type=str, default="language_features")
    parser.add_argument("--cache_name", type=str, default="ae_train_cache_dim3.npy")
    parser.add_argument("--max_samples_per_file", type=int, default=20000)

    parser.add_argument("--input_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--encoder_hidden", nargs="+", type=int, default=[256, 128, 32])
    parser.add_argument("--decoder_hidden", nargs="+", type=int, default=[32, 128, 256])

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32768)  # 4090 24G 推荐
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="ae_dim3.pth")
    args = parser.parse_args()

    set_seed(args.seed)

    scene_root = Path(args.scene_root)
    raw_dir = scene_root / args.raw_dirname
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_path = ckpt_dir / args.cache_name

    if not cache_path.exists():
        collect_training_features(
            raw_dir=str(raw_dir),
            cache_path=str(cache_path),
            max_samples_per_file=args.max_samples_per_file,
        )

    dataset = FeatureDataset(str(cache_path))
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_amp)

    model = FeatureAutoEncoder(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        encoder_hidden=args.encoder_hidden,
        decoder_hidden=args.decoder_hidden,
        normalize_input=True,
        normalize_latent=False,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    history = []

    print(f"[Info] Train samples: {n_train}, Val samples: {n_val}")
    print(f"[Info] Training autoencoder on {device}, AMP={use_amp}")

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp,
            args.mse_weight, args.cosine_weight
        )
        val_m = eval_one_epoch(
            model, val_loader, device, use_amp,
            args.mse_weight, args.cosine_weight
        )

        log_str = (
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_m['loss']:.6f}, train_cos={train_m['cosine']:.6f}, "
            f"val_loss={val_m['loss']:.6f}, val_cos={val_m['cosine']:.6f}"
        )
        print(log_str)

        history.append({
            "epoch": epoch,
            "train": train_m,
            "val": val_m,
        })

        save_obj = {
            "model_state": model.state_dict(),
            "config": {
                **to_config_dict(model),
                "encoder_hidden": args.encoder_hidden,
                "decoder_hidden": args.decoder_hidden,
            },
            "epoch": epoch,
            "history": history,
        }

        latest_path = ckpt_dir / "latest.pth"
        torch.save(save_obj, latest_path)

        if val_m["loss"] < best_val:
            best_val = val_m["loss"]
            best_path = ckpt_dir / args.output_name
            torch.save(save_obj, best_path)
            print(f"[Info] Saved best checkpoint to {best_path}")

    with open(ckpt_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[Done] Best val loss = {best_val:.6f}")


if __name__ == "__main__":
    main()
