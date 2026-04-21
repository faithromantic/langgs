#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda"
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width),
                device=self.data_device
            )

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).cuda()

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy
        ).transpose(0, 1).cuda()

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_language_feature(self, language_feature_dir, feature_level):
        language_feature_name = os.path.join(language_feature_dir, self.image_name)
        seg_feature_path = language_feature_name + "_s.npy"
        point_feature_path = language_feature_name + "_f.npy"

        if not os.path.exists(seg_feature_path):
            raise FileNotFoundError(f"Semantic segmentation map not found: {seg_feature_path}")
        if not os.path.exists(point_feature_path):
            raise FileNotFoundError(f"Semantic feature table not found: {point_feature_path}")

        seg_map = torch.from_numpy(np.load(seg_feature_path)).long()
        feature_map = torch.from_numpy(np.load(point_feature_path)).float()

        if feature_map.ndim != 2:
            raise ValueError(
                f"Expected semantic feature table [num_regions, semantic_dim], "
                f"got {tuple(feature_map.shape)} from {point_feature_path}"
            )

        if seg_map.ndim == 2:
            if feature_level != 0:
                raise ValueError(
                    f"Semantic map {seg_feature_path} has a single level, "
                    f"but feature_level={feature_level} was requested"
                )
            seg_map = seg_map.unsqueeze(0)
        elif seg_map.ndim != 3:
            raise ValueError(
                f"Expected semantic segmentation map [H, W] or [num_levels, H, W], "
                f"got {tuple(seg_map.shape)} from {seg_feature_path}"
            )

        if feature_level < 0 or feature_level >= seg_map.shape[0]:
            raise ValueError(
                f"Invalid feature_level={feature_level} for {seg_feature_path}; "
                f"available levels: 0..{seg_map.shape[0] - 1}"
            )

        seg_level = seg_map[feature_level]
        if seg_level.shape != (self.image_height, self.image_width):
            seg_level = F.interpolate(
                seg_level.unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_height, self.image_width),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        semantic_dim = feature_map.shape[1]
        gt_semantic = torch.zeros(
            (self.image_height * self.image_width, semantic_dim),
            dtype=feature_map.dtype,
        )

        seg_flat = seg_level.reshape(-1)
        valid_flat = (seg_flat >= 0) & (seg_flat < feature_map.shape[0])
        if valid_flat.any():
            gt_semantic[valid_flat] = feature_map[seg_flat[valid_flat]]

        gt_semantic = gt_semantic.reshape(
            self.image_height,
            self.image_width,
            semantic_dim,
        ).permute(2, 0, 1).contiguous()

        valid_mask = valid_flat.reshape(1, self.image_height, self.image_width).float()

        return gt_semantic.to(self.data_device), valid_mask.to(self.data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
