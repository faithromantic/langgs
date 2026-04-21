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
import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def _rasterize_precomputed_features(
    rasterizer: GaussianRasterizer,
    xyz: torch.Tensor,
    screenspace_points: torch.Tensor,
    opacities: torch.Tensor,
    scaling: torch.Tensor,
    rot: torch.Tensor,
    features: torch.Tensor,
):
    return rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=features,
        opacities=opacities,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
    )


def generate_neural_gaussians(
    viewpoint_camera,
    pc: GaussianModel,
    visible_mask=None,
    is_training=False,
    render_semantic=False,
):
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ob_view = anchor - viewpoint_camera.camera_center
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    ob_view = ob_view / ob_dist

    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

        feat = feat.unsqueeze(dim=-1)
        feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
               feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
               feat[:, ::1, :1] * bank_weight[:, :, 2:]
        feat = feat.squeeze(dim=-1)

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

    appearance = None
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(
            cat_local_view[:, 0], dtype=torch.long, device=ob_dist.device
        ) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > 0.0).view(-1)
    opacity = neural_opacity[mask]

    # color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])

    semantic = None
    if render_semantic:
        if pc.add_semantic_dist:
            semantic = pc.get_semantic_mlp(cat_local_view)
        else:
            semantic = pc.get_semantic_mlp(cat_local_view_wodist)
        semantic = semantic.reshape([anchor.shape[0] * pc.n_offsets, pc.semantic_dim])

    # cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])

    offsets = grid_offsets.view([-1, 3])

    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)

    if render_semantic:
        concatenated_all = torch.cat(
            [concatenated_repeated, color, scale_rot, offsets, semantic], dim=-1
        )
    else:
        concatenated_all = torch.cat(
            [concatenated_repeated, color, scale_rot, offsets], dim=-1
        )
    masked = concatenated_all[mask]
    if render_semantic:
        scaling_repeat, repeat_anchor, color, scale_rot, offsets, semantic = masked.split(
            [6, 3, 3, 7, 3, pc.semantic_dim], dim=-1
        )
    else:
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split(
            [6, 3, 3, 7, 3], dim=-1
        )

    scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
    rot = pc.rotation_activation(scale_rot[:, 3:7])

    offsets = offsets * scaling_repeat[:, :3]
    xyz = repeat_anchor + offsets

    if render_semantic:
        # Normalize semantic before rasterization, matching LangSplat style.
        semantic = semantic / (semantic.norm(dim=-1, keepdim=True) + 1e-9)

    if is_training:
        return xyz, color, opacity, scaling, rot, semantic, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot, semantic


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, visible_mask=None, retain_grad=False, render_semantic=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, semantic, neural_opacity, mask = generate_neural_gaussians(
            viewpoint_camera,
            pc,
            visible_mask,
            is_training=is_training,
            render_semantic=render_semantic,
        )
    else:
        xyz, color, opacity, scaling, rot, semantic = generate_neural_gaussians(
            viewpoint_camera,
            pc,
            visible_mask,
            is_training=is_training,
            render_semantic=render_semantic,
        )

    screenspace_points = torch.zeros_like(
        xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
    ) + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    semantic_feature_image = None
    if render_semantic:
        rendered_image, semantic_feature_image, radii = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=None,
            colors_precomp=color,
            semantic_precomp=semantic,
            opacities=opacity,
            scales=scaling,
            rotations=rot,
            cov3D_precomp=None,
        )
    else:
        rendered_image, radii = _rasterize_precomputed_features(
            rasterizer,
            xyz,
            screenspace_points,
            opacity,
            scaling,
            rot,
            color,
        )

    if is_training:
        return {
            "render": rendered_image,
            "semantic_feature_image": semantic_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
        }
    else:
        return {
            "render": rendered_image,
            "semantic_feature_image": semantic_feature_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                    scaling_modifier=1.0, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(
        pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(
        means3D=means3D,
        scales=scales[:, :3],
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    return radii_pure > 0
