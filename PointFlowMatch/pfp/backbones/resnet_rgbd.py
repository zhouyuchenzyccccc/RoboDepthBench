import re
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from diffusion_policy.model.vision.model_getter import get_resnet
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


def _natural_key(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*?)(\d+)$", name)
    if match is None:
        return name, -1
    prefix, index = match.groups()
    return prefix, int(index)


def _sorted_obs_keys(keys: Iterable[str]) -> list[str]:
    return sorted(keys, key=_natural_key)


def _resolve_modality_keys(
    shape_meta: dict,
    rgb_keys: Sequence[str] | None,
    depth_keys: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    if rgb_keys is not None and depth_keys is not None:
        return _sorted_obs_keys(rgb_keys), _sorted_obs_keys(depth_keys)

    inferred_rgb_keys = []
    inferred_depth_keys = []
    for key, attr in shape_meta["obs"].items():
        obs_type = attr.get("type", "low_dim")
        key_lower = key.lower()
        if obs_type == "low_dim":
            continue
        if obs_type == "depth" or key_lower.startswith("depth") or "depth" in key_lower:
            inferred_depth_keys.append(key)
            continue
        if obs_type == "rgb" or key_lower.startswith("img") or "rgb" in key_lower:
            inferred_rgb_keys.append(key)

    if rgb_keys is None:
        rgb_keys = inferred_rgb_keys
    if depth_keys is None:
        depth_keys = inferred_depth_keys
    return _sorted_obs_keys(rgb_keys), _sorted_obs_keys(depth_keys)


def _build_encoder_shape_meta(shape_meta: dict, keys: Sequence[str], channels: int | None = None) -> dict:
    obs_shape_meta = {}
    for key in keys:
        if key not in shape_meta["obs"]:
            raise KeyError(f"Observation key '{key}' is missing from shape_meta")
        attr = shape_meta["obs"][key]
        shape = list(attr["shape"])
        if len(shape) != 3:
            raise ValueError(f"Observation key '{key}' must have image shape [C, H, W]")
        if channels is not None:
            shape[0] = channels
        obs_shape_meta[key] = {
            "shape": shape,
            "type": "rgb",
        }
    return {"obs": obs_shape_meta}


class RGBDFeatureFusion(nn.Module):
    SUPPORTED_MODES = {
        "concat",
        "add",
        "weighted_sum",
        "gated",
        "film",
        "rgb_only",
        "depth_only",
    }

    def __init__(
        self,
        mode: str,
        rgb_dim: int,
        depth_dim: int,
        out_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            supported = ", ".join(sorted(self.SUPPORTED_MODES))
            raise ValueError(f"Unsupported fusion mode '{mode}'. Supported modes: {supported}")

        self.mode = mode
        self.out_dim = rgb_dim if out_dim is None else out_dim
        hidden_dim = self.out_dim if hidden_dim is None else hidden_dim

        self.rgb_proj = nn.Identity() if rgb_dim == self.out_dim else nn.Linear(rgb_dim, self.out_dim)
        self.depth_proj = (
            nn.Identity() if depth_dim == self.out_dim else nn.Linear(depth_dim, self.out_dim)
        )

        if self.mode == "concat":
            self.fuser = nn.Sequential(
                nn.Linear(rgb_dim + depth_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, self.out_dim),
            )
        elif self.mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(rgb_dim + depth_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Sigmoid(),
            )
        elif self.mode == "film":
            self.affine = nn.Sequential(
                nn.Linear(depth_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, self.out_dim * 2),
            )
        elif self.mode == "weighted_sum":
            self.depth_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, rgb_features: torch.Tensor, depth_features: torch.Tensor) -> torch.Tensor:
        if self.mode == "rgb_only":
            return self.rgb_proj(rgb_features)
        if self.mode == "depth_only":
            return self.depth_proj(depth_features)
        if self.mode == "concat":
            return self.fuser(torch.cat([rgb_features, depth_features], dim=-1))

        rgb_projected = self.rgb_proj(rgb_features)
        depth_projected = self.depth_proj(depth_features)

        if self.mode == "add":
            return rgb_projected + depth_projected
        if self.mode == "weighted_sum":
            alpha = torch.sigmoid(self.depth_weight)
            return (1.0 - alpha) * rgb_projected + alpha * depth_projected
        if self.mode == "gated":
            gate = self.gate(torch.cat([rgb_features, depth_features], dim=-1))
            return rgb_projected + gate * depth_projected
        if self.mode == "film":
            gamma, beta = self.affine(depth_features).chunk(2, dim=-1)
            return rgb_projected * (1.0 + gamma) + beta
        raise RuntimeError(f"Fusion mode '{self.mode}' is not implemented")


class ResnetRGBD(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str = "resnet18",
        depth_model_name: str | None = None,
        crop_shape: tuple[int, int] | None = (76, 76),
        random_crop: bool = True,
        use_group_norm: bool = True,
        share_rgb_model: bool = False,
        share_depth_model: bool = False,
        imagenet_norm: bool = True,
        depth_imagenet_norm: bool = False,
        fusion_mode: str = "concat",
        fusion_hidden_dim: int | None = None,
        fused_feature_dim: int | None = None,
        rgb_keys: Sequence[str] | None = None,
        depth_keys: Sequence[str] | None = None,
        depth_repeat_channels: int = 3,
    ) -> None:
        super().__init__()
        self.rgb_keys, self.depth_keys = _resolve_modality_keys(shape_meta, rgb_keys, depth_keys)
        if len(self.rgb_keys) == 0:
            raise ValueError("No RGB observation keys were found for ResnetRGBD")
        if len(self.depth_keys) == 0:
            raise ValueError("No depth observation keys were found for ResnetRGBD")
        if len(self.rgb_keys) != len(self.depth_keys):
            raise ValueError("RGB and depth observation keys must have the same length")
        if depth_repeat_channels < 1:
            raise ValueError("depth_repeat_channels must be >= 1")

        self.depth_repeat_channels = depth_repeat_channels
        self.rgb_encoder = MultiImageObsEncoder(
            shape_meta=_build_encoder_shape_meta(shape_meta, self.rgb_keys),
            rgb_model=get_resnet(name=rgb_model_name),
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            share_rgb_model=share_rgb_model,
            imagenet_norm=imagenet_norm,
        )
        self.depth_encoder = MultiImageObsEncoder(
            shape_meta=_build_encoder_shape_meta(
                shape_meta,
                self.depth_keys,
                channels=depth_repeat_channels,
            ),
            rgb_model=get_resnet(name=depth_model_name or rgb_model_name),
            crop_shape=crop_shape,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            share_rgb_model=share_depth_model,
            imagenet_norm=depth_imagenet_norm,
        )

        rgb_feature_dim = self.rgb_encoder.output_shape()[0]
        depth_feature_dim = self.depth_encoder.output_shape()[0]
        self.fusion = RGBDFeatureFusion(
            mode=fusion_mode,
            rgb_dim=rgb_feature_dim,
            depth_dim=depth_feature_dim,
            out_dim=fused_feature_dim,
            hidden_dim=fusion_hidden_dim,
        )
        return

    def _to_channel_last(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 5:
            images = images.unsqueeze(-1)
        if images.ndim != 6:
            raise ValueError("Expected image tensor with shape [B, T, N, H, W, C] or [B, T, N, C, H, W]")
        if images.shape[-1] <= 4:
            return images
        if images.shape[3] <= 4:
            return images.permute(0, 1, 2, 4, 5, 3)
        raise ValueError("Could not infer channel dimension for image tensor")

    def _split_modalities(
        self,
        images: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor] | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(images, dict):
            if "rgb" not in images or "depth" not in images:
                raise KeyError("RGBD dict input must contain 'rgb' and 'depth' keys")
            rgb_images = images["rgb"]
            depth_images = images["depth"]
        elif isinstance(images, (tuple, list)):
            if len(images) != 2:
                raise ValueError("RGBD tuple input must contain (rgb, depth)")
            rgb_images, depth_images = images
        elif isinstance(images, torch.Tensor):
            images = self._to_channel_last(images)
            if images.shape[-1] < 4:
                raise ValueError("Stacked RGBD tensor must have at least 4 channels in the last dimension")
            rgb_images = images[..., :3]
            depth_images = images[..., 3:]
        else:
            raise TypeError("Unsupported RGBD input type")

        rgb_images = self._to_channel_last(rgb_images)
        depth_images = self._to_channel_last(depth_images)
        if rgb_images.shape[:3] != depth_images.shape[:3]:
            raise ValueError("RGB and depth tensors must match in batch, time, and view dimensions")
        if rgb_images.shape[2] != len(self.rgb_keys):
            raise ValueError("RGB tensor view dimension does not match configured rgb_keys")
        if depth_images.shape[2] != len(self.depth_keys):
            raise ValueError("Depth tensor view dimension does not match configured depth_keys")
        return rgb_images, depth_images

    def _prepare_depth_channels(self, depth_images: torch.Tensor) -> torch.Tensor:
        depth_channels = depth_images.shape[-1]
        if depth_channels == self.depth_repeat_channels:
            return depth_images
        if depth_channels == 1:
            return depth_images.repeat_interleave(self.depth_repeat_channels, dim=-1)
        depth_images = depth_images.mean(dim=-1, keepdim=True)
        return depth_images.repeat_interleave(self.depth_repeat_channels, dim=-1)

    def _flatten_visual_batch(self, images: torch.Tensor) -> torch.Tensor:
        return images.reshape(-1, *images.shape[2:]).permute(0, 1, 4, 2, 3).float()

    def forward(
        self,
        images: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor] | dict[str, torch.Tensor],
        robot_state_obs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rgb_images, depth_images = self._split_modalities(images)
        B = rgb_images.shape[0]

        rgb_images = self._flatten_visual_batch(rgb_images)
        depth_images = self._flatten_visual_batch(self._prepare_depth_channels(depth_images))

        rgb_obs_dict = {key: rgb_images[:, i] for i, key in enumerate(self.rgb_keys)}
        depth_obs_dict = {key: depth_images[:, i] for i, key in enumerate(self.depth_keys)}

        rgb_features = self.rgb_encoder(rgb_obs_dict)
        depth_features = self.depth_encoder(depth_obs_dict)
        fused_features = self.fusion(rgb_features, depth_features)

        if robot_state_obs is not None:
            robot_state_obs = robot_state_obs.float().reshape(-1, *robot_state_obs.shape[2:])
            fused_features = torch.cat([fused_features, robot_state_obs], dim=-1)

        return fused_features.reshape(B, -1)