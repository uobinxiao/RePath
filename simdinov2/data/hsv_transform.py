from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch import nn
from skimage import color

class HSVJitter(nn.Module):
    """HSB/HSV color jitter."""

    def __init__(
        self,
        hue_sigma_range: Tuple[float, float] = (0.0, 0.05),
        saturation_sigma_range: Tuple[float, float] = (0.0, 0.20),
        brightness_sigma_range: Tuple[float, float] = (0.0, 0.20),
        p: float = 1.0,
        resample_each_call: bool = True,
        clip: bool = True,
    ) -> None:
        super().__init__()
        if len(hue_sigma_range) != 2 or len(saturation_sigma_range) != 2 or len(brightness_sigma_range) != 2:
            raise ValueError("ranges must be 2-tuples (min, max).")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        # Upstream allows negative values and uses them as signed shifts/log-noise.
        self.hue_sigma_range = self._sanitize_range(hue_sigma_range)
        self.saturation_sigma_range = self._sanitize_range(saturation_sigma_range)
        self.brightness_sigma_range = self._sanitize_range(brightness_sigma_range)

        self.p = float(p)
        self.resample_each_call = bool(resample_each_call)
        self.clip = bool(clip)

        # Buffers used when resample_each_call=False
        self.register_buffer("_dh", torch.zeros(1))
        self.register_buffer("_eps_s", torch.zeros(1))
        self.register_buffer("_eps_v", torch.zeros(1))

        if not self.resample_each_call:
            dh, eps_s, eps_v = self._sample_params()
            self._dh.copy_(dh)
            self._eps_s.copy_(eps_s)
            self._eps_v.copy_(eps_v)

    def extra_repr(self) -> str:
        base = (
            f"hue_sigma_range={self.hue_sigma_range}, "
            f"saturation_sigma_range={self.saturation_sigma_range}, "
            f"brightness_sigma_range={self.brightness_sigma_range}, "
            f"p={self.p}, resample_each_call={self.resample_each_call}, clip={self.clip}"
        )
        if not self.resample_each_call:
            base += (
                f", dh={self._dh.item():.4f}, eps_s={self._eps_s.item():.4f}, eps_v={self._eps_v.item():.4f}"
            )
        return base

    @torch.no_grad()
    def forward(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        # Decide whether to apply
        if torch.rand(1).item() > self.p:
            return img

        if self.resample_each_call:
            dh, eps_s, eps_v = self._sample_params()
        else:
            dh, eps_s, eps_v = self._dh, self._eps_s, self._eps_v

        input_is_pil = isinstance(img, Image.Image)
        input_is_tensor = torch.is_tensor(img)

        if input_is_pil:
            arr = np.asarray(img)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError("PIL input must be an RGB image.")
            rgb = arr.astype(np.float32) / 255.0
            out_dtype = None
        elif input_is_tensor:
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError("Tensor input must be shape (3, H, W).")
            out_dtype = img.dtype
            if img.dtype == torch.uint8:
                rgb_t = (img.float() / 255.0).clamp(0, 1)
            else:
                rgb_t = img.float().clamp(0, 1)
            rgb = rgb_t.permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError("img must be PIL.Image or torch.Tensor (C,H,W).")

        hsv = color.rgb2hsv(rgb)

        # Upstream: hue += (dh % 1.0); hue %= 1.0
        dh_mod = float((dh % 1.0).item())
        h = (hsv[..., 0] + dh_mod) % 1.0

        # Upstream: S *= exp(eps_s); V *= exp(eps_v)
        s = hsv[..., 1] * float(np.exp(float(eps_s.item())))
        v = hsv[..., 2] * float(np.exp(float(eps_v.item())))

        if self.clip:
            s = np.clip(s, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)

        hsv_j = np.stack([h, s, v], axis=-1)
        rgb_j = color.hsv2rgb(hsv_j)

        if self.clip:
            rgb_j = np.clip(rgb_j, 0.0, 1.0)

        if input_is_pil:
            out = (rgb_j * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(out)

        out_t = torch.from_numpy(rgb_j).permute(2, 0, 1)
        if out_dtype == torch.uint8:
            return (out_t * 255.0 + 0.5).to(torch.uint8)

        return out_t.to(dtype=out_dtype)

    def _sample_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (dh, eps_s, eps_v) exactly like the upstream file.

        - dh is a hue shift sampled from Uniform(hue_sigma_range)
        - eps_s is a log-domain saturation noise sampled from Uniform(saturation_sigma_range)
        - eps_v is a log-domain brightness noise sampled from Uniform(brightness_sigma_range)
        """
        dh = torch.empty(1).uniform_(self.hue_sigma_range[0], self.hue_sigma_range[1])
        eps_s = torch.empty(1).uniform_(self.saturation_sigma_range[0], self.saturation_sigma_range[1])
        eps_v = torch.empty(1).uniform_(self.brightness_sigma_range[0], self.brightness_sigma_range[1])
        return dh, eps_s, eps_v

    @staticmethod
    def _sanitize_range(r: Tuple[float, float]) -> Tuple[float, float]:
        """Return a (lo, hi) tuple with lo <= hi, preserving sign semantics."""
        a, b = float(r[0]), float(r[1])
        return (a, b) if a <= b else (b, a)
