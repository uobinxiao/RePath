from typing import Tuple, Union
import numbers
import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision.transforms.functional as F
from skimage import color

class HEDJitter(nn.Module):
    """
    Randomly perturb the HED color space of an RGB image.

    Args:
        theta (float): Jitter magnitude. alpha ~ U[1-theta, 1+theta], beta ~ U[-theta, theta].
                       Formula: s' = alpha * s + beta, applied to each H, E, D channel.
        p (float):     Probability of applying the transform.
        resample_each_call (bool): If True, resample alpha/beta on every __call__/forward;
                                   otherwise, sample once at initialization and keep them
                                   fixed for reproducible experiments.
        clip (bool):   Whether to clip to [0,1] after hed2rgb before returning the result
                       in the input type and range.
        eps (float):   Small value to avoid division by zero in edge cases.
    Notes:
        - Supported inputs: PIL.Image (RGB) or torch.Tensor (C,H,W). Tensors can be:
          * float32 in [0,1]
          * uint8 in [0,255]
        - The output type matches the input type (PIL -> PIL, Tensor -> Tensor).
    """
    def __init__(self, theta: float = 0.05, p: float = 1.0, resample_each_call: bool = True, clip: bool = True, eps: float = 1e-8):
        super().__init__()
        assert isinstance(theta, numbers.Number), "theta should be a single number."
        assert 0.0 <= p <= 1.0, "p must be in [0, 1]"
        self.theta = float(theta)
        self.p = float(p)
        self.resample_each_call = bool(resample_each_call)
        self.clip = bool(clip)
        self.eps = float(eps)

        if not self.resample_each_call:
            self.register_buffer("_alpha", self._sample_alpha_beta()[0])
            self.register_buffer("_beta",  self._sample_alpha_beta()[1])
        else:
            self.register_buffer("_alpha", torch.ones(1, 3))
            self.register_buffer("_beta",  torch.zeros(1, 3))

    def extra_repr(self) -> str:
        if self.resample_each_call:
            ab = "alpha=~U[1-θ,1+θ], beta=~U[-θ,θ]"
        else:
            ab = f"alpha={self._alpha.cpu().numpy()}, beta={self._beta.cpu().numpy()}"
        return (f"theta={self.theta}, p={self.p}, resample_each_call={self.resample_each_call}, "
                f"{ab}")

    @torch.no_grad()
    def forward(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if torch.rand(1).item() > self.p:
            return img

        if self.resample_each_call:
            alpha_t, beta_t = self._sample_alpha_beta()
        else:
            alpha_t, beta_t = self._alpha, self._beta

        input_is_pil = isinstance(img, Image.Image)
        input_is_tensor = torch.is_tensor(img)

        if input_is_pil:
            img_np = np.asarray(img)  # uint8, H×W×3
            img_float = img_np.astype(np.float32) / 255.0
        elif input_is_tensor:
            if img.ndim != 3 or img.shape[0] != 3:
                raise ValueError("Tensor input must be shape (3, H, W)")
            if img.dtype == torch.uint8:
                img_float = (img.float() / 255.0).clamp(0, 1)
            else:
                img_float = img.float().clamp(0, 1)
            img_float = img_float.permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError("img must be PIL.Image or torch.Tensor (C,H,W).")

        hed = color.rgb2hed(img_float)             # float, can be any range
        alpha = alpha_t.cpu().numpy().reshape(1, 1, 3)
        beta  = beta_t.cpu().numpy().reshape(1, 1, 3)
        hed_j = alpha * hed + beta
        rgb_j = color.hed2rgb(hed_j)

        if self.clip:
            rgb_j = np.clip(rgb_j, 0.0, 1.0)

        if input_is_pil:
            out_np = (rgb_j * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(out_np)
        else:
            out_t = torch.from_numpy(rgb_j).permute(2, 0, 1)  # (3,H,W), float in [0,1]
            return (out_t.mul(255.0).add_(0.5).to(torch.uint8)
                    if img.dtype == torch.uint8 else out_t)

    def _sample_alpha_beta(self) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.empty(1, 3).uniform_(1 - self.theta, 1 + self.theta)
        b = torch.empty(1, 3).uniform_(-self.theta, self.theta)
        return a, b
