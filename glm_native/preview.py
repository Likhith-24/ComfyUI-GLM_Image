"""
GLM-Image tiny preview decoder.

Uses the latent_rgb_factors stored on `GLMImageLatentFormat` to do an
ultra-fast 16-channel-latent → RGB approximation (no VAE, no GPU pass).
Useful for live previews mid-sampling and as a sanity check before VAE
decode.
"""

# MANUAL bug-fix (Apr 2026): Native GLM-Image — tiny latent previewer.

import torch

from .latent_format import GLMImageLatentFormat


def latent_to_rgb_preview(samples: torch.Tensor) -> torch.Tensor:
    """Convert a (B, 16, h, w) latent into a (B, h*8, w*8, 3) RGB preview tensor
    in [0, 1] using the latent_rgb_factors. Output is upsampled by nearest.
    """
    if samples.ndim != 4 or samples.shape[1] != 16:
        raise ValueError(f"expected (B,16,h,w) latent, got {tuple(samples.shape)}")
    fmt = GLMImageLatentFormat()
    factors = torch.tensor(fmt.latent_rgb_factors,
                           dtype=samples.dtype, device=samples.device)  # (16,3)
    bias = torch.tensor(fmt.latent_rgb_factors_bias,
                        dtype=samples.dtype, device=samples.device)     # (3,)

    # samples: (B, 16, h, w) -> (B, h, w, 16) @ (16, 3) = (B, h, w, 3)
    x = samples.permute(0, 2, 3, 1).contiguous()
    rgb = torch.einsum("bhwc,co->bhwo", x, factors) + bias
    rgb = rgb.clamp(0.0, 1.0)
    # Upsample 8x to align with VAEDecode output spatial scale
    rgb = rgb.permute(0, 3, 1, 2)
    rgb = torch.nn.functional.interpolate(rgb, scale_factor=8, mode="nearest")
    rgb = rgb.permute(0, 2, 3, 1).contiguous()
    return rgb
