"""
GLM-Image latent format for native ComfyUI integration.

Identity scale_factor (1.0) — the GLMImageVAEWrapper already handles
per-channel mean/std normalization, so latents arriving here are already
in sampler-space.
"""

# MANUAL bug-fix (Apr 2026): Phase 3 of native GLM-Image integration.

import torch
from comfy.latent_formats import LatentFormat


class GLMImageLatentFormat(LatentFormat):
    latent_channels = 16
    latent_dimensions = 2  # 2D (image), not 3D (video)

    def __init__(self):
        self.scale_factor = 1.0
        # latent_rgb_factors set to a Wan21-ish placeholder for KSampler
        # preview rendering. Visually approximate; not parity-critical.
        self.latent_rgb_factors = [
            [-0.1299, -0.1692,  0.2932],
            [ 0.0671,  0.0406,  0.0442],
            [ 0.3568,  0.2548,  0.1747],
            [ 0.0372,  0.2344,  0.1420],
            [ 0.0313,  0.0189, -0.0328],
            [ 0.0296, -0.0956, -0.0665],
            [-0.3477, -0.4059, -0.2925],
            [ 0.0166,  0.1902,  0.1975],
            [-0.0412,  0.0267, -0.1364],
            [-0.1293,  0.0740,  0.1636],
            [ 0.0680,  0.3019,  0.1128],
            [ 0.0032,  0.0581,  0.0639],
            [-0.1251,  0.0927,  0.1699],
            [ 0.0060, -0.0633,  0.0005],
            [ 0.3477,  0.2275,  0.2950],
            [ 0.1984,  0.0913,  0.1861],
        ]
        self.latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]

    # process_in / process_out are identity (inherited).
