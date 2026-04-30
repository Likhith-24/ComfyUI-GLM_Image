"""
GLM-Image VAE wrapper for native ComfyUI integration.

Wraps the AutoencoderKL inside a loaded GlmImagePipeline so that stock
VAEDecode / VAEEncode / VAEEncodeForInpaint nodes can call it directly.

GLM uses per-channel `latents_mean` and `latents_std` (16-element vectors,
not a scalar `scaling_factor`). The wrapper applies these inside `decode()`
and `encode()` so latents flowing in/out of KSampler are already
"sampler-space" (mean 0, std 1).

Truth source: diffusers/pipelines/glm_image/pipeline_glm_image.py
  - encode side : line 904  `condition_latent = (condition_latent - mean) / std`
  - decode side : line 1037 `latents = latents * std + mean`
"""

# MANUAL bug-fix (Apr 2026): Phase 1 of native GLM-Image integration.

import torch


class GLMImageVAEWrapper:
    """Duck-typed VAE compatible with comfy.sd.VAE for the methods that
    VAEDecode / VAEEncode / VAEEncodeTiled / VAEEncodeForInpaint actually
    invoke. NOT a subclass of comfy.sd.VAE because that class' __init__
    requires a state_dict load — we already have a live nn.Module."""

    def __init__(self, ae_module, latents_mean, latents_std,
                 vae_scale_factor=8, latent_channels=16, dtype=None):
        self.first_stage_model = ae_module
        self._ae = ae_module
        self.latents_mean = torch.tensor(list(latents_mean), dtype=torch.float32).view(1, latent_channels, 1, 1)
        self.latents_std = torch.tensor(list(latents_std), dtype=torch.float32).view(1, latent_channels, 1, 1)
        self.vae_scale_factor = vae_scale_factor
        self.latent_channels = latent_channels
        self.downscale_ratio = vae_scale_factor
        self.upscale_ratio = vae_scale_factor
        self.latent_dim = 2  # 2D latents (not video)
        self.output_channels = 3
        self.not_video = True
        self.disable_offload = False
        # Use bf16 for AE compute when available, else fp32
        if dtype is None:
            try:
                dtype = next(ae_module.parameters()).dtype
            except StopIteration:
                dtype = torch.float32
        self.vae_dtype = dtype
        self.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
        # Conservative memory estimate (mirrors comfy.sd.VAE AutoencoderKL default)
        self.memory_used_decode = lambda shape, dt: (2178 * shape[2] * shape[3] * 64) * 2  # bytes-ish
        self.memory_used_encode = lambda shape, dt: (1767 * shape[2] * shape[3]) * 2
        self.size = None
        self.process_input = lambda image: image * 2.0 - 1.0
        self.process_output = lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)

    # ---- ComfyUI-facing API ---------------------------------------------

    def spacial_compression_decode(self):
        return self.vae_scale_factor

    def spacial_compression_encode(self):
        return self.vae_scale_factor

    def temporal_compression_decode(self):
        return None  # 2D, not video

    def get_sd(self):
        # Used by some downstream nodes; return underlying state dict.
        return self._ae.state_dict()

    # ---- Decode ---------------------------------------------------------

    @torch.no_grad()
    def decode(self, samples_in):
        """
        samples_in: torch.Tensor [B, 16, H/8, W/8]  (sampler-space latents)
        returns:    torch.Tensor [B, H, W, 3]       in [0, 1]  (ComfyUI image format)
        """
        device = next(self._ae.parameters()).device
        dtype = self.vae_dtype
        x = samples_in.to(device=device, dtype=dtype)

        mean = self.latents_mean.to(device=device, dtype=dtype)
        std = self.latents_std.to(device=device, dtype=dtype)

        # Un-normalize: sampler-space -> diffusers VAE-space
        latents = x * std + mean

        # Decode: [B, 3, H, W] in [-1, 1]
        out = self._ae.decode(latents, return_dict=False)[0]

        # ComfyUI standard: [B, H, W, 3] in [0, 1]
        out = (out.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out

    # ---- Encode ---------------------------------------------------------

    @torch.no_grad()
    def encode(self, pixel_samples):
        """
        pixel_samples: torch.Tensor [B, H, W, 3] in [0, 1]
        returns:       torch.Tensor [B, 16, H/8, W/8] sampler-space latents
        """
        device = next(self._ae.parameters()).device
        dtype = self.vae_dtype
        x = pixel_samples.to(device=device, dtype=dtype)
        # Crop H/W to multiples of vae_scale_factor (mirrors stock behaviour)
        sf = self.vae_scale_factor
        h = (x.shape[1] // sf) * sf
        w = (x.shape[2] // sf) * sf
        x = x[:, :h, :w, :]
        # [B, H, W, 3] -> [B, 3, H, W] in [-1, 1]
        x = x.permute(0, 3, 1, 2)
        x = x * 2.0 - 1.0

        # Encode -> distribution; sample (the diffusers `retrieve_latents` helper
        # uses sample_mode='argmax' for img2img conditioning, but standard ComfyUI
        # img2img uses .sample(); we follow .sample() for randomness compatibility
        # with KSampler denoise<1 workflows.)
        enc = self._ae.encode(x)
        if hasattr(enc, "latent_dist"):
            latents = enc.latent_dist.sample()
        elif isinstance(enc, (tuple, list)):
            latents = enc[0]
        else:
            latents = enc

        mean = self.latents_mean.to(device=device, dtype=latents.dtype)
        std = self.latents_std.to(device=device, dtype=latents.dtype)
        latents = (latents - mean) / std
        return latents.float()

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64,
                     tile_t=None, overlap_t=None):
        # Fallback: do non-tiled encode. Tiling for GLM AE not implemented yet.
        return self.encode(pixel_samples)

    def decode_tiled(self, samples, tile_x=512, tile_y=512, overlap=64,
                     tile_t=None, overlap_t=None):
        return self.decode(samples)
