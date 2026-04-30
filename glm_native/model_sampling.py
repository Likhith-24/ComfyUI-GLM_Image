"""
GLM-Image dynamic flow-shift utilities.

GLM-Image's pipeline computes a per-image shift parameter:

    mu = sqrt(image_seq_len / 256) * (max_shift - base_shift) + base_shift
       = sqrt(N / 256) * 0.75 + 0.25

then exponentiates the FlowMatchEulerDiscrete sigmas with `exp(mu)`.

This module provides a `ModelPatcher`-friendly helper that swaps the
model_sampling object for one with the right dynamic shift baked in,
so stock KSampler / scheduler nodes produce GLM-correct sigmas.
"""

# MANUAL bug-fix (Apr 2026): Native GLM-Image — dynamic flow-shift.

import math
import torch

import comfy.model_sampling


def compute_glm_mu(image_seq_len: int, base_shift=0.25, max_shift=0.75,
                   base_image_seq_len=256) -> float:
    """The exact formula from FlowMatchEulerDiscreteScheduler.set_timesteps."""
    return math.sqrt(image_seq_len / base_image_seq_len) * (max_shift - base_shift) + base_shift


class ModelSamplingGLMFlow(comfy.model_sampling.ModelSamplingDiscreteFlow):
    """ModelSamplingDiscreteFlow with `mu` applied via exp(mu) shift on
    the linear timesteps, matching diffusers' use_dynamic_shifting=True."""

    def __init__(self, model_config=None, mu: float = 0.5):
        # Force shift=1.0 first; we'll re-bake sigmas after super().__init__
        if model_config is not None and hasattr(model_config, "sampling_settings"):
            ss = dict(model_config.sampling_settings)
            ss["shift"] = 1.0
            model_config.sampling_settings = ss
        super().__init__(model_config)
        self.mu = float(mu)
        # Re-bake sigmas using FlowMatchEulerDiscreteScheduler's dynamic-shift formula:
        #   sigmas = exp(mu) * sigmas / (1 + (exp(mu) - 1) * sigmas)
        ts = torch.arange(1, self.multiplier + 1, 1) / self.multiplier
        e_mu = math.exp(self.mu)
        sigmas = e_mu * ts / (1.0 + (e_mu - 1.0) * ts)
        self.register_buffer("sigmas", sigmas)

    def sigma(self, timestep):
        # Used by percent_to_sigma; fall back to interp on registered sigmas.
        e_mu = math.exp(self.mu)
        t = timestep / self.multiplier if isinstance(timestep, (int, float)) else timestep / self.multiplier
        return e_mu * t / (1.0 + (e_mu - 1.0) * t)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        e_mu = math.exp(self.mu)
        t = 1.0 - percent
        return e_mu * t / (1.0 + (e_mu - 1.0) * t)


def patch_model_sampling_for_latent(model_patcher, latent_h: int, latent_w: int,
                                    base_shift=0.25, max_shift=0.75):
    """Replace `model_patcher.model.model_sampling` with a dynamic-shift
    instance computed from the post-patch latent token count.

    GlmImageTransformer2DModel applies patch_size=2 to a /8 latent, so the
    image_seq_len = (latent_h // 2) * (latent_w // 2).
    """
    seq = (latent_h // 2) * (latent_w // 2)
    mu = compute_glm_mu(seq, base_shift=base_shift, max_shift=max_shift)
    sampling = ModelSamplingGLMFlow(model_patcher.model.model_config, mu=mu)
    sampling.to(next(model_patcher.model.parameters()).device if any(True for _ in model_patcher.model.parameters()) else "cpu")
    model_patcher.add_object_patch("model_sampling", sampling)
    return mu
