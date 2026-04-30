"""
GLM-Image MODEL wrapper for native ComfyUI integration.

Builds a `comfy.model_base.BaseModel`-derived object whose
`diffusion_model` is the in-memory `GlmImageTransformer2DModel` from a
loaded `GlmImagePipeline`, then wraps it in `comfy.model_patcher.ModelPatcher`.

The override of `_apply_model()` translates ComfyUI's
`(x, sigma, c_crossattn, **kwargs)` interface into the GLM transformer's
native call signature (hidden_states, encoder_hidden_states, prior_token_id,
prior_token_drop, timestep, target_size, crop_coords, attention_mask,
kv_caches, image_rotary_emb).

`extra_conds()` unpacks the GLM-specific extras packed by GLMImageCLIPWrapper.

Truth source: pipeline_glm_image.py lines 1006-1060 (denoising step).
"""

# MANUAL bug-fix (Apr 2026): Phase 3 of native GLM-Image integration.

import math
import torch

import comfy.conds
import comfy.model_base
import comfy.model_management
import comfy.model_patcher

from .latent_format import GLMImageLatentFormat


# ---- Lightweight model_config stand-in (BaseModel reads attrs only) ---


class _GLMModelConfig:
    """Minimal `model_config` shim so `BaseModel.__init__` runs without
    loading a state_dict. We bypass unet creation and slot in our own
    diffusion_model post-init."""

    def __init__(self, transformer):
        # The exact key BaseModel checks to skip unet creation:
        self.unet_config = {
            "disable_unet_model_creation": True,
            "image_model": "glm_image",
        }
        self.unet_extra_config = {}
        self.latent_format = GLMImageLatentFormat()
        self.manual_cast_dtype = None
        self.custom_operations = None
        self.optimizations = {"fp8": False}
        # Store the live transformer so the model can grab it.
        self._transformer = transformer
        # Memory factor: rough heuristic for 30-layer DiT at bf16.
        self.memory_usage_factor = 2.5
        self.scaled_fp8 = None
        self.sampling_settings = {
            "shift": 1.0,        # diffusers default; matches FlowMatchEuler config
            "multiplier": 1000,
        }


# ---- BaseModel subclass -----------------------------------------------


class GLMImageBaseModel(comfy.model_base.BaseModel):
    def __init__(self, transformer, vlm, scheduler_config,
                 device=None, model_type=comfy.model_base.ModelType.FLUX):
        cfg = _GLMModelConfig(transformer)
        super().__init__(cfg, model_type, device=device)
        # Replace the (skipped) diffusion_model with the live transformer.
        self.diffusion_model = transformer
        # Hold reference to the VLM for img2img kv-cache write phase.
        self._vlm = vlm
        self._scheduler_config = scheduler_config
        self._patch_size = transformer.config.patch_size

    # ---- Memory ------------------------------------------------------

    def memory_required(self, input_shape, cond_shapes={}):
        # input_shape = [B, 16, h, w]. Rough VRAM cost for one DiT pass.
        b, c, h, w = input_shape
        # ~2.5 bytes / latent-pixel-step is a safe conservative figure
        return b * c * h * w * 2.5 * 4 + 256 * 1024 * 1024

    def get_dtype(self):
        try:
            return next(self.diffusion_model.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ---- Conditioning packing --------------------------------------

    def extra_conds(self, **kwargs):
        out = {}
        # cross_attn (T5 glyph embeds) packed by ConditioningSetArea/CLIPTextEncode
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out["c_crossattn"] = comfy.conds.CONDRegular(cross_attn)

        for key in ("attention_mask", "glm_prior_ids", "glm_prior_drop",
                    "glm_target_size", "glm_crop_coords"):
            v = kwargs.get(key, None)
            if v is not None:
                out[key] = comfy.conds.CONDRegular(v)

        # KV caches (img2img reference) - non-tensor, use CONDConstant.
        kv = kwargs.get("glm_kv_caches", None)
        if kv is not None:
            out["glm_kv_caches"] = comfy.conds.CONDConstant(kv)
        is_empty = kwargs.get("glm_is_empty", None)
        if is_empty is not None:
            out["glm_is_empty"] = comfy.conds.CONDConstant(is_empty)

        return out

    # ---- The hot path ------------------------------------------------

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None,
                    control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)
        dtype = self.get_dtype_inference()
        device = xc.device
        xc = xc.to(dtype)

        # diffusers pipeline does `timestep = t.expand(B) - 1`. ComfyUI's
        # FLUX/CONST sampling expects timestep equivalent to sigma * multiplier.
        timestep = self.model_sampling.timestep(sigma).float() - 1.0
        timestep = timestep.to(device)

        context = c_crossattn
        if context is not None:
            context = comfy.model_management.cast_to_device(context, device, dtype)

        # Unpack GLM-specific extras
        attention_mask = kwargs.get("attention_mask", None)
        prior_ids = kwargs.get("glm_prior_ids", None)
        prior_drop = kwargs.get("glm_prior_drop", None)
        target_size = kwargs.get("glm_target_size", None)
        crop_coords = kwargs.get("glm_crop_coords", None)
        kv_caches = kwargs.get("glm_kv_caches", None)

        B = xc.shape[0]

        # Broadcast/cast extras to (B, ...)
        def _bcast(v, want_dtype=None):
            if v is None:
                return None
            v = v.to(device)
            if want_dtype is not None:
                v = v.to(want_dtype)
            if v.shape[0] == 1 and B > 1:
                v = v.expand(B, *v.shape[1:]).contiguous()
            return v

        if attention_mask is not None:
            attention_mask = _bcast(attention_mask)
        if prior_ids is not None:
            prior_ids = _bcast(prior_ids)
        if prior_drop is not None:
            prior_drop = _bcast(prior_drop).to(torch.bool)
        else:
            # Fallback: assume cond branch -> drop=False
            if prior_ids is not None:
                prior_drop = torch.zeros_like(prior_ids, dtype=torch.bool)
        if target_size is None:
            # Derive from xc spatial size
            h_pix = xc.shape[-2] * 8
            w_pix = xc.shape[-1] * 8
            target_size = torch.tensor([[h_pix, w_pix]], dtype=dtype,
                                       device=device).expand(B, -1).contiguous()
        else:
            target_size = _bcast(target_size, want_dtype=dtype)
        if crop_coords is None:
            crop_coords = torch.zeros((B, 2), dtype=dtype, device=device)
        else:
            crop_coords = _bcast(crop_coords, want_dtype=dtype)

        if context is None:
            # Empty fallback — provide a 0-length seq
            context = torch.zeros((B, 0, 1472), dtype=dtype, device=device)

        # Call GLM diffusers transformer with its native signature.
        out = self.diffusion_model(
            hidden_states=xc,
            encoder_hidden_states=context,
            prior_token_id=prior_ids if prior_ids is not None else _zero_priors(xc, device),
            prior_token_drop=prior_drop if prior_drop is not None else _zero_priors(xc, device, dtype=torch.bool),
            timestep=timestep,
            target_size=target_size,
            crop_coords=crop_coords,
            attention_mask=attention_mask,
            kv_caches=kv_caches,
            return_dict=False,
        )[0]

        # ComfyUI calculate_denoised expects model_output in same convention.
        return self.model_sampling.calculate_denoised(sigma, out.float(), x)


def _zero_priors(xc, device, dtype=torch.long):
    """Fallback prior_token_id when CLIP didn't supply one. Shape inferred
    from xc spatial: (B, h*w) at the post-patch scale."""
    B = xc.shape[0]
    h = xc.shape[-2]
    w = xc.shape[-1]
    # patch_size already applied at VAE-latent scale 1; pipeline uses (h*w/4)
    # because patch_size=2 in the transformer divides further. We keep h*w
    # which matches pipeline's `token_h * token_w` only when patch=1.
    # Be safe: use h*w  -> if diffusers complains, adjust to (h//2)*(w//2).
    n = (h // 2) * (w // 2) if h % 2 == 0 and w % 2 == 0 else h * w
    return torch.zeros((B, n), dtype=dtype, device=device)


# ---- Builder: pipeline -> ModelPatcher --------------------------------


def build_glm_model_patcher(pipeline):
    """Construct the BaseModel + ModelPatcher pair from a loaded GlmImagePipeline."""
    transformer = pipeline.transformer
    vlm = pipeline.vision_language_encoder
    scheduler_config = pipeline.scheduler.config

    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()

    base = GLMImageBaseModel(
        transformer=transformer,
        vlm=vlm,
        scheduler_config=scheduler_config,
        device=offload_device,
    )

    # Estimate model_size for offload accounting.
    try:
        size = sum(p.numel() * p.element_size() for p in transformer.parameters())
    except Exception:
        size = 0

    patcher = comfy.model_patcher.ModelPatcher(
        base, load_device=load_device, offload_device=offload_device, size=size
    )
    base.current_patcher = patcher
    return patcher
