"""
GLM-Image reference image encoder for img2img / multi-reference generation.

Replicates pipeline_glm_image.py lines 803-876 (write phase): for every
reference image,
    1. Run vision_language_encoder.get_image_features(pixel_values)
    2. Tokenize via get_image_tokens()
    3. Forward the transformer ONCE in mode="write" with kv_caches[i]
       to populate the reference KV cache.

The resulting `kv_caches` list is packed into the conditioning so the
sampler can pass them through to the transformer at every denoising step.
"""

# MANUAL bug-fix (Apr 2026): Native GLM-Image — reference encoder.

import torch


def encode_references(pipeline, ref_images: list, target_h: int, target_w: int):
    """ref_images: list of (3, H, W) float tensors in [0, 1] (max 5).
    Returns a list of GlmImageKVCache objects matching the diffusers API.
    """
    if not ref_images:
        return None

    transformer = pipeline.transformer
    vlm = pipeline.vision_language_encoder
    processor = pipeline.processor

    device = next(transformer.parameters()).device
    dtype = next(transformer.parameters()).dtype

    kv_caches = []
    for ref in ref_images[:5]:
        # ref: (3, H, W) -> (1, 3, H, W) in [0, 1]; processor expects PIL or [0,255]
        img = (ref.clamp(0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        from PIL import Image
        pil = Image.fromarray(img)

        inputs = processor(
            images=pil,
            return_tensors="pt",
        )
        # Accelerate-aware device handling
        has_hook = (
            getattr(vlm, "_hf_hook", None) is not None
            or any(getattr(m, "_hf_hook", None) is not None for m in vlm.modules())
        )
        in_device = torch.device("cpu") if has_hook else device

        pixel_values = inputs["pixel_values"].to(in_device)
        with torch.no_grad():
            image_features = vlm.get_image_features(pixel_values=pixel_values)
            image_tokens = vlm.get_image_tokens(image_features)

        # Build a fresh empty kv-cache, fill via transformer write-phase
        try:
            from diffusers.models.transformers.transformer_glm_image import GlmImageKVCache
        except Exception:
            from diffusers.models.transformers.transformer_glm_image import GlmImageKVCache  # type: ignore

        kv = GlmImageKVCache(transformer.config.num_layers)

        # Conservative target size for the reference forward
        ref_h = target_h
        ref_w = target_w
        target_size = torch.tensor([[ref_h, ref_w]], dtype=dtype, device=device)
        crop_coords = torch.zeros((1, 2), dtype=dtype, device=device)

        # The reference forward uses a tiny "write" pass — image_tokens act
        # as both prior_token_id and the hidden_states context.
        with torch.no_grad():
            # diffusers allocates an empty hidden_states sized to the
            # token grid at /16 (8 * patch_size=2). We emulate that.
            patch = transformer.config.patch_size
            hh = ref_h // (8 * patch)
            ww = ref_w // (8 * patch)
            hidden_states = torch.zeros((1, 16, hh * 2, ww * 2),
                                        dtype=dtype, device=device)
            transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=torch.zeros((1, 0, 1472), dtype=dtype, device=device),
                prior_token_id=image_tokens.to(device),
                prior_token_drop=torch.zeros_like(image_tokens, dtype=torch.bool, device=device),
                timestep=torch.zeros((1,), dtype=dtype, device=device),
                target_size=target_size,
                crop_coords=crop_coords,
                attention_mask=None,
                kv_caches=kv,
                return_dict=False,
            )
        kv_caches.append(kv)

    return kv_caches
