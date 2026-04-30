"""
GLM-Image CLIP wrapper for native ComfyUI integration.

This is NOT a real `comfy.sd.CLIP` — it's a duck-typed object that
implements the same `tokenize()` / `encode_from_tokens()` API so that
stock `CLIPTextEncode` works.

When the user writes `CLIPTextEncode("a friendly dog")`, ComfyUI calls:
    cond_tensor, extras = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True, return_dict=True)
We pack ALL the GLM-specific conditioning extras (prior_token_ids,
target_size, crop_coords, attention_mask, prior_token_drop) into the
`extras` dict so the MODEL wrapper can unpack them in `extra_conds()`.

Truth source for encoding: pipeline_glm_image.py lines 776-910.
"""

# MANUAL bug-fix (Apr 2026): Phase 2 of native GLM-Image integration.

import re
import torch
import torch.nn as nn


# Default target size used when CLIPTextEncode is called BEFORE we know the
# final latent size. The user can override per-conditioning via the
# `GLMImageSetTargetSize` node (Phase 5) or it gets auto-corrected if the
# `EmptyGLMImageLatent` records `glm_target_size` — but the latent dict
# isn't visible from CLIPTextEncode, so we default to 1024×1024 (matches
# diffusers default) and recommend matching `EmptyGLMImageLatent`.
_DEFAULT_TARGET_H = 1024
_DEFAULT_TARGET_W = 1024


class GLMImageCLIPWrapper:
    """Single-CLIP design: runs both T5 (encoder_hidden_states) and the
    AR vision-language model (prior_token_ids) inside encode_from_tokens.
    Auto-detects empty prompt for uncond branch and sets prior_token_drop=True."""

    def __init__(self, pipeline, target_h=_DEFAULT_TARGET_H, target_w=_DEFAULT_TARGET_W):
        self.pipeline = pipeline
        self.text_encoder = pipeline.text_encoder           # T5
        self.tokenizer = pipeline.tokenizer                 # ByT5Tokenizer
        self.processor = pipeline.processor                 # GlmImageProcessor
        self.vlm = pipeline.vision_language_encoder         # GlmImageForConditionalGeneration
        self.transformer_config = pipeline.transformer.config
        self.target_h = target_h
        self.target_w = target_w

        # ComfyUI plumbing — KSampler peeks at .cond_stage_model and .patcher
        # for memory management. We expose minimal stubs.
        self.cond_stage_model = None  # not a comfy.sd.CLIP — duck-type only
        self.patcher = None
        self.layer_idx = None
        self.use_clip_schedule = False

        # Track current latent size for target_size override (settable from
        # KSampler-side latent dict if present — else defaults).
        self._current_target = None

    # ---- API expected by CLIPTextEncode --------------------------------

    def clone(self):
        new = GLMImageCLIPWrapper.__new__(GLMImageCLIPWrapper)
        new.__dict__.update(self.__dict__)
        return new

    def tokenize(self, text, return_word_ids=False, **kwargs):
        """Return a dict that round-trips into encode_from_tokens. We stash
        the raw text so the encoder can decide T5 vs AR routing."""
        return {"glm_text": text}

    def encode_from_tokens_scheduled(self, tokens, unprojected=False,
                                     add_dict={}, show_pbar=True):
        """ComfyUI 1.x calls this. Wrap encode_from_tokens result into the
        scheduled-cond format expected by KSampler:
            [(cond_tensor, {extras_dict})]
        """
        cond, extras = self.encode_from_tokens(tokens, return_pooled=True,
                                               return_dict=True)
        # encode_from_tokens already returns a dict with "pooled_output";
        # CLIPTextEncode expects {"pooled_output": ...} merged into extras.
        out = {**extras, **add_dict}
        return [(cond, out)]

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        text = tokens.get("glm_text", "") if isinstance(tokens, dict) else str(tokens)
        is_empty = (text is None) or (str(text).strip() == "")

        device = self._device()
        dtype = self._dtype()

        # ---- T5 / ByT5: extract glyph text & encode -------------------
        glyph_embeds, attention_mask = self._encode_t5(text, device, dtype)

        # ---- AR vision-language model: produce prior_token_ids --------
        prior_token_ids = self._generate_prior_tokens(text)
        # Match T5 device for downstream packing
        prior_token_ids = prior_token_ids.to(device)

        # ---- prior_token_drop: True for uncond/empty, else False ------
        prior_token_drop = torch.full_like(prior_token_ids, is_empty, dtype=torch.bool)

        # ---- Pack target/crop --------------------------------------------
        target_h, target_w = self._current_target or (self.target_h, self.target_w)
        target_size = torch.tensor([[target_h, target_w]], dtype=dtype, device=device)
        crop_coords = torch.zeros((1, 2), dtype=dtype, device=device)

        cond = glyph_embeds  # [1, T, 1472]

        if return_dict:
            extras = {
                "pooled_output": None,
                "attention_mask": attention_mask,
                "glm_prior_ids": prior_token_ids,
                "glm_prior_drop": prior_token_drop,
                "glm_target_size": target_size,
                "glm_crop_coords": crop_coords,
                "glm_is_empty": is_empty,
            }
            return cond, extras
        if return_pooled:
            return cond, None
        return cond

    # ---- Internals -----------------------------------------------------

    def _device(self):
        return self._execution_device()

    def _execution_device(self):
        """Best-effort resolution of the diffusers pipeline execution device.
        Works under enable_model_cpu_offload (cpu_offload_with_hook).
        """
        # 1) Diffusers caches it
        dev = getattr(self.pipeline, "_execution_device", None)
        if dev is not None:
            return dev
        # 2) Hook may expose it
        hook = getattr(self.vlm, "_hf_hook", None)
        if hook is not None and getattr(hook, "execution_device", None) is not None:
            return hook.execution_device
        for m in self.vlm.modules():
            h = getattr(m, "_hf_hook", None)
            if h is not None and getattr(h, "execution_device", None) is not None:
                return h.execution_device
        # 3) Fall back to first parameter device, else cuda if available
        try:
            return next(self.vlm.parameters()).device
        except StopIteration:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _dtype(self):
        try:
            return next(self.text_encoder.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _encode_t5(self, text, device, dtype):
        """Mirrors pipeline_glm_image.py lines 893-910."""
        # Same SDNQ+cpu_offload caveat as in _generate_prior_tokens: force
        # the encoder fully onto the execution device first.
        try:
            self.text_encoder.to(device)
        except Exception:
            pass
        # Extract quoted glyph text. If empty prompt or no quotes, supply
        # a single empty glyph so the T5 path still runs (returns near-zero).
        if not text:
            ocr_texts = [""]
        else:
            ocr_texts = re.findall(r"'([^']*)'", text)
            if not ocr_texts:
                # No quoted glyphs — pipeline-style fallback uses the full text
                ocr_texts = [text]

        enc = self.tokenizer(
            ocr_texts,
            max_length=2048,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = self.text_encoder(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
            )
        last = out.last_hidden_state.to(dtype)  # [n_glyph, seq, 1472]
        # Concatenate glyphs along seq dim, masked to valid tokens
        mask = enc.attention_mask.bool()
        flat = last[mask].unsqueeze(0)  # [1, total_valid, 1472]
        attn_mask = torch.ones((1, flat.shape[1]), dtype=torch.long, device=device)
        if flat.shape[1] == 0:
            # Provide minimum 1-token sequence to avoid empty-seq bugs
            flat = torch.zeros((1, 1, last.shape[-1]), dtype=dtype, device=device)
            attn_mask = torch.zeros((1, 1), dtype=torch.long, device=device)
        return flat, attn_mask

    def _generate_prior_tokens(self, text):
        """Mirrors pipeline_glm_image.py lines 803-876 (txt2img branch)."""
        target_h, target_w = self._current_target or (self.target_h, self.target_w)
        # Resolve input device. Diffusers' enable_model_cpu_offload uses
        # cpu_offload_with_hook which moves WEIGHTS to execution device on
        # pre-forward but does NOT auto-move inputs. Additionally, with
        # SDNQ-quantized weights pre-pinned on CUDA, the hook's parameter
        # check returns cuda and SKIPS moving CPU-resident submodules
        # (e.g. nn.Embedding). So we explicitly move the VLM to the
        # execution device first; this is a no-op for already-resident
        # tensors but properly migrates the CPU-resident embedding.
        vlm_device = self._execution_device()
        try:
            self.vlm.to(vlm_device)
        except Exception as e:
            print(f"[GLM-Image] vlm.to({vlm_device}) raised {type(e).__name__}: {e}")
        # GlmImageProcessor.apply_chat_template — diffusers internal API
        messages = [
            {"role": "user",
             "content": [{"type": "text", "text": text or ""}]}
        ]
        try:
            inputs = self.processor.apply_chat_template(
                [messages],
                tokenize=True,
                target_h=target_h,
                target_w=target_w,
                return_dict=True,
                return_tensors="pt",
            ).to(vlm_device)
        except Exception:
            # Some processors don't accept the list-wrap; try unwrapped
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                target_h=target_h,
                target_w=target_w,
                return_dict=True,
                return_tensors="pt",
            ).to(vlm_device)

        # token_h * token_w: how many image tokens to autoregressively generate
        patch = self.transformer_config.patch_size
        token_h = target_h // (8 * patch)
        token_w = target_w // (8 * patch)
        n_image_tokens = token_h * token_w

        with torch.no_grad():
            input_len = inputs["input_ids"].shape[1]
            outputs = self.vlm.generate(
                **inputs,
                max_new_tokens=n_image_tokens + 1,
                do_sample=True,
            )
        # Slice the newly-generated image tokens
        prior_token_ids = outputs[:, input_len:input_len + n_image_tokens]
        # Some generations stop early; pad to expected length with codebook idx 0
        if prior_token_ids.shape[1] < n_image_tokens:
            pad = torch.zeros(
                (prior_token_ids.shape[0], n_image_tokens - prior_token_ids.shape[1]),
                dtype=prior_token_ids.dtype, device=prior_token_ids.device,
            )
            prior_token_ids = torch.cat([prior_token_ids, pad], dim=1)
        return prior_token_ids
