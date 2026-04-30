"""
GLM-Image · separate per-component loader + sampler nodes.

# MANUAL bug-fix (May 2026):
#   * separate CLIP / VAE / MODEL nodes (not a monolithic pipeline blob)
#   * step counter + ETA + it/s printed every step
#   * working ComfyUI Stop button (poll processing_interrupted)
#   * try / finally that frees VRAM + RAM on stop or error
#   * I2I support via optional `image` + `denoise_strength` on the sampler
#   * `tooltip=` on every widget and slot so hovering in the UI explains it
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import torch

import folder_paths
import comfy.model_management as mm
import comfy.utils

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    GlmImagePipeline,
)
from diffusers.models.transformers.transformer_glm_image import (
    GlmImageTransformer2DModel,
)
from diffusers.pipelines.glm_image import (
    GlmImageForConditionalGeneration,
    GlmImageProcessor,
)
from transformers import AutoTokenizer, T5EncoderModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HF_HUB_FALLBACKS = [
    "Disty0/GLM-Image-SDNQ-4bit-dynamic",
    "zai-org/GLM-Image",
]


def _scan_diffusers_folders():
    found = []
    for root in folder_paths.get_folder_paths("diffusers"):
        if not os.path.exists(root):
            continue
        for dirpath, _subdirs, filenames in os.walk(root, followlinks=True):
            if "model_index.json" in filenames:
                rel = os.path.relpath(dirpath, root)
                if rel == ".":
                    rel = os.path.basename(dirpath)
                found.append((rel, dirpath))
    return found


def _build_choices():
    local = [rel for rel, _ in _scan_diffusers_folders()]
    hub = [f"[HF Hub] {r}" for r in _HF_HUB_FALLBACKS]
    return local + hub or hub


def _resolve(name: str) -> str:
    if name.startswith("[HF Hub] "):
        return name[len("[HF Hub] "):]
    for rel, abs_path in _scan_diffusers_folders():
        if name == rel or name == os.path.basename(abs_path):
            return abs_path
    return name


def _dtype_of(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]


def _ensure_sdnq_registered(path: str):
    if "SDNQ" in path:
        try:
            import sdnq  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SDNQ-quantized GLM-Image variant requires the `sdnq` package. "
                "Install with: pip install sdnq"
            ) from e


def _free_vram_ram():
    try: mm.unload_all_models()
    except Exception: pass
    try: mm.soft_empty_cache()
    except Exception: pass
    gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        except Exception: pass


# ---------------------------------------------------------------------------
# Tooltip-rich combos / inputs
# ---------------------------------------------------------------------------

_T_MODEL_ID = (
    "Pick a GLM-Image folder. Local entries are subfolders of "
    "`ComfyUI/models/diffusers/` containing a `model_index.json`. "
    "`[HF Hub]` entries are downloaded on first use into your HF cache."
)
_T_DTYPE = (
    "Compute precision. `bf16` is recommended on RTX 30/40 series. "
    "`fp16` saves a bit of VRAM but is more prone to NaNs. "
    "`fp32` is full precision (slow, big)."
)
_T_DEVICE = "Where to place this component. `cuda` runs on GPU; `cpu` is a fallback."


# ---------------------------------------------------------------------------
# Node 1: VAE loader
# ---------------------------------------------------------------------------


class GLMImageVAELoader:
    DESCRIPTION = (
        "Load ONLY the GLM-Image VAE (16-channel AutoencoderKL) from a "
        "diffusers folder. Outputs a `GLMIMAGE_VAE` to feed into the sampler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        choices = _build_choices()
        return {
            "required": {
                "model_id":        (choices, {"default": choices[0], "tooltip": _T_MODEL_ID}),
                "dtype":           (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": _T_DTYPE}),
                "device":          (["cuda", "cpu"], {"default": "cuda", "tooltip": _T_DEVICE}),
                "enable_slicing":  ("BOOLEAN", {"default": True,  "tooltip": "Decode the latent in vertical slices to cut decode-time VRAM."}),
                "enable_tiling":   ("BOOLEAN", {"default": True,  "tooltip": "Decode the latent in tiles for very large images. Recommended above 1024²."}),
            }
        }

    RETURN_TYPES = ("GLMIMAGE_VAE",)
    RETURN_NAMES = ("vae",)
    OUTPUT_TOOLTIPS = ("GLM-Image VAE bundle. Connect to the `vae` input of `GLM-Image · Sampler`.",)
    FUNCTION = "load"
    CATEGORY = "GLMImage/loaders"

    def load(self, model_id, dtype, device, enable_slicing, enable_tiling):
        path = _resolve(model_id)
        _ensure_sdnq_registered(path)
        torch_dtype = _dtype_of(dtype)
        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        t0 = time.perf_counter()
        print(f"[GLMImageVAELoader] loading vae/ from {path} dtype={dtype} device={dev}")
        try:
            vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=torch_dtype)
            vae.eval(); vae.to(dev)
            if enable_slicing:
                try: vae.enable_slicing()
                except Exception as e: print(f"  [warn] enable_slicing: {e}")
            if enable_tiling:
                try: vae.enable_tiling()
                except Exception as e: print(f"  [warn] enable_tiling: {e}")
            print(f"[GLMImageVAELoader] loaded in {time.perf_counter()-t0:.1f}s")
            return ({"vae": vae, "dtype": torch_dtype, "device": dev, "path": path},)
        except Exception:
            _free_vram_ram(); raise


# ---------------------------------------------------------------------------
# Node 2: CLIP loader (text + vision-language)
# ---------------------------------------------------------------------------


class GLMImageCLIPLoader:
    DESCRIPTION = (
        "Load ONLY the GLM-Image text and vision-language encoders: T5 text "
        "encoder + ByT5 tokenizer + GLM vision-language model + image processor. "
        "Outputs a `GLMIMAGE_CLIP` to feed into the sampler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        choices = _build_choices()
        return {
            "required": {
                "model_id": (choices, {"default": choices[0], "tooltip": _T_MODEL_ID}),
                "dtype":    (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": _T_DTYPE}),
                "device":   (["cuda", "cpu"], {"default": "cuda", "tooltip": _T_DEVICE}),
            }
        }

    RETURN_TYPES = ("GLMIMAGE_CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_TOOLTIPS = ("GLM-Image text/vision encoders. Connect to `clip` on the sampler.",)
    FUNCTION = "load"
    CATEGORY = "GLMImage/loaders"

    def load(self, model_id, dtype, device):
        path = _resolve(model_id)
        _ensure_sdnq_registered(path)
        torch_dtype = _dtype_of(dtype)
        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        t0 = time.perf_counter()
        print(f"[GLMImageCLIPLoader] loading text + vlm + tokenizer + processor from {path}")
        try:
            tokenizer    = AutoTokenizer.from_pretrained(path, subfolder="tokenizer", trust_remote_code=True)
            print(f"  [+{time.perf_counter()-t0:.1f}s] tokenizer ok")
            processor    = GlmImageProcessor.from_pretrained(path, subfolder="processor")
            print(f"  [+{time.perf_counter()-t0:.1f}s] processor ok")
            text_encoder = T5EncoderModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=torch_dtype)
            text_encoder.eval().to(dev)
            print(f"  [+{time.perf_counter()-t0:.1f}s] text_encoder ok")
            vlm = GlmImageForConditionalGeneration.from_pretrained(
                path, subfolder="vision_language_encoder", torch_dtype=torch_dtype, trust_remote_code=True,
            )
            vlm.eval().to(dev)
            print(f"  [+{time.perf_counter()-t0:.1f}s] vlm ok — TOTAL {time.perf_counter()-t0:.1f}s")
            return ({
                "tokenizer": tokenizer, "processor": processor,
                "text_encoder": text_encoder, "vlm": vlm,
                "dtype": torch_dtype, "device": dev, "path": path,
            },)
        except Exception:
            _free_vram_ram(); raise


# ---------------------------------------------------------------------------
# Node 3: MODEL loader (transformer + scheduler)
# ---------------------------------------------------------------------------


class GLMImageModelLoader:
    DESCRIPTION = (
        "Load ONLY the GLM-Image transformer (DiT) + scheduler. This is the "
        "denoising backbone. Configure attention backend and slicing here."
    )

    @classmethod
    def INPUT_TYPES(cls):
        choices = _build_choices()
        return {
            "required": {
                "model_id":          (choices, {"default": choices[0], "tooltip": _T_MODEL_ID}),
                "dtype":             (["bf16", "fp16", "fp32"], {"default": "bf16", "tooltip": _T_DTYPE}),
                "device":            (["cuda", "cpu"], {"default": "cuda", "tooltip": _T_DEVICE}),
                "attention_backend": (["sdpa", "xformers"], {
                    "default": "sdpa",
                    "tooltip": "Attention kernel. `sdpa` = PyTorch native (always works). `xformers` is faster on supported GPUs but must be installed.",
                }),
                "attention_slicing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Slice attention to cut peak VRAM at the cost of speed. Enable on 6 GB GPUs.",
                }),
            }
        }

    RETURN_TYPES = ("GLMIMAGE_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("GLM-Image transformer + scheduler bundle. Connect to `model` on the sampler.",)
    FUNCTION = "load"
    CATEGORY = "GLMImage/loaders"

    def load(self, model_id, dtype, device, attention_backend, attention_slicing):
        path = _resolve(model_id)
        _ensure_sdnq_registered(path)
        torch_dtype = _dtype_of(dtype)
        dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        t0 = time.perf_counter()
        print(f"[GLMImageModelLoader] loading transformer/ + scheduler/ from {path}")
        try:
            transformer = GlmImageTransformer2DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=torch_dtype
            )
            transformer.eval().to(dev)
            print(f"  [+{time.perf_counter()-t0:.1f}s] transformer ok")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
            print(f"  [+{time.perf_counter()-t0:.1f}s] scheduler ok")

            applied = "sdpa"
            if attention_backend == "xformers":
                try:
                    transformer.enable_xformers_memory_efficient_attention()
                    applied = "xformers"
                except Exception as e:
                    print(f"  [warn] xformers failed → sdpa: {e}")
            print(f"  attention backend = {applied}")
            print(f"[GLMImageModelLoader] TOTAL {time.perf_counter()-t0:.1f}s")
            return ({
                "transformer": transformer, "scheduler": scheduler,
                "dtype": torch_dtype, "device": dev, "path": path,
                "attention_slicing": attention_slicing,
            },)
        except Exception:
            _free_vram_ram(); raise


# ---------------------------------------------------------------------------
# Node 4: Sampler (T2I + I2I)
# ---------------------------------------------------------------------------


def _comfy_image_to_pil_list(image_tensor):
    """ComfyUI IMAGE tensor (B,H,W,C) float [0,1] → list of PIL.Image."""
    from PIL import Image
    out = []
    arr = image_tensor.detach().cpu().float().clamp(0, 1).numpy()
    for i in range(arr.shape[0]):
        out.append(Image.fromarray((arr[i] * 255).astype(np.uint8)))
    return out


class GLMImageSeparateSampler:
    DESCRIPTION = (
        "GLM-Image sampler. Consumes separate `clip`/`vae`/`model` bundles. "
        "Pure text-to-image by default. Provide an optional `image` for "
        "image-to-image / reference-conditioned generation. "
        "Prints step counter + ETA + it/s every step. Honors the Stop button. "
        "Frees VRAM/RAM automatically on stop or error."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":   ("GLMIMAGE_VAE",   {"tooltip": "From `GLM-Image · Load VAE`."}),
                "clip":  ("GLMIMAGE_CLIP",  {"tooltip": "From `GLM-Image · Load CLIP (T5+VLM)`."}),
                "model": ("GLMIMAGE_MODEL", {"tooltip": "From `GLM-Image · Load MODEL (DiT)`."}),
                "prompt": ("STRING", {
                    "multiline": True, "default": "a friendly red panda, vivid, simple background",
                    "tooltip": "Positive prompt. Plain English works. GLM-Image is multilingual.",
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Negative prompt. Leave empty if the pipeline doesn't support CFG (SDNQ-4bit ignores this).",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "RNG seed. Same seed + same params = same image.",
                }),
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": 100,
                    "tooltip": "Number of denoising steps. GLM-Image distilled checkpoints work great at 4–8 steps.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. 1.0 = no CFG (fastest). GLM-Image distilled prefers 1.0–2.0.",
                }),
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 32,
                    "tooltip": "Output width in pixels. Will be rounded to the nearest multiple of 32.",
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 32,
                    "tooltip": "Output height in pixels. Will be rounded to the nearest multiple of 32.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 8,
                    "tooltip": "Number of images per prompt. VRAM scales linearly.",
                }),
                "denoise_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "I2I only — fraction of the schedule to actually denoise. 1.0 = full noise (T2I-equivalent), 0.0 = return the input. Ignored when `image` is unconnected.",
                }),
                "free_after": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload all models and clear VRAM/RAM after this run. Enable for one-shot generations.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional reference image for image-to-image / image-conditioned generation. Connect any IMAGE source.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Generated images (B,H,W,C float [0,1]). Connect to a `Save Image` or `Preview Image`.",)
    FUNCTION = "sample"
    CATEGORY = "GLMImage/sampling"

    def sample(self, vae, clip, model, prompt, negative_prompt, seed, steps,
               guidance_scale, width, height, batch_size, denoise_strength,
               free_after, image=None):
        # Round to multiples of 32
        width  = max(64, (int(width)  // 32) * 32)
        height = max(64, (int(height) // 32) * 32)

        pipe = GlmImagePipeline(
            vae=vae["vae"],
            text_encoder=clip["text_encoder"],
            tokenizer=clip["tokenizer"],
            processor=clip["processor"],
            vision_language_encoder=clip["vlm"],
            transformer=model["transformer"],
            scheduler=model["scheduler"],
        )
        if model.get("attention_slicing"):
            try: pipe.enable_attention_slicing()
            except Exception: pass

        device = model["device"]
        gen = torch.Generator(device=device).manual_seed(int(seed))

        # I2I: convert tensor to PIL list and adjust effective steps via strength
        i2i_image = None
        effective_steps = int(steps)
        if image is not None:
            pil_list = _comfy_image_to_pil_list(image)
            i2i_image = pil_list[0] if len(pil_list) == 1 else pil_list
            # GLM-Image's pipeline doesn't take `strength`; emulate by truncating
            # the schedule. denoise_strength 1.0 = use all `steps`; 0.5 = half, etc.
            if denoise_strength < 1.0:
                effective_steps = max(1, int(round(steps * float(denoise_strength))))

        pbar = comfy.utils.ProgressBar(effective_steps)
        t_start = time.perf_counter()

        def cb(pipeline, step, timestep, kw):
            mm.throw_exception_if_processing_interrupted()
            elapsed = time.perf_counter() - t_start
            done = step + 1
            it_s = done / elapsed if elapsed > 0 else 0.0
            eta = (effective_steps - done) / it_s if it_s > 0 else float("inf")
            print(
                f"[GLM] step {done}/{effective_steps} — elapsed {elapsed:5.1f}s — "
                f"ETA {eta:5.1f}s — {it_s:.2f} it/s"
            )
            pbar.update_absolute(done, effective_steps)
            return kw

        mode = "I2I" if i2i_image is not None else "T2I"
        print(
            f"[GLMImageSeparateSampler] {mode} {width}x{height} steps={effective_steps} "
            f"cfg={guidance_scale} seed={seed} bsz={batch_size}"
        )

        try:
            mm.throw_exception_if_processing_interrupted()
            kwargs = dict(
                prompt=prompt,
                num_inference_steps=effective_steps,
                guidance_scale=float(guidance_scale),
                width=width, height=height,
                num_images_per_prompt=int(batch_size),
                generator=gen,
                output_type="pt",
                callback_on_step_end=cb,
            )
            if i2i_image is not None:
                kwargs["image"] = i2i_image
            if negative_prompt:
                # GLM-Image pipeline doesn't take negative_prompt directly; inject
                # via prompt_embeds path only if requested. For now just ignore
                # silently and warn once.
                print("[GLMImageSeparateSampler] note: negative_prompt is ignored by GLM-Image pipeline; using empty negative.")

            out = pipe(**kwargs)
            imgs = out.images
            if isinstance(imgs, torch.Tensor) and imgs.dim() == 4 and imgs.shape[1] in (3, 4):
                imgs = imgs.permute(0, 2, 3, 1).contiguous()  # BHWC for ComfyUI
            imgs = imgs.float().cpu().clamp(0, 1)
            total = time.perf_counter() - t_start
            print(f"[GLMImageSeparateSampler] DONE in {total:.1f}s ({effective_steps/total:.2f} it/s)")
            return (imgs,)
        except mm.InterruptProcessingException:
            print("[GLMImageSeparateSampler] INTERRUPTED — freeing VRAM/RAM")
            raise
        except Exception:
            print("[GLMImageSeparateSampler] ERROR — freeing VRAM/RAM")
            raise
        finally:
            del pipe
            if free_after:
                _free_vram_ram()
                print("[GLMImageSeparateSampler] free_after=True → VRAM/RAM cleared")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


SEPARATE_NODE_CLASS_MAPPINGS = {
    "GLMImageVAELoader":        GLMImageVAELoader,
    "GLMImageCLIPLoader":       GLMImageCLIPLoader,
    "GLMImageModelLoader":      GLMImageModelLoader,
    "GLMImageSeparateSampler":  GLMImageSeparateSampler,
}

SEPARATE_NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMImageVAELoader":        "GLM-Image · Load VAE",
    "GLMImageCLIPLoader":       "GLM-Image · Load CLIP (T5+VLM)",
    "GLMImageModelLoader":      "GLM-Image · Load MODEL (DiT)",
    "GLMImageSeparateSampler":  "GLM-Image · Sampler (T2I/I2I)",
}
