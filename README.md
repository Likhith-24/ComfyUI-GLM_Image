# ComfyUI-GLM_Image

ComfyUI custom nodes for **GLM-Image** (Zhipu AI / `zai-org/GLM-Image` and SDNQ-quantized variants such as `Disty0/GLM-Image-SDNQ-4bit-dynamic`) via Hugging Face `diffusers`.

## What this pack does

GLM-Image is a multilingual flow-matching DiT image generator that ships as a multi-component diffusers checkpoint (`vae/`, `text_encoder/`, `vision_language_encoder/`, `transformer/`, `scheduler/`, `tokenizer/`, `processor/`). Loading the whole pipeline as a single blob makes it impossible to swap components, hard to free VRAM, and slow to start.

This pack solves that by exposing **four nodes**: three independent loaders (VAE, CLIP, MODEL) and one sampler that consumes them. Each loader peaks VRAM only for its own component, then releases. The sampler:

- Prints a per-step counter, ETA, and `it/s` to the console.
- Honors the ComfyUI Stop button via `comfy.model_management.throw_exception_if_processing_interrupted()`.
- Frees VRAM and RAM in a `try/finally` on stop or error.
- Supports both text-to-image and image-to-image (via optional `image` + `denoise_strength`).

Models are read from `ComfyUI/models/diffusers/<repo-name>/` (any folder containing `model_index.json` is auto-detected). HF Hub IDs are listed as fallbacks; selecting one downloads on first use into your HF cache.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo> ComfyUI-GLM_Image
pip install -r ComfyUI-GLM_Image/requirements.txt
```

Embedded-Python users (ComfyUI portable):

```bash
..\..\python_embeded\python.exe -m pip install -r ComfyUI-GLM_Image\requirements.txt
```

`requirements.txt` pulls `transformers` and `diffusers` from git so the GLM-Image pipeline classes are available. SDNQ-quantized checkpoints additionally require `pip install sdnq`.

Place the diffusers folder at:

```
ComfyUI/models/diffusers/GLM-Image/
ComfyUI/models/diffusers/GLM-Image-SDNQ-4bit-dynamic/
```

Restart ComfyUI. Nodes appear under `GLMImage/loaders` and `GLMImage/sampling`.

## Nodes

### GLM-Image · Load VAE (`GLMImageVAELoader`)

Loads only the 16-channel `AutoencoderKL` from the chosen diffusers folder.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model_id` | combo | first scanned folder | Local diffusers folders + `[HF Hub]` fallbacks |
| `dtype` | combo | `bf16` | `bf16` / `fp16` / `fp32` |
| `device` | combo | `cuda` | `cuda` / `cpu` |
| `enable_slicing` | bool | `True` | Slice decode to cut VRAM |
| `enable_tiling` | bool | `True` | Tile decode for >1024² output |

**Output:** `vae` (`GLMIMAGE_VAE`).

**Use case:** Stage 1 of any GLM-Image graph. Load → freeze → connect to the sampler. Disable slicing/tiling only if you need maximum decode speed and have spare VRAM.

### GLM-Image · Load CLIP (T5+VLM) (`GLMImageCLIPLoader`)

Loads the T5 text encoder, ByT5 tokenizer, GLM vision-language model, and its image processor.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model_id` | combo | first scanned folder | Same source as VAE loader |
| `dtype` | combo | `bf16` | — |
| `device` | combo | `cuda` | — |

**Output:** `clip` (`GLMIMAGE_CLIP`).

**Use case:** Stage 2. Holds all text/vision conditioning components in one bundle so the sampler doesn't need four separate inputs.

### GLM-Image · Load MODEL (DiT) (`GLMImageModelLoader`)

Loads the `GlmImageTransformer2DModel` (DiT denoiser) and the `FlowMatchEulerDiscreteScheduler`.

| Input | Type | Default | Notes |
|---|---|---|---|
| `model_id` | combo | first scanned folder | — |
| `dtype` | combo | `bf16` | — |
| `device` | combo | `cuda` | — |
| `attention_backend` | combo | `sdpa` | `sdpa` (always works) or `xformers` |
| `attention_slicing` | bool | `False` | Enable on 6 GB GPUs |

**Output:** `model` (`GLMIMAGE_MODEL`).

**Use case:** Stage 3. The denoising backbone — typically the largest VRAM consumer, which is exactly why it lives in its own node.

### GLM-Image · Sampler (`GLMImageSeparateSampler`)

Consumes the three bundles and runs sampling.

| Input | Type | Default | Notes |
|---|---|---|---|
| `vae` / `clip` / `model` | bundles | — | From the three loaders |
| `prompt` | STRING (multiline) | demo prompt | Multilingual, plain English fine |
| `negative_prompt` | STRING (multiline) | `""` | Ignored by SDNQ-4bit checkpoints |
| `seed` | INT | `42` | — |
| `steps` | INT | `4` | Distilled checkpoints work at 4–8 |
| `guidance_scale` | FLOAT | `1.5` | `1.0` = no CFG; distilled prefers 1.0–2.0 |
| `width` / `height` | INT | `512` | Rounded to nearest multiple of 32 |
| `batch_size` | INT | `1` | Linear VRAM cost |
| `denoise_strength` | FLOAT | `1.0` | I2I only; truncates schedule (0.0 = return input) |
| `free_after` | bool | `False` | Unload models + clear caches after run |
| `image` *(optional)* | IMAGE | — | Connect to enable image-to-image |

**Output:** `images` (IMAGE, BHWC float `[0, 1]`).

**Use case:** T2I by leaving `image` unconnected; I2I by feeding any IMAGE source and tuning `denoise_strength`. The console prints `[GLM] step X/Y — elapsed Zs — ETA Ws — it/s R` every step.

## Use in image/video generation pipelines (Flux / Qwen-Image / Wan / Z-Image / ERNIE-VL)

This pack is purpose-built for **GLM-Image only**. The four custom types (`GLMIMAGE_VAE`, `GLMIMAGE_CLIP`, `GLMIMAGE_MODEL`, plus the sampler's IMAGE output) are not interchangeable with native ComfyUI `MODEL`/`CLIP`/`VAE` types.

| Model family | Applicability | Notes |
|---|---|---|
| **GLM-Image** | Native | Use these nodes directly for T2I and I2I. |
| **Flux** | Indirect | The IMAGE output of the GLM sampler can be fed into a Flux Img2Img graph (encode with a Flux VAE, sample with a Flux KSampler). The GLM bundles do not connect to Flux's `MODEL`/`CLIP`/`VAE`. |
| **Qwen-Image** | Indirect | Same as Flux: use GLM-Image as a generator stage, then re-encode the IMAGE for a Qwen-Image refinement pass. |
| **Wan 2.x (video)** | Indirect | Use GLM-Image to generate a stylized first frame or reference image, then drive Wan animation from that IMAGE. |
| **Z-Image** | Indirect | Same pattern: generate with GLM, refine/restyle with a Z-Image graph. |
| **ERNIE-VL** | Not applicable | ERNIE-VL is a multimodal LLM, not a diffusion image generator. No integration here. |

For cross-pack chaining, the load order in the graph follows the user-mandated convention: **CLIP → VAE → MODEL → Sampler**, sequentially.

## Example wiring

```
[GLM-Image · Load CLIP (T5+VLM)] ─┐
[GLM-Image · Load VAE]           ─┼──> [GLM-Image · Sampler] ──> [Save Image]
[GLM-Image · Load MODEL (DiT)]   ─┘                ^
                                                   │ (optional)
                                            [Load Image]
```

## Notes

- A legacy monolithic loader was removed in favor of the four-node split.
- SDNQ-4bit variants ignore `negative_prompt`; the sampler logs a one-time note.
- On stop or error, the sampler unloads all models, calls `mm.soft_empty_cache()`, runs `gc.collect()`, and clears the CUDA cache.

## License

Apache-2.0 (see `LICENSE` if present, otherwise see repository for license).
