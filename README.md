# ComfyUI-GLMImage

A ComfyUI custom node package for integrating the **GLM-Image** model (by Zhipu AI) using Hugging Face Diffusers. This package allows you to generate high-fidelity images using GLM-Image's advanced text rendering and instruction following capabilities directly within ComfyUI.

## Features

- **Native Diffusers Integration**: Uses the official `GlmImagePipeline`.
- **Memory Efficient**: Supports `bfloat16` and device offloading (CPU/CUDA/Balanced).
- **Core Nodes**:
    - **GLMImage Loader**: Loads the model efficiently.
    - **GLMImage Sampler**: Text-to-Image generation with support for batch size, guidance scale, and steps.
    - **GLMImage Img2Img**: Image-to-Image generation (experimental/dependent on pipeline support).
- **Clean Architecture**: Follows best practices for ComfyUI custom nodes.

## Installation

### Prerequisites
- **VRAM**: ~23GB Recommended (for bf16 loading). Lower VRAM might work with aggressive offloading (CPU offload), but performance will vary.
- **ComfyUI**: Latest version.

### Steps

1.  **Clone the Repository**
    Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes
    git clone https://github.com/your-repo/ComfyUI-GLMImage.git
    cd ComfyUI-GLMImage
    ```

2.  **Install Dependencies**
    It is crucial to have `diffusers` and `transformers` installed from source or latest versions to support GLM-Image correctly.
    ```bash
    pip install -r requirements.txt
    ```
    *If you are using the ComfyUI embedded python:*
    ```bash
    ../../../python_embeded/python.exe -m pip install -r requirements.txt
    ```

## Usage

### 1. GLMImage Loader
- **dtype**: Select `bf16` (bfloat16) for best performance/VRAM balance. `fp16` is also supported.
- **device_map**:
    - `balanced`: Splits model across GPUs/CPU if needed.
    - `sequential`: Slower but safer for memory.
    - `cuda`: Forces all to GPU (Requires huge VRAM).
    - `auto`: Accelerate decides.

### 2. GLMImage Sampler (Text-to-Image)
Connect the `pipeline` output from the loader to this node.
- **Prompt**: Your text description.
- **Negative Prompt**: Optional.
- **Steps**: 50 is a good baseline.
- **Guidance Scale**: Recommended low (~1.5 - 5.0) for GLM-Image.
- **Width/Height**: Must be multiples of 32.

### 3. GLMImage Img2Img
Connect an input image and the pipeline.
- **Strength**: 0.0 to 1.0 (Higher = more change).

## Example Workflows

### Text-to-Image Basic
```json
{
  "3": {
    "inputs": {
      "seed": 156680208700286,
      "steps": 50,
      "cfg": 8.0,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": [
        "4",
        0
      ],
      "positive": [
        "6",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "latent_image": [
        "5",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "10": {
    "inputs": {
      "dtype": "bf16",
      "device_map": "balanced"
    },
    "class_type": "GLMImageLoader",
    "_meta": {
      "title": "GLMImage Loader"
    }
  },
  "11": {
    "inputs": {
      "prompt": "A cinematic shot of a cyberpunk city, neon lights, rain, high detail",
      "negative_prompt": "blurry, low quality",
      "seed": 0,
      "steps": 50,
      "guidance_scale": 1.5,
      "width": 1024,
      "height": 1024,
      "batch_size": 1,
      "pipeline": [
        "10",
        0
      ]
    },
    "class_type": "GLMImageSampler",
    "_meta": {
      "title": "GLMImage Text-to-Image Sampler"
    }
  },
  "12": {
    "inputs": {
      "images": [
        "11",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  }
}
```
*Note: The KSampler node above is just standard context, for GLMImage you purely use the GLMImage nodes (10, 11, 12).*

## Tips
- **Resolution**: GLM-Image works well at 1024x1024.
- **Precision**: Always try `bf16` first if your card supports it (RTX 30 series and up).

## Credits
Based on the GLM-Image model by Zhipu AI and Hugging Face Diffusers library.
Developed following Kijai's excellent custom node templates.
