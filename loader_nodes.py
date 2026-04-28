import torch
import os
import folder_paths
from diffusers import GlmImagePipeline

class GLMImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device_map": (["balanced", "sequential", "cuda", "cpu", "auto"], {"default": "balanced"}),
            }
        }

    RETURN_TYPES = ("GLMIMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "GLMImage"
    DESCRIPTION = "Loads the GLM-Image model from HuggingFace. First run will download ~23GB."

    def load_model(self, dtype, device_map):
        model_id = "zai-org/GLM-Image"
        
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        print(f"Loading GLM-Image model from {model_id}...")
        
        # Uses standard Diffusers caching mechanism (usually ~/.cache/huggingface/hub)
        # To customize download path, users can set HF_HOME environment variable.
        pipeline = GlmImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map if device_map != "auto" else None
        )
        
        return (pipeline,)
