import torch
import numpy as np
import comfy.model_management
import comfy.utils
from .utils import pil2tensor, get_closest_multiple_of_32

class GLMImageSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("GLMIMAGE_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "A futuristic city with flying cars"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 32}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "GLMImage"

    def generate(self, pipeline, prompt, seed, steps, guidance_scale, width, height, batch_size, negative_prompt=""):
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        width = get_closest_multiple_of_32(width)
        height = get_closest_multiple_of_32(height)

        pbar = comfy.utils.ProgressBar(steps)
        
        # Callback to update pbar
        def callback(step, timestep, latents):
            pbar.update(1)

        print(f"Generating GLM-Image: {width}x{height}, steps={steps}")
        
        try:
            images = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=batch_size,
                generator=generator,
                output_type="pil",
                callback=callback, # Note: Verify specific pipeline callback support, generic diffusers usually supports it.
            ).images
        except TypeError:
            # Fallback if callback signature doesn't match custom pipeline
            images = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=batch_size,
                generator=generator,
                output_type="pil"
            ).images

        # Convert to ComfyUI format
        image_tensors = []
        for img in images:
            image_tensors.append(pil2tensor(img))
            
        return (torch.cat(image_tensors, dim=0),)

class GLMImageImg2Img:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("GLMIMAGE_PIPELINE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "strength": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_img2img"
    CATEGORY = "GLMImage"

    def generate_img2img(self, pipeline, image, prompt, strength, seed, steps, guidance_scale, negative_prompt=""):
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        
        # Prepare input images
        input_images_pil = []
        for i in range(image.shape[0]):
             # Convert tensor [H, W, 3] -> PIL
             img_np = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
             input_images_pil.append(Image.fromarray(img_np))

        pbar = comfy.utils.ProgressBar(steps)
        
        batch_results = []
        
        # Process batch (Pipelines usually handle batching, but input image batching can be tricky if they differ)
        # We will iterate for safety unless pipeline explicitly supports list of images for batch.
        for pil_img in input_images_pil:
             res = pipeline(
                 prompt=prompt,
                 image=pil_img,
                 negative_prompt=negative_prompt if negative_prompt else None,
                 strength=strength,
                 num_inference_steps=steps,
                 guidance_scale=guidance_scale,
                 generator=generator,
                 output_type="pil"
             ).images[0]
             batch_results.append(pil2tensor(res))
             pbar.update(steps // len(input_images_pil)) 

        return (torch.cat(batch_results, dim=0),)
