from .loader_nodes import GLMImageLoader
from .sampler_nodes import GLMImageSampler, GLMImageImg2Img

NODE_CLASS_MAPPINGS = {
    "GLMImageLoader": GLMImageLoader,
    "GLMImageSampler": GLMImageSampler,
    "GLMImageImg2Img": GLMImageImg2Img
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMImageLoader": "GLMImage Loader",
    "GLMImageSampler": "GLMImage Text-to-Image Sampler",
    "GLMImageImg2Img": "GLMImage Image-to-Image Sampler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
