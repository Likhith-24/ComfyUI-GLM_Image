# MANUAL bug-fix (Apr 2026): native ComfyUI integration for GLM-Image.
# This subpackage decomposes a loaded GlmImagePipeline into MODEL/CLIP/VAE
# objects compatible with the stock KSampler / SamplerCustomAdvanced /
# CLIPTextEncode / VAEDecode / VAEEncode nodes.
#
# Public surface:
#   - GLMImageVAEWrapper  (vae.py)   — Phase 1
#   - GLMImageCLIPWrapper (clip.py)  — Phase 2
#   - build_glm_model     (model.py) — Phase 3
