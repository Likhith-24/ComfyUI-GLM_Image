import torch
import numpy as np
from PIL import Image

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def batch_tensor_to_pil(img_tensor):
    return [tensor2pil(img_tensor[i]) for i in range(img_tensor.shape[0])]

def get_closest_multiple_of_32(val):
    return int((val // 32) * 32)
