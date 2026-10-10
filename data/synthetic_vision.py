import random
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict

class SyntheticVisionGenerator:
    """
    Generates synthetic multi-modal pairs:
    - 64x64 RGB image tensor (C, H, W) normalized to [0, 1]
    - Paired question and chain-of-thought answer text
    Tasks:
    - Counting shapes (1 to 4)
    - Color identification (red, green, blue, yellow)
    - Shape classification (circle, rectangle)
    """
    COLORS = {
        "red": (230, 40, 40),
        "green": (40, 200, 40),
        "blue": (40, 80, 240),
        "yellow": (240, 220, 30),
    }

    def __init__(self, img_size: int = 64, seed: int = 42):
        self.img_size = img_size
        self.rng = random.Random(seed)

    def generate_sample(self) -> Tuple[torch.Tensor, str]:
        task_type = self.rng.choice(["count", "color", "shape"])
        img = Image.new("RGB", (self.img_size, self.img_size), color=(20, 20, 25)) # dark background
        draw = ImageDraw.Draw(img)
        
        if task_type == "count":
            count = self.rng.randint(1, 4)
            color_name = self.rng.choice(list(self.COLORS.keys()))
            color_val = self.COLORS[color_name]
            
            # Draw non-overlapping circles
            r = 6
            positions = []
            grid = [(16, 16), (48, 16), (16, 48), (48, 48)]
            self.rng.shuffle(grid)
            for i in range(count):
                cx, cy = grid[i]
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color_val)
                
            prompt = f"<img_start><img_end><user_start>Count the {color_name} circles.<user_end><assistant_start><thought_start>Scanning visual tokens: found {count} {color_name} circles.<thought_end>{count}<assistant_end>"
            
        elif task_type == "color":
            shape_type = self.rng.choice(["circle", "square"])
            color_name = self.rng.choice(list(self.COLORS.keys()))
            color_val = self.COLORS[color_name]
            cx, cy = self.img_size // 2, self.img_size // 2
            r = 14
            
            if shape_type == "circle":
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color_val)
            else:
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color_val)
                
            prompt = f"<img_start><img_end><user_start>What color is the {shape_type}?<user_end><assistant_start><thought_start>Inspecting center patch: chromatic intensity indicates {color_name}.<thought_end>{color_name}<assistant_end>"
            
        else: # shape
            shape_type = self.rng.choice(["circle", "square"])
            color_name = self.rng.choice(list(self.COLORS.keys()))
            color_val = self.COLORS[color_name]
            cx, cy = self.img_size // 2, self.img_size // 2
            r = 14
            
            if shape_type == "circle":
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color_val)
            else:
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color_val)
                
            prompt = f"<img_start><img_end><user_start>What shape is in the image?<user_end><assistant_start><thought_start>Feature extractor reveals geometry: {shape_type}.<thought_end>{shape_type}<assistant_end>"

        # Convert image to float tensor (3, H, W) in [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # (C, H, W)
        return img_tensor, prompt

    def generate_batch(self, batch_size: int = 8) -> Tuple[torch.Tensor, List[str]]:
        images = []
        prompts = []
        for _ in range(batch_size):
            img_t, prompt = self.generate_sample()
            images.append(img_t)
            prompts.append(prompt)
        images_tensor = torch.stack(images, dim=0) # (B, 3, H, W)
        return images_tensor, prompts
