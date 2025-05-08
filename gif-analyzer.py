# Do stuff, probably
import imageio.v2 as imageio
import numpy as np
import os
import time
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def load_frames(gif_path):
    image = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = image.convert('RGB')
            frames.append(np.array(frame))
            image.seek(image.tell() + 1)
    except EOFError:
        pass
    return frames

def is_different(f1, f2, threshold=0.90):
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(g1, g2, full=True)
    return score < threshold


import time

def extract_frames(gif_path, threshold=0.30, out_root="frames"):
    timestamp = time.strftime("%Y")
    filename = gif_path
    out_dir = os.path.join(out_root, filename + f"-{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    frames = load_frames(gif_path)
    output = [frames[0]]
    indices = [0]

    for i in range(1, len(frames)):
        if is_different(frames[i], output[-1], threshold):
            output.append(frames[i])
            indices.append(i)
            Image.fromarray(frames[i]).save(f"{out_dir}/frame_{i}.png")

    return indices

# Example usage
gif_path = input("Path to GIF?")
frame_indices = extract_frames(gif_path)
print("Distinct frame indices:", frame_indices)
