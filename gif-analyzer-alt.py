import imageio.v2 as imageio
import numpy as np
import os
import time
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def load_frames(gif_path):
    frames = []
    reader = imageio.get_reader(gif_path)
    for frame in reader:
        frames.append(np.array(frame))
    return frames

def is_different(f1, f2, threshold=0.90):
    g1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(g1, g2, full=True)
    return score < threshold

def get_next_build_dir(base_name, root="frames"):
    os.makedirs(root, exist_ok=True)
    i = 1
    while True:
        dir_name = os.path.join(root, f"{base_name}_{i:03}")
        if not os.path.exists(dir_name):
            return dir_name
        i += 1

def extract_frames(gif_path, threshold=0.30, out_root="frames"):
    base_name = os.path.splitext(os.path.basename(gif_path))[0]
    out_dir = get_next_build_dir(base_name, out_root)
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
gif_path = input("Filepath to gif?")
frame_indices = extract_frames(gif_path)
print("Distinct frame indices:", frame_indices)

