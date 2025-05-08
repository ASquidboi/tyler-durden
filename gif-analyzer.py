#Do stuff, probably
from PIL import Image
import numpy as np
import os
import cv2
from skimage.metrics import structural_similarity as ssim

def load_frames(gif_path):
    image = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = image.convert('RGB').copy()  # Force full decode
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
    saved_files = []

    # Save first frame
    path = os.path.join(out_dir, f"frame_0.png")
    Image.fromarray(frames[0]).save(path)
    saved_files.append(path)

    for i in range(1, len(frames)):
        if is_different(frames[i], output[-1], threshold):
            output.append(frames[i])
            indices.append(i)
            path = os.path.join(out_dir, f"frame_{i}.png")
            Image.fromarray(frames[i]).save(path)
            saved_files.append(path)

    return saved_files

# Example usage
gif_path = os.path.expanduser(input("Path to GIF? "))
saved = extract_frames(gif_path)
print("Saved frame files:")
for path in saved:
    print(path)

