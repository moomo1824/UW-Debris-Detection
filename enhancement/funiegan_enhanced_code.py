# -*- coding: utf-8 -*-
"""funiegan_enhanced_code.ipynb
"""

!nvidia-smi

from google.colab import files
uploaded = files.upload()

print("Uploaded:", list(uploaded.keys()))

import os, glob, zipfile, shutil

os.chdir("/content")

code_candidates = glob.glob("FUnIE-GAN-master*.zip") + glob.glob("*FUnIE*GAN*.zip")
assert len(code_candidates) > 0, "Upload your code zip (FUnIE-GAN-master.zip)."
CODE_ZIP = sorted(code_candidates)[0]

img_candidates = glob.glob("images*.zip") + glob.glob("*image*.zip") + glob.glob("*images*.zip")
assert len(img_candidates) > 0, "Upload your input zip (images.zip)."
IMAGES_ZIP = sorted(img_candidates)[0]

print("Using CODE_ZIP:", CODE_ZIP)
print("Using IMAGES_ZIP:", IMAGES_ZIP)

CODE_DIR = "/content/FUnIE-GAN-master"
if os.path.exists(CODE_DIR):
    shutil.rmtree(CODE_DIR)

with zipfile.ZipFile(CODE_ZIP, "r") as z:
    z.extractall("/content")

assert os.path.exists(CODE_DIR), f"Expected folder not found: {CODE_DIR}. Check extracted folder name."

INPUT_BASE = "/content/input_images"
if os.path.exists(INPUT_BASE):
    shutil.rmtree(INPUT_BASE)
os.makedirs(INPUT_BASE, exist_ok=True)

with zipfile.ZipFile(IMAGES_ZIP, "r") as z:
    z.extractall(INPUT_BASE)

print("Code extracted to:", CODE_DIR)
print("Images extracted to:", INPUT_BASE)
print("Top level inside input_images:", os.listdir(INPUT_BASE)[:50])

import os

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def find_image_root(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                return dirpath
    return None

INPUT_DIR = find_image_root("/content/input_images")
print("Found INPUT_DIR:", INPUT_DIR)

assert INPUT_DIR is not None, "No images found inside the zip. Check that images.zip actually contains image files."

!pip -q install pillow

import sys, os, torch

PYTORCH_DIR = "/content/FUnIE-GAN-master/PyTorch"
sys.path.insert(0, PYTORCH_DIR)

from nets import funiegan

MODEL_PATH = "/content/FUnIE-GAN-master/PyTorch/models/funie_generator.pth"
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = funiegan.GeneratorFunieGAN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Loaded model:", MODEL_PATH)
print("Device:", device)

import os, shutil
from PIL import Image
import torch
import torchvision.transforms as T

OUTPUT_DIR = "/content/enhanced_images"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

to_model = T.Compose([
    T.Resize((256, 256), interpolation=Image.BICUBIC),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

@torch.no_grad()
def enhance_pil(img_pil: Image.Image) -> Image.Image:
    orig_w, orig_h = img_pil.size
    x = to_model(img_pil.convert("RGB")).unsqueeze(0).to(device)
    y = model(x).squeeze(0).cpu()
    y = (y * 0.5) + 0.5
    y = torch.clamp(y, 0, 1)
    out = T.ToPILImage()(y)
    out = out.resize((orig_w, orig_h), resample=Image.BICUBIC)
    return out

def iter_images(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                yield os.path.join(dirpath, fn)

count, errors = 0, 0

for in_path in iter_images(INPUT_DIR):
    rel = os.path.relpath(in_path, INPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        img = Image.open(in_path)
        out_img = enhance_pil(img)
        out_img.save(out_path)
        count += 1
        if count % 50 == 0:
            print("Processed:", count)
    except Exception as e:
        errors += 1
        print("FAILED:", in_path, "->", e)

print("Done.")
print("Enhanced:", count, "| Errors:", errors)
print("Output folder:", OUTPUT_DIR)

import os

candidates = ["/kaggle/working", "/kaggle", "/content", "/mnt/data", "/workspace", "."]
for p in candidates:
    print(p, "exists =", os.path.exists(p), "is_dir =", os.path.isdir(p))

!zip -r enhanced_images.zip enhanced_images

from google.colab import files
files.download("enhanced_images.zip")
