import os
import shutil
import csv
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict

import torch
import clip
from PIL import Image
import imagehash
from tqdm import tqdm

# Configuration

ROOT_DIR = Path(r"F:\Projects\CleanGallery\Folders to get images")        # Path of the images to clean
REVIEW_DIR = Path(r"F:\Projects\CleanGallery\cleanup_review")
HASH_THRESHOLD = 3
MIN_FILE_SIZE_KB = 1                     # Minimum file size to consider (in KB)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_LABELS = [
    "a photograph",
    "an internet meme with text",
    "a screenshot of a phone or computer screen",
    "a scanned document",
    "a computer user interface"
]

# Utilities

def open_image(path):
    if os.name == "nt":
        os.startfile(path)
    elif os.name == "posix":
        subprocess.run(["xdg-open", str(path)], check=False)

def image_resolution(path):
    try:
        with Image.open(path) as img:
            return img.width * img.height
    except:
        return 0

# 1. Scan for images

def scan_images(root):
    images = []
    for path in root.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            if path.stat().st_size >= MIN_FILE_SIZE_KB * 1024:
                images.append(path)
    return images

# 2. Perceptual Hashing

def compute_hashes(paths):
    hashes = {}
    for p in tqdm(paths, desc="Computing pHash"):
        try:
            with Image.open(p) as img:
                hashes[p] = imagehash.phash(img)
        except:
            continue
    return hashes

def group_duplicates(hashes):
    groups = []
    used = set()
    items = list(hashes.items())

    for i, (p1, h1) in enumerate(items):
        if p1 in used:
            continue
        group = [p1]
        for p2, h2 in items[i + 1:]:
            if p2 in used:
                continue
            if abs(h1 - h2) <= HASH_THRESHOLD:
                group.append(p2)
        if len(group) > 1:
            used.update(group)
            groups.append(group)

    return groups

# 3. CLIP Classification

def load_clip():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    text = clip.tokenize(CLIP_LABELS).to(DEVICE)
    return model, preprocess, text

def classify_image(path, model, preprocess, text_tokens):
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        similarity = (image_features @ text_features.T).softmax(dim=-1)
    idx = similarity.argmax().item()
    return CLIP_LABELS[idx], similarity[0][idx].item()

# 4. Review and Cleanup

def prepare_review_dirs():
    for sub in ["duplicates", "memes", "screenshots", "documents"]:
        (REVIEW_DIR / sub).mkdir(parents=True, exist_ok=True)

def move_for_review(path, category):
    target = REVIEW_DIR / category / path.name
    shutil.copy2(path, target)
    return target


def main():
    REVIEW_DIR.mkdir(exist_ok=True)

    images = scan_images(ROOT_DIR)
    print(f"Found {len(images)} images")

    hashes = compute_hashes(images)
    duplicate_groups = group_duplicates(hashes)

    duplicates_to_review = []
    for group in duplicate_groups:
        best = max(group, key=image_resolution)
        for p in group:
            if p != best:
                duplicates_to_review.append(p)

    model, preprocess, text_tokens = load_clip()
    prepare_review_dirs()

    review_log = []

    for img in tqdm(images, desc="Classifying"):
        if img in duplicates_to_review:
            moved = move_for_review(img, "duplicates")
            review_log.append([img, moved, "duplicate", ""])
            continue

        label, score = classify_image(img, model, preprocess, text_tokens)

        if "meme" in label:
            cat = "memes"
        elif "screenshot" in label or "interface" in label:
            cat = "screenshots"
        elif "document" in label:
            cat = "documents"
        else:
            continue

        moved = move_for_review(img, cat)
        review_log.append([img, moved, label, score])

    # Save CSV
    with open(REVIEW_DIR / "review.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_path", "review_path", "reason", "confidence"])
        writer.writerows(review_log)

    print("\nReview phase starting...\n")

    for original, review_path, reason, score in review_log:
        print(f"\nOriginal: {original}")
        print(f"Reason: {reason}  Confidence: {score}")
        open_image(review_path)

        resp = input("Delete this file? [y/N/q]: ").strip().lower()
        if resp == "y":
            os.remove(original)
            print("Deleted.")
        elif resp == "q":
            print("Aborting review.")
            break
        else:
            print("Skipped.")

if __name__ == "__main__":
    main()