import os
import json
from pathlib import Path
from PIL import Image
import torch
import clip
import numpy as np
from tqdm import tqdm

# ================= CONFIG =================

Dir = Path("F:\Projects\CleanGallery\Folders to get images")        # Path of the images to clean
Output_JSON = Path("image_index.json")
Min_Size = 1                     # Minimum file size to consider (in KB)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SOURCE_LABELS = [
    "a photograph taken with a camera",
    "a screenshot of a phone or computer screen"
]

CONTENT_LABELS = [
    "an internet meme with text",
    "a scanned document",
    "a computer user interface",
    "a photograph"
]

DUPLICATE_THRESHOLD = 0.95
CONFIDENCE_MIN = 0.65
CONFIDENCE_MARGIN = 0.15

# ================= LOAD CLIP =================

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def encode_text(labels):
    tokens = clip.tokenize(labels).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb

SOURCE_TEXT = encode_text(SOURCE_LABELS)
CONTENT_TEXT = encode_text(CONTENT_LABELS)

# ================= UTILS =================

def scan_images(root):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.stat().st_size > Min_Size * 1024:
            yield p

def encode_image(path):
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def classify(emb, text_emb, labels):
    sims = emb @ text_emb.T
    top = np.argsort(-sims)
    best, second = top[0], top[1]
    confidence = sims[best]
    margin = sims[best] - sims[second]
    status = "ok"
    if confidence < CONFIDENCE_MIN or margin < CONFIDENCE_MARGIN:
        status = "uncertain"
    return {
        "label": labels[best],
        "confidence": float(confidence),
        "margin": float(margin),
        "status": status
    }

# ================= MAIN =================

records = []
embeddings = []

paths = list(scan_images(Dir))

for p in tqdm(paths, desc="Encoding images"):
    try:
        emb = encode_image(p)
        embeddings.append(emb)

        source = classify(emb, SOURCE_TEXT, SOURCE_LABELS)
        content = classify(emb, CONTENT_TEXT, CONTENT_LABELS)

        records.append({
            "path": str(p),
            "embedding": emb.tolist(),
            "source": source,
            "content": content
        })
    except Exception as e:
        print("Error:", p, e)

# ================= DUPLICATES =================

emb_matrix = np.array(embeddings)
sim_matrix = emb_matrix @ emb_matrix.T

duplicate_groups = []
visited = set()

for i in range(len(records)):
    if i in visited:
        continue
    group = [i]
    for j in range(i + 1, len(records)):
        if sim_matrix[i, j] > DUPLICATE_THRESHOLD:
            group.append(j)
            visited.add(j)
    if len(group) > 1:
        duplicate_groups.append(group)
        visited.update(group)

for group in duplicate_groups:
    for idx in group:
        records[idx]["duplicate_group"] = group

with open(Output_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"Indexed {len(records)} images")
print(f"Found {len(duplicate_groups)} duplicate groups")
