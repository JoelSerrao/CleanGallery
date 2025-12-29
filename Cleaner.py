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
Output_JSON = Path("F:\Projects\CleanGallery\image_index.json")
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
        emb = emb / emb.norm(dim=-1, keepdim=True)
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
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0)   # Torch tensor

def classify(image_emb, text_emb, labels):
    # image_emb: [512]
    # text_emb: [N, 512]
    sims = (image_emb @ text_emb.T).softmax(dim=-1)

    best = sims.argmax().item()
    sorted_vals, _ = torch.sort(sims, descending=True)

    confidence = sorted_vals[0].item()
    margin = (sorted_vals[0] - sorted_vals[1]).item()

    status = "ok"
    if confidence < CONFIDENCE_MIN or margin < CONFIDENCE_MARGIN:
        status = "uncertain"

    return {
        "label": labels[best],
        "confidence": confidence,
        "margin": margin,
        "status": status
    }

# ================= MAIN =================

records = []
embeddings = []

paths = list(scan_images(Dir))

for p in tqdm(paths, desc="Encoding images"):
    try:
        image_emb = encode_image(p)

        source = classify(image_emb, SOURCE_TEXT, SOURCE_LABELS)
        content = classify(image_emb, CONTENT_TEXT, CONTENT_LABELS)

        # Store NumPy version for duplicates + JSON
        emb_np = image_emb.cpu().numpy()
        embeddings.append(emb_np)

        records.append({
            "path": str(p),
            "embedding": emb_np.tolist(),
            "source": source,
            "content": content
        })

    except Exception as e:
        print("Error:", p, e)

# ================= DUPLICATE DETECTION =================

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

# ================= SAVE =================

with open(Output_JSON, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

print(f"Indexed {len(records)} images")
print(f"Found {len(duplicate_groups)} duplicate groups")
