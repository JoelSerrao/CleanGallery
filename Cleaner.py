from pathlib import Path
from PIL import Image
import torch
import clip
from tqdm import tqdm
import sqlite3
import numpy as np

# ================= CONFIG =================

Dir = Path(r"F:\Projects\CleanGallery\Folders to get images")        # Path of the images to clean
DB_path = Path(r"F:\Projects\CleanGallery\image_DB.db")               # Path to the image database
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

CONF_MIN = 0.65
CONF_MARGIN = 0.15

# ================= DB =================

conn = sqlite3.connect(DB_path)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE,
    embedding BLOB,

    source_label TEXT,
    source_confidence REAL,
    source_margin REAL,
    source_status TEXT,

    content_label TEXT,
    content_confidence REAL,
    content_margin REAL,
    content_status TEXT,

    duplicate_cluster INTEGER
)
""")

conn.commit()

# ================= CLIP =================

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

def to_blob(vec):
    return vec.astype(np.float32).tobytes()

def classify(emb_torch, text_emb, labels):
    sims = (emb_torch @ text_emb.T).squeeze(0)

    values, indices = torch.sort(sims, descending=True)
    best = indices[0].item()

    conf = values[0].item()
    margin = (values[0] - values[1]).item()

    status = "ok" if conf >= CONF_MIN and margin >= CONF_MARGIN else "uncertain"

    return labels[best], conf, margin, status


def scan_images():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in Dir.rglob("*"):
        if p.suffix.lower() in exts and p.stat().st_size > Min_Size * 1024:
            yield p

# ================= MAIN =================

for path in tqdm(list(scan_images()), desc="Indexing images"):
    try:
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb_torch = model.encode_image(img)
            emb_torch /= emb_torch.norm(dim=-1, keepdim=True)

        s_label, s_conf, s_margin, s_status = classify(emb_torch, SOURCE_TEXT, SOURCE_LABELS)
        c_label, c_conf, c_margin, c_status = classify(emb_torch, CONTENT_TEXT, CONTENT_LABELS)

        emb_np = emb_torch.cpu().numpy()[0]

        cur.execute("""
        INSERT OR REPLACE INTO images
        (path, embedding,
         source_label, source_confidence, source_margin, source_status,
         content_label, content_confidence, content_margin, content_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(path),
            to_blob(emb_np),
            s_label, s_conf, s_margin, s_status,
            c_label, c_conf, c_margin, c_status
        ))

    except Exception as e:
        print("Failed:", path, e)

conn.commit()
conn.close()

# ================= DUPLICATE CLUSTERING =================
import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN

def from_blob(blob):
    return np.frombuffer(blob, dtype=np.float32)

conn = sqlite3.connect(DB_path)
cur = conn.cursor()

cur.execute("SELECT id, embedding FROM images")
rows = cur.fetchall()

ids = []
embeddings = []

for img_id, blob in rows:
    ids.append(img_id)
    embeddings.append(from_blob(blob))

X = np.vstack(embeddings)

db = DBSCAN(
    eps=0.05,
    min_samples=2,
    metric="cosine"
)

labels = db.fit_predict(X)

for img_id, label in zip(ids, labels):
    if label >= 0:
        cur.execute(
            "UPDATE images SET duplicate_cluster=? WHERE id=?",
            (int(label), img_id)
        )

conn.commit()
conn.close()

print("Duplicate clustering complete.")

