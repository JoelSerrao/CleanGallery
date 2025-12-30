from pathlib import Path
from PIL import Image
import torch
import clip
from tqdm import tqdm
import sqlite3
import numpy as np
from sklearn.cluster import DBSCAN

# ================= CONFIG =================

Dir = Path(r"E:\Phone Backup\White phone data\Internal\GBWhatsApp\Media\GBWhatsApp Images")
DB_Path = Path(r"F:\Projects\CleanGallery\image_DB.db")

MIN_SIZE_KB = 1

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

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ================= DB =================

conn = sqlite3.connect(DB_Path)
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

    width INTEGER,
    height INTEGER,

    duplicate_cluster INTEGER,
    deleted INTEGER DEFAULT 0
)
""")

conn.commit()

# ================= CLIP =================

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

@torch.no_grad()
def encode_text(labels):
    tokens = clip.tokenize(labels).to(DEVICE)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb

SOURCE_TEXT = encode_text(SOURCE_LABELS)
CONTENT_TEXT = encode_text(CONTENT_LABELS)

# ================= UTILS =================

def to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()

def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def classify(img_emb, text_emb, labels):
    sims = (img_emb @ text_emb.T).squeeze(0)
    values, indices = torch.sort(sims, descending=True)

    best = indices[0].item()
    conf = values[0].item()
    margin = (values[0] - values[1]).item()

    status = "ok" if conf >= CONF_MIN and margin >= CONF_MARGIN else "uncertain"
    return labels[best], conf, margin, status

def scan_images():
    for p in Dir.rglob("*"):
        if (
            p.suffix.lower() in VALID_EXTS
            and p.is_file()
            and p.stat().st_size > MIN_SIZE_KB * 1024
        ):
            yield p

# ================= INDEXING =================

for path in tqdm(list(scan_images()), desc="Indexing images"):
    try:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            width, height = pil_img.size
            img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        s_label, s_conf, s_margin, s_status = classify(
            img_emb, SOURCE_TEXT, SOURCE_LABELS
        )
        c_label, c_conf, c_margin, c_status = classify(
            img_emb, CONTENT_TEXT, CONTENT_LABELS
        )

        emb_np = img_emb.cpu().numpy()[0]

        cur.execute("""
        INSERT OR REPLACE INTO images (
            path, embedding,
            source_label, source_confidence, source_margin, source_status,
            content_label, content_confidence, content_margin, content_status,
            width, height,
            deleted
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            str(path),
            to_blob(emb_np),
            s_label, s_conf, s_margin, s_status,
            c_label, c_conf, c_margin, c_status,
            width, height
        ))

    except Exception as e:
        print(f"Failed: {path} — {e}")

conn.commit()

# ================= DUPLICATE CLUSTERING =================

cur.execute("""
SELECT id, embedding
FROM images
WHERE deleted = 0
""")

rows = cur.fetchall()

if rows:
    ids = []
    vectors = []

    for img_id, blob in rows:
        ids.append(img_id)
        vectors.append(from_blob(blob))

    X = np.vstack(vectors)

    db = DBSCAN(
        eps=0.05,
        min_samples=2,
        metric="cosine"
    )

    labels = db.fit_predict(X)

    for img_id, label in zip(ids, labels):
        cur.execute(
            "UPDATE images SET duplicate_cluster=? WHERE id=?",
            (int(label) if label >= 0 else None, img_id)
        )

    conn.commit()

conn.close()

print("Indexing and duplicate clustering complete.")
