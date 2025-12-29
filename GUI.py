import json
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image

# ================= CONFIG =================

INDEX_FILE = Path(r"F:\Projects\CleanGallery\image_index.json")
TRASH_DIR = Path("trash")
TRASH_DIR.mkdir(exist_ok=True)

# ================= LOAD =================

with open(INDEX_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# ================= FILTER =================

def is_candidate(item):
    if "duplicate_group" in item:
        return True
    if item["source"]["label"].startswith("a screenshot"):
        return True
    if item["content"]["label"].startswith("an internet meme"):
        return True
    return False

candidates = [x for x in data if is_candidate(x)]

st.title("Image Cleanup Review")

st.write(f"Candidates: {len(candidates)}")

# ================= SORT =================

sort_key = st.selectbox(
    "Sort by",
    ["confidence", "path"]
)

if sort_key == "confidence":
    candidates.sort(
        key=lambda x: max(
            x["source"]["confidence"],
            x["content"]["confidence"]
        ),
        reverse=True
    )

# ================= DISPLAY =================

to_delete = []

for item in candidates:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(Image.open(item["path"]), use_container_width=True)

    with col2:
        st.text(item["path"])

        st.write("Source:", item["source"])
        st.write("Content:", item["content"])

        if "duplicate_group" in item:
            st.warning("Duplicate group")

        if st.checkbox("Mark for deletion", key=item["path"]):
            to_delete.append(item["path"])

st.divider()

# ================= DELETE =================

if st.button("Delete selected"):
    for p in to_delete:
        src = Path(p)
        dst = TRASH_DIR / src.name
        shutil.move(src, dst)
    st.success(f"Moved {len(to_delete)} files to trash")
