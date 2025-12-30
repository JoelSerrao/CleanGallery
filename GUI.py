import sqlite3
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image

# ================= CONFIG =================

DB_Path = Path(r"F:\Projects\CleanGallery\image_DB.db")
Trash_dir = Path(r"F:\Projects\CleanGallery\trash")
Trash_dir.mkdir(exist_ok=True)

# ================= DB =================

conn = sqlite3.connect(DB_Path, check_same_thread=False)
cur = conn.cursor()

# ================= FILTERS =================

st.sidebar.header("Filters")
show_duplicates = st.sidebar.checkbox("Only duplicates", False)
show_screenshots = st.sidebar.checkbox("Only screenshots", False)

query = """
SELECT id, path, duplicate_cluster,
       source_label, source_confidence,
       content_label, content_confidence,
       width, height
FROM images
WHERE deleted = 0
"""

if show_duplicates:
    query += " AND duplicate_cluster IS NOT NULL"
if show_screenshots:
    query += " AND source_label LIKE 'a screenshot%'"

query += " ORDER BY COALESCE(duplicate_cluster, id)"

cur.execute(query)
rows = cur.fetchall()

if not rows:
    st.info("No images to review.")
    st.stop()

# ================= GROUP INTO PAGES =================

pages = []
seen = set()

for r in rows:
    img_id, path, cluster, *_ = r
    if img_id in seen:
        continue

    if cluster is not None:
        group = [x for x in rows if x[2] == cluster]
        for g in group:
            seen.add(g[0])
        pages.append(group)
    else:
        pages.append([r])
        seen.add(img_id)

# ================= STATE =================

if "page" not in st.session_state:
    st.session_state.page = 0
if "marked" not in st.session_state:
    st.session_state.marked = set()
if "confirm_mode" not in st.session_state:
    st.session_state.confirm_mode = False

# Clamp page index after deletions
st.session_state.page = min(st.session_state.page, len(pages) - 1)

# ================= HEADER =================

st.title("CleanGallery")
st.caption(f"{st.session_state.page + 1} / {len(pages)}")

current = pages[st.session_state.page]

# ================= HELPERS =================

def resolution(item):
    return item[7] * item[8]

def toggle(path, value):
    if value:
        st.session_state.marked.add(path)
    else:
        st.session_state.marked.discard(path)

# ================= NAVIGATION =================

st.divider()
nav1, nav2, nav3 = st.columns([1, 1, 2])

with nav1:
    if st.button("◀ Previous", disabled=st.session_state.page == 0):
        st.session_state.page -= 1
        st.rerun()

with nav2:
    if st.button("Next ▶", disabled=st.session_state.page == len(pages) - 1):
        st.session_state.page += 1
        st.rerun()

with nav3:
    st.write(f"🗑️ Selected: {len(st.session_state.marked)}")

# ================= DELETE OPTIONS =================

st.divider()

if st.checkbox("Select all images in this instance"):
    for item in current:
        path = item[1]
        st.session_state.marked.add(path)

# ================= CONTENT =================

if len(current) > 1:
    st.subheader("Duplicate images")

    best = max(current, key=resolution)

    cols = st.columns(len(current))
    for col, item in zip(cols, current):
        _, path, _, s_label, s_conf, c_label, c_conf, w, h = item
        with col:
            checked = path in st.session_state.marked
            st.checkbox(
                "Delete",
                value=checked,
                key=f"delete_{path}",
                on_change=toggle,
                args=(path, not checked)
            )

            if item == best:
                st.markdown("✅ **Best quality**")

            st.image(Image.open(path), width='stretch')
            st.caption(f"{path}")
            st.caption(f"📷 {s_label} ({s_conf:.2f})")
            st.caption(f"🧠 {c_label} ({c_conf:.2f})")
            st.caption(f"📐 Resolution: {w}×{h}")

else:
    _, path, _, s_label, s_conf, c_label, c_conf, w, h = current[0]

    checked = path in st.session_state.marked
    st.checkbox(
        "Delete",
        value=checked,
        key=f"delete_{path}",
        on_change=toggle,
        args=(path, not checked)
    )

    st.image(Image.open(path), width='stretch')
    st.write(f"📍 **Location:** {path}")
    st.write(f"📷 **Source:** {s_label} ({s_conf:.2f})")
    st.write(f"🧠 **Content:** {c_label} ({c_conf:.2f})")
    st.write(f"📐 Resolution: {w}×{h}")

# ================= FILMSTRIP =================

st.divider()
thumbs = st.columns(min(6, len(pages)))
for i, col in enumerate(thumbs):
    idx = max(0, st.session_state.page - 3 + i)
    if idx < len(pages):
        img = pages[idx][0][1]
        with col:
            st.image(Image.open(img), use_container_width=True)
            if st.button("⬆", key=f"jump_{st.session_state.page}_{idx}_{i}"):
                st.session_state.page = idx

# ================= SESSION SUMMARY =================

st.divider()

if not st.session_state.confirm_mode:

    if st.button("Review deletion summary", disabled=len(st.session_state.marked) == 0):
        st.session_state.confirm_mode = True
        st.rerun()

else:
    st.subheader("Review deletion summary")

    marked = list(st.session_state.marked)

    # --- Fetch metadata for summary ---
    q_marks = ",".join("?" for _ in marked)
    cur.execute(f"""
        SELECT path, duplicate_cluster, source_label
        FROM images
        WHERE path IN ({q_marks})
    """, marked)

    summary_rows = cur.fetchall()

    # --- Stats ---
    total = len(summary_rows)
    dup_count = sum(1 for r in summary_rows if r[1] is not None)
    screenshot_count = sum(1 for r in summary_rows if r[2].startswith("a screenshot"))

    st.write(f"**Total images selected:** {total}")
    st.write(f"**Duplicates:** {dup_count}")
    st.write(f"**Screenshots:** {screenshot_count}")

    # --- Folder breakdown ---
    folders = {}
    for path, _, _ in summary_rows:
        folder = str(Path(path).parent)
        folders[folder] = folders.get(folder, 0) + 1

    st.write("**Folders affected:**")
    for f, c in folders.items():
        st.write(f"- {f} ({c})")

    # --- Thumbnail preview ---
    st.write("**Preview:**")
    cols = st.columns(6)
    for i, (path, _, _) in enumerate(summary_rows[:18]):
        with cols[i % 6]:
            if Path(path).exists():
                st.image(Image.open(path), use_container_width=True)

    st.warning("These images will be moved to the trash folder.")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Go back"):
            st.session_state.confirm_mode = False
            st.rerun()

    with col2:
        if st.button("✅ Confirm & move to trash"):
            for p in marked:
                src = Path(p)
                dst = Trash_dir / src.name
                if src.exists():
                    shutil.move(src, dst)
                    cur.execute(
                        "UPDATE images SET deleted = 1 WHERE path = ?",
                        (p,)
                    )

            conn.commit()
            st.session_state.marked.clear()
            st.session_state.confirm_mode = False
            st.success("Images moved to trash.")
            st.rerun()

