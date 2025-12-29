import sqlite3
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image
from streamlit.components.v1 import html

DB_PATH = Path(r"F:\Projects\CleanGallery\image_DB.db")
TRASH = Path("trash")
TRASH.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

# ================= KEYBOARD =================

html("""
<script>
document.addEventListener('keydown', function(e) {
    if (e.key === 'd') window.location.search='?action=delete';
    if (e.key === 'k') window.location.search='?action=keep';
    if (e.key === 'j') window.location.search='?action=next';
    if (e.key === 'l') window.location.search='?action=prev';
});
</script>
""")

# ================= STATE =================

if "index" not in st.session_state:
    st.session_state.index = 0
if "to_delete" not in st.session_state:
    st.session_state.to_delete = set()

params = st.query_params
action = params.get("action", [None])[0]

# ================= LOAD CANDIDATES =================

cur.execute("""
SELECT id, path, source_label, source_confidence,
       content_label, content_confidence, duplicate_cluster
FROM images
WHERE
    duplicate_cluster IS NOT NULL
    OR source_label LIKE 'a screenshot%'
    OR content_label LIKE 'an internet meme%'
""")

rows = cur.fetchall()

if not rows:
    st.info("No candidates.")
    st.stop()

# ================= ACTION HANDLING =================

if action == "next":
    st.session_state.index = min(st.session_state.index + 1, len(rows) - 1)
elif action == "prev":
    st.session_state.index = max(st.session_state.index - 1, 0)
elif action == "delete":
    st.session_state.to_delete.add(rows[st.session_state.index][1])
    st.session_state.index = min(st.session_state.index + 1, len(rows) - 1)
elif action == "keep":
    st.session_state.index = min(st.session_state.index + 1, len(rows) - 1)

# ================= DISPLAY =================

img_id, path, s_label, s_conf, c_label, c_conf, dup = rows[st.session_state.index]

st.title("Image Review (Keyboard Enabled)")
st.write(f"{st.session_state.index + 1} / {len(rows)}")
st.image(Image.open(path), use_container_width=True)

st.text(path)
st.write("Source:", s_label, f"{s_conf:.2f}")
st.write("Content:", c_label, f"{c_conf:.2f}")

if dup is not None:
    st.warning("Duplicate cluster")

st.write("""
**Keyboard shortcuts**
- **D** → delete
- **K** → keep
- **J / L** → next / previous
""")

# ================= DELETE =================

if st.button("Delete marked images"):
    for p in st.session_state.to_delete:
        dst = TRASH / Path(p).name
        shutil.move(p, dst)
    st.success(f"Moved {len(st.session_state.to_delete)} files to trash")
    st.session_state.to_delete.clear()
