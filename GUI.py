import sqlite3
import shutil
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import math
import base64
import io
import hashlib

# ================= CONFIG =================

DB_Path = Path(r"F:\Projects\CleanGallery\image_DB.db")
Trash_dir = Path(r"F:\Projects\CleanGallery\trash")
Trash_dir.mkdir(exist_ok=True)

# ================= INITIALIZE SESSION STATE =================

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.marked = set()
    st.session_state.confirm_mode = False
    st.session_state.filter_state = {"duplicates": False, "screenshots": False}
    st.session_state.loaded_images = []
    st.session_state.images_per_page = 50
    st.session_state.current_offset = 0
    st.session_state.sort_by = "path"
    st.session_state.grid_columns = 6
    st.session_state.cache_version = 0 

# ================= DB =================

conn = sqlite3.connect(DB_Path, check_same_thread=False)
cur = conn.cursor()

# ================= PAGE STYLING =================

st.set_page_config(
    page_title="CleanGallery",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .image-card {
        border: 2px solid transparent;
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 10px;
        transition: all 0.2s;
    }
    .image-card:hover {
        border-color: #3B82F6;
        background-color: #f0f8ff;
    }
    .image-card.selected {
        border-color: #EF4444;
        background-color: #fef2f2;
    }
    .badge {
        display: inline-block;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: bold;
        margin-right: 4px;
    }
    .stats-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .selected-count {
        background: #EF4444;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================

with st.sidebar:
    st.markdown("### 🎛️ Filters & Controls")
    
    # Display mode
    st.markdown("#### 📱 Display")
    grid_cols = st.slider("Grid columns", 3, 8, st.session_state.grid_columns)
    if grid_cols != st.session_state.grid_columns:
        st.session_state.grid_columns = grid_cols
        st.rerun()
    
    # Image filter
    st.markdown("#### 🏷️ Filter by Type")
    show_duplicates = st.checkbox(
        "Only duplicates", 
        value=st.session_state.filter_state["duplicates"]
    )
    show_screenshots = st.checkbox(
        "Only screenshots", 
        value=st.session_state.filter_state["screenshots"]
    )
    
    # Update filter state
    st.session_state.filter_state["duplicates"] = show_duplicates
    st.session_state.filter_state["screenshots"] = show_screenshots
    
    # Sort options
    st.markdown("#### 🔄 Sort By")
    sort_options = {
        "path": "File Path",
        "date": "Date Modified",
        "size": "File Size", 
        "cluster": "Duplicate Cluster"
    }
    selected_sort = st.selectbox(
        "Sort images by",
        options=list(sort_options.keys()),
        format_func=lambda x: sort_options[x]
    )
    if selected_sort != st.session_state.sort_by:
        st.session_state.sort_by = selected_sort
        st.session_state.loaded_images = []
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Quick stats
    cur.execute("SELECT COUNT(*) FROM images WHERE deleted = 0")
    total_images = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM images WHERE deleted = 0 AND duplicate_cluster IS NOT NULL")
    duplicate_images = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM images WHERE deleted = 0 AND source_label LIKE 'a screenshot%'")
    screenshot_count = cur.fetchone()[0]
    
    st.markdown("### 📊 Statistics")
    st.markdown(f"""
    <div class="stats-card">
        <b>Total Images:</b> {total_images}<br>
        <b>Duplicates:</b> {duplicate_images}<br>
        <b>Screenshots:</b> {screenshot_count}<br>
        <b>Marked for deletion:</b> {len(st.session_state.marked)}
    </div>
    """, unsafe_allow_html=True)
    
    # Quick actions
    st.sidebar.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🗑️ Clear all marks", use_container_width=True):
        st.session_state.marked.clear()
        st.rerun()
    
    # Add a clear cache button
    if st.button("🔄 Refresh Cache", use_container_width=True):
        st.session_state.cache_version += 1
        st.success("Cache refreshed! Images will reload.")
        st.rerun()

# ================= LOAD IMAGES FUNCTION =================

@st.cache_data(ttl=3600, show_spinner=False)
def load_images(limit=1000, offset=0, filters=None, sort_by="path", _cache_version=0):
    """Load images from database with caching"""
    query = """
    SELECT id, path, duplicate_cluster,
           source_label, source_confidence,
           content_label, content_confidence,
           width, height
    FROM images
    WHERE deleted = 0
    """
    
    params = []
    
    if filters and filters.get("duplicates"):
        query += " AND duplicate_cluster IS NOT NULL"
    
    if filters and filters.get("screenshots"):
        query += " AND source_label LIKE 'a screenshot%'"
    
    # Add sorting
    if sort_by == "cluster":
        query += " ORDER BY COALESCE(duplicate_cluster, id), path"
    else:  # path
        query += " ORDER BY path"
    
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cur.execute(query, params)
    rows = cur.fetchall()
    
    # Convert to list of dicts for easier handling
    images = []
    for row in rows:
        img_dict = {
            'id': row[0],
            'path': row[1],
            'duplicate_cluster': row[2],
            'source_label': row[3],
            'source_confidence': row[4],
            'content_label': row[5],
            'content_confidence': row[6],
            'width': row[7],
            'height': row[8],
        }
        
        images.append(img_dict)
    
    return images

# ================= IMAGE UTILITIES =================

@st.cache_data(ttl=600, max_entries=1000, show_spinner=False)
def get_image_thumbnail(path, size=(200, 200), _cache_version=0):
    """Get thumbnail for image with caching"""
    try:
        # Check if file exists
        if not Path(path).exists():
            # Return a placeholder for deleted images
            placeholder = Image.new('RGB', size, color='#CCCCCC')
            # Add text to indicate deleted
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(placeholder)
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            text = "Deleted"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
            draw.text(position, text, fill="white", font=font)
            return placeholder
        
        img = Image.open(path)
        img.thumbnail(size)
        return img
    except Exception as e:
        # Return a placeholder if image can't be loaded
        placeholder = Image.new('RGB', size, color='#CCCCCC')
        return placeholder

# ================= HEADER =================

st.markdown('<h1 class="main-header">🗑️ CleanGallery</h1>', unsafe_allow_html=True)

# Header stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Images", total_images)
with col2:
    st.metric("Duplicates", duplicate_images)
with col3:
    st.metric("Marked", len(st.session_state.marked))
with col4:
    if st.session_state.marked:
        total_size = sum(Path(p).stat().st_size for p in st.session_state.marked if Path(p).exists())
        st.metric("Total Size", f"{total_size / (1024*1024):.1f} MB")

# ================= SELECTION TOOLBAR =================

st.markdown("---")

if st.session_state.marked:
    # Selection toolbar
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("✅ Select All Visible", use_container_width=True):
            visible_paths = [img['path'] for img in st.session_state.loaded_images]
            st.session_state.marked.update(visible_paths)
            st.rerun()
    
    with col2:
        if st.button("❌ Deselect All", use_container_width=True):
            st.session_state.marked.clear()
            st.rerun()
    
    with col3:
        if st.button("📱 Mark Screenshots", use_container_width=True):
            for img in st.session_state.loaded_images:
                if "screenshot" in img['source_label'].lower():
                    st.session_state.marked.add(img['path'])
            st.rerun()
    
    with col4:
        if st.button("🔄 Mark Duplicates", use_container_width=True):
            for img in st.session_state.loaded_images:
                if img['duplicate_cluster'] is not None:
                    st.session_state.marked.add(img['path'])
            st.rerun()
    
    with col5:
        if st.button("🚮 Review Marked", type="primary", use_container_width=True):
            st.session_state.confirm_mode = True
            st.rerun()

# ================= IMAGE GRID =================

# Load images
images = load_images(
    limit=st.session_state.images_per_page,
    offset=st.session_state.current_offset,
    filters=st.session_state.filter_state,
    sort_by=st.session_state.sort_by,
    _cache_version=st.session_state.cache_version
)

st.session_state.loaded_images = images

# Display grid
if images:
    # Create rows based on grid columns
    grid_cols = st.session_state.grid_columns
    image_count = len(images)
    
    for i in range(0, image_count, grid_cols):
        # Create columns for this row
        cols = st.columns(grid_cols)
        
        # Fill each column with an image
        for col_idx in range(grid_cols):
            img_idx = i + col_idx
            if img_idx < image_count:
                img = images[img_idx]
                path = img['path']
                is_selected = path in st.session_state.marked
                
                with cols[col_idx]:
                    # Check if file exists before displaying
                    file_exists = Path(path).exists()
                    
                    # Create a container that acts as our card
                    card_class = "image-card selected" if is_selected else "image-card"
                    
                    # Create columns inside the card: checkbox on left, image on right
                    check_col, img_col = st.columns([1, 10])
                    
                    with check_col:
                        # Display checkbox (only if file exists)
                        if file_exists:
                            checked = st.checkbox(
                                "",
                                value=is_selected,
                                key=f"checkbox_{img['id']}_{img_idx}_{st.session_state.cache_version}",
                                label_visibility="collapsed"
                            )
                            
                            # Update marked set if checkbox changed
                            if checked != is_selected:
                                if checked:
                                    st.session_state.marked.add(path)
                                else:
                                    st.session_state.marked.discard(path)
                                st.rerun()
                        else:
                            # Show disabled checkbox for deleted files
                            st.checkbox(
                                "",
                                value=False,
                                disabled=True,
                                key=f"deleted_{img['id']}_{img_idx}",
                                label_visibility="collapsed"
                            )
                    
                    with img_col:
                        # Display badges
                        badge_html = ""
                        if img['duplicate_cluster'] is not None:
                            badge_html += '<span class="badge">DUP</span>'
                        if "screenshot" in img['source_label'].lower():
                            badge_html += '<span class="badge">SCR</span>'
                        
                        if badge_html:
                            st.markdown(badge_html, unsafe_allow_html=True)
                        
                        # Display image
                        try:
                            thumbnail = get_image_thumbnail(
                                path, 
                                _cache_version=st.session_state.cache_version
                            )
                            st.image(thumbnail, use_container_width=True)
                            
                            # Add "Deleted" label if file doesn't exist
                            if not file_exists:
                                st.warning("⚠️ File deleted")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                        
                        # Display filename and info
                        with st.expander("📋", expanded=False):
                            st.caption(f"**{Path(path).name}**")
                            st.caption(f"{img['width']}×{img['height']}")
                            st.caption(f"{img['source_label']} ({img['source_confidence']:.2f})")
                            if not file_exists:
                                st.error("File has been deleted")

# ================= LOAD MORE =================

if len(images) == st.session_state.images_per_page:
    if st.button("📥 Load More Images", use_container_width=True):
        st.session_state.current_offset += st.session_state.images_per_page
        st.rerun()

# ================= SELECTION STATUS =================

if st.session_state.marked:
    st.markdown("---")
    st.markdown(f'<div class="selected-count">🗑️ {len(st.session_state.marked)} images selected</div>', unsafe_allow_html=True)
    
    # Quick delete button
    if st.button("🚀 Quick Delete Selected", type="primary", use_container_width=True):
        st.session_state.confirm_mode = True
        st.rerun()

# ================= BULK ACTIONS =================

st.markdown("---")
st.markdown("### 🚀 Bulk Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📱 Select All Screenshots", use_container_width=True):
        # Query all screenshots
        query = """
        SELECT path FROM images 
        WHERE deleted = 0 AND source_label LIKE 'a screenshot%'
        """
        if st.session_state.filter_state["duplicates"]:
            query += " AND duplicate_cluster IS NOT NULL"
        cur.execute(query)
        screenshot_paths = [row[0] for row in cur.fetchall()]
        st.session_state.marked.update(screenshot_paths)
        st.success(f"Selected {len(screenshot_paths)} screenshots")
        st.rerun()

with col2:
    if st.button("🔄 Select All Duplicates", use_container_width=True):
        # Query all duplicates
        query = """
        SELECT path FROM images 
        WHERE deleted = 0 AND duplicate_cluster IS NOT NULL
        """
        if st.session_state.filter_state["screenshots"]:
            query += " AND source_label LIKE 'a screenshot%'"
        cur.execute(query)
        duplicate_paths = [row[0] for row in cur.fetchall()]
        st.session_state.marked.update(duplicate_paths)
        st.success(f"Selected {len(duplicate_paths)} duplicates")
        st.rerun()

with col3:
    if st.button("⭐ Keep Best in Clusters", use_container_width=True):
        # For each duplicate cluster, keep the best image
        query = """
        SELECT id, path, width, height, duplicate_cluster
        FROM images 
        WHERE deleted = 0 AND duplicate_cluster IS NOT NULL
        ORDER BY duplicate_cluster, (width * height) DESC
        """
        cur.execute(query)
        rows = cur.fetchall()
        
        clusters = {}
        for row in rows:
            cluster_id = row[4]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(row[1])
        
        # Mark all but the first (best) image in each cluster
        for cluster_paths in clusters.values():
            best_image = cluster_paths[0]
            for path in cluster_paths[1:]:
                st.session_state.marked.add(path)
            st.session_state.marked.discard(best_image)
        
        st.success(f"Applied to {len(clusters)} duplicate clusters")
        st.rerun()

# ================= DELETE CONFIRMATION =================

if st.session_state.confirm_mode:
    st.markdown("---")
    st.markdown("## 🗑️ Delete Confirmation")
    
    marked_list = list(st.session_state.marked)
    
    # Fetch detailed info
    q_marks = ",".join("?" for _ in marked_list)
    cur.execute(f"""
        SELECT path, duplicate_cluster, source_label, 
               source_confidence, width, height
        FROM images
        WHERE path IN ({q_marks})
    """, marked_list)
    
    summary_rows = cur.fetchall()
    
    # Statistics
    total_count = len(summary_rows)
    duplicate_count = sum(1 for r in summary_rows if r[1] is not None)
    screenshot_count = sum(1 for r in summary_rows if "screenshot" in r[2].lower())
    total_size = sum(Path(r[0]).stat().st_size for r in summary_rows if Path(r[0]).exists())
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", total_count)
    with col2:
        st.metric("Duplicates", duplicate_count)
    with col3:
        st.metric("Screenshots", screenshot_count)
    with col4:
        st.metric("Size", f"{total_size/(1024*1024):.1f} MB")
    
    # Preview
    st.markdown("### 👁️ Preview")
    preview_cols = st.columns(6)
    for i, (path, _, source_label, conf, w, h) in enumerate(summary_rows[:12]):
        with preview_cols[i % 6]:
            try:
                img = get_image_thumbnail(path, size=(100, 100), _cache_version=st.session_state.cache_version)
                st.image(img, use_container_width=True)
                st.caption(f"{w}×{h}")
                if not Path(path).exists():
                    st.error("⚠️ Already deleted")
            except:
                st.error("❌")
    
    # Warning
    st.warning(f"⚠️ **{len(marked_list)} images will be moved to:** `{Trash_dir}`")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Cancel", use_container_width=True):
            st.session_state.confirm_mode = False
            st.rerun()
    
    with col2:
        if st.button("✅ Confirm & Delete", type="primary", use_container_width=True):
            # Move files
            moved_count = 0
            failed_files = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(marked_list):
                status_text.text(f"Moving {i+1}/{len(marked_list)}: {Path(file_path).name}")
                progress_bar.progress((i + 1) / len(marked_list))
                
                src = Path(file_path)
                if src.exists():
                    dst = Trash_dir / src.name
                    counter = 1
                    while dst.exists():
                        dst = Trash_dir / f"{src.stem}_{counter}{src.suffix}"
                        counter += 1
                    
                    try:
                        shutil.move(str(src), str(dst))
                        cur.execute(
                            "UPDATE images SET deleted = 1 WHERE path = ?",
                            (file_path,)
                        )
                        moved_count += 1
                    except Exception as e:
                        failed_files.append((file_path, str(e)))
            
            conn.commit()
            
            # Clear state and increment cache version to force refresh
            st.session_state.marked.clear()
            st.session_state.confirm_mode = False
            st.session_state.cache_version += 1  # Force cache refresh
            
            # Clear Streamlit cache for thumbnails
            try:
                get_image_thumbnail.clear()
                load_images.clear()
            except:
                pass
            
            # Show results
            progress_bar.empty()
            status_text.empty()
            
            if moved_count > 0:
                st.success(f"✅ Moved {moved_count} images to trash!")
            
            if failed_files:
                st.error(f"❌ Failed to move {len(failed_files)} images")
                for file_path, error in failed_files:
                    st.write(f"• `{file_path}`: {error}")
            
            st.rerun()

# Close connection
conn.close()