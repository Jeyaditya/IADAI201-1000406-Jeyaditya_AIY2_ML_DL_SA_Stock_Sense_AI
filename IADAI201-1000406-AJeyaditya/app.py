"""
StockSense Pro: Cognitive Retail Vision & Material Intelligence

Author: A Jeyaditya (Student ID: 1000406)
School: Jain Vidyalaya IB World School
CRS: Artificial Intelligence | Scenario 2: Smart Retail Vision
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Import modular retail and material intelligence functions
from src.stock_logic import (
    DEFAULT_CATEGORIES,
    STATUS_NAMES,
    parse_detections,
    generate_triaged_inventory,
)

# -----------------------------------------------------------------------------
# Page Configuration & Light Theme Foundations
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="StockSense Pro | Shelf Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Light Styling
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, div, li {
        color: #0F172A !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    .hero-banner {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 50%, #0284C7 100%) !important;
        padding: 2rem 2.25rem !important;
        border-radius: 16px !important;
        color: #FFFFFF !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25) !important;
    }
    .hero-banner h1, .hero-banner p, .hero-banner div {
        color: #FFFFFF !important;
    }

    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1.1rem 1rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
        text-align: center !important;
        border-top: 4px solid #3B82F6 !important;
    }
    .metric-title {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        color: #64748B !important;
        letter-spacing: 0.06em !important;
    }
    .metric-value {
        font-size: 1.85rem !important;
        font-weight: 900 !important;
        color: #0F172A !important;
        margin-top: 0.2rem !important;
    }

    .column-header-oos {
        background: #FEE2E2 !important;
        border: 1px solid #FCA5A5 !important;
        border-top: 4px solid #EF4444 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 12px !important;
        font-weight: 800 !important;
        color: #991B1B !important;
        font-size: 0.95rem !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .column-header-low {
        background: #FEF3C7 !important;
        border: 1px solid #FCD34D !important;
        border-top: 4px solid #F59E0B !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 12px !important;
        font-weight: 800 !important;
        color: #92400E !important;
        font-size: 0.95rem !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .column-header-optimal {
        background: #DCFCE7 !important;
        border: 1px solid #86EFAC !important;
        border-top: 4px solid #10B981 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-bottom: 12px !important;
        font-weight: 800 !important;
        color: #166534 !important;
        font-size: 0.95rem !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }

    .product-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
    }
    .product-card-title {
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
    }
    .material-pill {
        display: inline-block !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #475569 !important;
        background: #F1F5F9 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        padding: 2px 7px !important;
        margin-top: 6px !important;
    }
    .recycling-info {
        font-size: 0.72rem !important;
        color: #64748B !important;
        margin-top: 4px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path("IADAI201-1000406-AJeyaditya/models/best.pt")
MAX_DISPLAY_DIM = 1024


@st.cache_resource(show_spinner=False)
def load_detection_model(model_path_str: str) -> Optional[YOLO]:
    path = Path(model_path_str)
    if not path.exists():
        return None
    return YOLO(str(path))


def resize_for_display(image: Image.Image, max_dim: int = MAX_DISPLAY_DIM) -> Image.Image:
    w, h = image.size
    scale = min(1.0, max_dim / w, max_dim / h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image


# -----------------------------------------------------------------------------
# Sidebar: Parameter Controls & Academic Submission Details
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg",
        width=52,
    )
    st.markdown("## **StockSense Pro**")
    st.markdown("*Cognitive Retail Vision & Materials*")
    st.divider()

    st.markdown("### ⚙️ Detection Tuning")
    conf_thresh = st.slider(
        "Confidence Cutoff ($\\tau_{conf}$)",
        min_value=0.15,
        max_value=0.85,
        value=0.32,  # Set to 0.32 to suppress background hallucinations
        step=0.01,
        help="Higher values (>0.30) eliminate false positives on background textures.",
    )

    iou_thresh = st.slider(
        "NMS IoU Cutoff ($\\tau_{iou}$)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Non-Maximum Suppression threshold to merge overlapping duplicate boxes.",
    )

    st.markdown("### 📦 Stock Boundaries")
    low_stock_limit = st.slider(
        "Critical Low Ceiling",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="Item counts between 1 and this ceiling trigger Low Stock warnings.",
    )

    target_shelf_capacity = st.number_input(
        "Target Facing Capacity",
        min_value=2,
        max_value=20,
        value=4,
        step=1,
        help="Nominal target facings used to calculate Fill Factor and Urgency Score.",
    )

    st.divider()
    st.markdown("### 🎓 Academic Submission")
    st.markdown(
        """
        - **Student:** A Jeyaditya
        - **Registration No:** 1000406  
        - **School:** Jain Vidyalaya IB World School  
        - **Course:** Machine Learning & Deep Learning  
        - **CRS:** Artificial Intelligence  
        - **Scenario:** Scenario 2 (Smart Retail Vision)
        """
    )


# Verify Model Checkpoint
model = load_detection_model(str(MODEL_PATH))
if model is None:
    st.error(
        f"🚨 **Model checkpoint missing:** Could not find `{MODEL_PATH}`. Ensure `best.pt` is inside `models/`."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Top Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1 style="font-size:2.2rem;font-weight:800;margin:0;">StockSense Pro: Cognitive Shelf & Material Intelligence</h1>
        <p style="font-size:1.05rem;opacity:0.95;margin-top:0.4rem;">
            Real-Time SKU Auditing • Three-Column Stock Triage Board • Packaging Substrate Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Clean Direct File Upload (No Demonstration Presets)
# -----------------------------------------------------------------------------
st.markdown("### 📤 Upload Retail Scene")
uploaded_file = st.file_uploader(
    "Choose a retail shelf or checkout photograph (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload an image to execute real-time inventory assessment.",
)

if uploaded_file is None:
    st.info("👆 Please upload a retail shelf or checkout photograph above to initiate the vision pipeline.")
    st.stop()

try:
    selected_image = Image.open(uploaded_file).convert("RGB")
    image_label = uploaded_file.name
except Exception as e:
    st.error(f"Error reading image file: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Color-Accurate Inference Pipeline
# -----------------------------------------------------------------------------
resized_image = resize_for_display(selected_image)

# Explicit RGB -> BGR conversion
rgb_np = np.array(resized_image)
bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

with st.spinner("Analyzing shelf facings and profiling packaging substrates..."):
    t_start = time.perf_counter()
    results = model.predict(
        source=bgr_np,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=640,
        verbose=False,
    )
    t_latency = (time.perf_counter() - t_start) * 1000

# Convert annotated result back to RGB for display
annotated_bgr = results[0].plot()
annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

# Execute modular stock logic with 3-tier partitioning
counts = parse_detections(results, category_names=DEFAULT_CATEGORIES, conf_threshold=conf_thresh)
triage_data = generate_triaged_inventory(
    counts,
    low_threshold=low_stock_limit,
    target_capacity=target_shelf_capacity,
)

total_items = sum(counts.values())
oos_count = len(triage_data["depleted"])
low_count = len(triage_data["low_stock"])
optimal_count = len(triage_data["optimal"])
fill_factor = triage_data["fill_factor"]


# -----------------------------------------------------------------------------
# Top-Level Operational KPI Cards
# -----------------------------------------------------------------------------
st.markdown("### 📊 Operational Shelf Performance")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color: #3B82F6;">
            <div class="metric-title">Total Units Detected</div>
            <div class="metric-value">{total_items}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color: #10B981;">
            <div class="metric-title">Normalized Fill Factor</div>
            <div class="metric-value">{fill_factor:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color: #EF4444;">
            <div class="metric-title">Depleted Facings</div>
            <div class="metric-value" style="color: #EF4444 !important;">{oos_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color: #F59E0B;">
            <div class="metric-title">Low Stock Facings</div>
            <div class="metric-value" style="color: #F59E0B !important;">{low_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k5:
    st.markdown(
        f"""
        <div class="metric-card" style="border-top-color: #6366F1;">
            <div class="metric-title">Inference Latency</div>
            <div class="metric-value">{t_latency:.1f} <span style="font-size:1rem;font-weight:400;color:#64748B !important;">ms</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# -----------------------------------------------------------------------------
# Spatial Localization Visualizer
# -----------------------------------------------------------------------------
st.markdown("### 🔍 Spatial Detection & Localization")
c_img_l, c_img_r = st.columns(2)

with c_img_l:
    st.markdown("**Original Shelf Input**")
    st.image(resized_image, width="stretch", caption=f"Source: {image_label}")

with c_img_r:
    st.markdown("**YOLO11n Neural Detections**")
    st.image(annotated_rgb, width="stretch", caption=f"Resolution: 640x640 | Objects Found: {total_items}")

st.write("")


# -----------------------------------------------------------------------------
# 3-Column Inventory Triage Board
# -----------------------------------------------------------------------------
st.markdown("### 🗂️ Interactive Inventory Triage Board")
st.caption("Inventory categorized into 3 operational action tiers. Review stock levels and packaging substrates without vertical scrolling.")

col_oos, col_low, col_opt = st.columns(3)

# Column 1: Depleted Stock (0 Units)
with col_oos:
    st.markdown(
        f"""
        <div class="column-header-oos">
            <span>🔴 DEPLETED STOCK (0 Units)</span>
            <span>{len(triage_data['depleted'])} items</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not triage_data["depleted"]:
        st.info("No depleted facings. All monitored categories have stock.")
    else:
        for item in triage_data["depleted"]:
            st.markdown(
                f"""
                <div class="product-card" style="border-left: 5px solid #EF4444 !important;">
                    <div class="product-card-title">
                        <span>{item['display_name']}</span>
                        <span style="color:#DC2626 !important; font-weight:900;">0 Units</span>
                    </div>
                    <div class="material-pill">📦 {item['material']}</div>
                    <div class="recycling-info">
                        <b>Substrate:</b> {item['form']} &nbsp;|&nbsp; <b>Status:</b> Immediate Restock Needed
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Column 2: Critical Low Stock (1-3 Units)
with col_low:
    st.markdown(
        f"""
        <div class="column-header-low">
            <span>🟡 CRITICAL LOW STOCK (1–3 Units)</span>
            <span>{len(triage_data['low_stock'])} items</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not triage_data["low_stock"]:
        st.info("No low stock warnings. Remaining items are well-stocked.")
    else:
        for item in triage_data["low_stock"]:
            st.markdown(
                f"""
                <div class="product-card" style="border-left: 5px solid #F59E0B !important;">
                    <div class="product-card-title">
                        <span>{item['display_name']}</span>
                        <span style="color:#D97706 !important; font-weight:900;">{item['count']} Units</span>
                    </div>
                    <div class="material-pill">📦 {item['material']}</div>
                    <div class="recycling-info">
                        <b>Substrate:</b> {item['form']} &nbsp;|&nbsp; <b>Restock Urgency:</b> {item['urgency_score']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Column 3: Optimal Inventory (>3 Units)
with col_opt:
    st.markdown(
        f"""
        <div class="column-header-optimal">
            <span>🟢 OPTIMAL INVENTORY (&gt;3 Units)</span>
            <span>{len(triage_data['optimal'])} items</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not triage_data["optimal"]:
        st.warning("No categories currently have optimal stock (>3 units).")
    else:
        for item in triage_data["optimal"]:
            st.markdown(
                f"""
                <div class="product-card" style="border-left: 5px solid #10B981 !important;">
                    <div class="product-card-title">
                        <span>{item['display_name']}</span>
                        <span style="color:#16A34A !important; font-weight:900;">{item['count']} Units</span>
                    </div>
                    <div class="material-pill">📦 {item['material']}</div>
                    <div class="recycling-info">
                        <b>Substrate:</b> {item['form']} &nbsp;|&nbsp; <b>Recyclability:</b> {item['recyclability']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
