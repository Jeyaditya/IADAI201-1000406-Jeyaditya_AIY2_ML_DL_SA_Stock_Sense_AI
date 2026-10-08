# 1000406_Jeyaditya_AIY2_ML_DL_SA_Stock_Sense_AI

# StockSense Pro: AI-Powered Cognitive Retail Vision & Material Intelligence

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO11n](https://img.shields.io/badge/Model-YOLO11n-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Academic Submission Metadata
* **Student Name:** A Jeyaditya (JD)
* **Candidate Registration Number:** 1000406
* **Educational Institution:** Jain Vidyalaya IB World School
* **Course:** Machine Learning and Deep Learning
* **Career-related Study (CRS):** Artificial Intelligence
* **Scenario Selected:** Scenario 2 — Smart Retail Vision System

App link:(Click here to access)[]

---

## 1. Executive Summary & Problem Formulation
In high-velocity commercial retail environments, shelf out-of-stock (OOS) conditions and misplaced inventory account for substantial annual revenue losses and heavy manual audit overhead. Traditional inventory tracking depends on manual barcode scanning, which is time-consuming, labor-intensive, and susceptible to human error.

**StockSense Pro** is an autonomous, cognitive retail computer vision system designed to execute real-time shelf inventory audits directly from static retail counter and shelf imagery. Moving beyond basic object detection, StockSense Pro introduces:
1. **Multi-Class SKU Auditing:** Detects, classifies, and counts 10 primary retail product supercategories in dense, occluded environments.
2. **3-Column Stock Triage Board:** Categorizes inventory into three operational action tiers (**Depleted Facing**, **Critical Low Stock**, and **Optimal Inventory**).
3. **Packaging Substrate & Material Intelligence:** Maps localized SKUs to physical packaging substrates (e.g., Tinplate Steel, BOPP Foil, PET Polymer, Kraftboard) for circular economy handling and material tracking.
4. **Autonomous Replenishment Queuing:** Calculates dynamic restocking urgency indexes to prioritize immediate shelf replenishment.

---

## 2. System Architecture & Technical Stack
The end-to-end processing pipeline decouples computer vision inference from operational inventory classification:

```
                  [Input Retail Imagery (Static / Shelf)]
                                     │
                                     ▼
               [Color Space Alignment: PIL (RGB) -> cv2 (BGR)]
                                     │
                                     ▼
             [YOLO11n Neural Detection Backbone (imgsz=640)]
                                     │
                                     ▼
             [Decoupled Head & Non-Maximum Suppression (NMS)]
             (Dynamic Cutoff: tau_conf = 0.32, tau_iou = 0.50)
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
       [Bounding Box Coordinates]             [Predicted Class IDs]
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
               [Modular Inventory Logic Engine (stock_logic.py)]
                  │                                     │
                  ├─────────────────┐                   │
                  ▼                 ▼                   ▼
          [3-Tier Triage]  [Substrate Profiler]  [Fill Factor Engine]
          (OOS / Low / OK)  (Steel / Foil / PET)  (Capacity Metric)
                  │                 │                   │
                  └─────────────────┼───────────────────┘
                                    │
                                    ▼
           [Streamlit Cognitive Dashboard (Light Theme AAA)]
           • Executive KPI Metric Cards
           • Side-by-Side Spatial Detection Inspector
           • 3-Column Interactive Replenishment Board
```

* **Deep Learning Framework:** PyTorch & Ultralytics YOLO11n (2.59M parameters, 6.5 GFLOPs).
* **Color Space & Vision Pipeline:** OpenCV (`cv2`) & Pillow (`PIL`).
* **Web UI & Visualization:** Streamlit with custom CSS (enforcing WCAG 2.1 AAA contrast compliance).
* **Data Processing & Testing:** NumPy, Pandas, PyYAML, Pytest.

---

## 3. Dataset Engineering & Supercategory Taxonomy
StockSense Pro is trained on retail scenes sampled from the **Retail Product Checkout (RPC)** benchmark dataset. To resolve visual ambiguity while covering core supermarket departments, fine-grained SKUs are mapped into **10 retail supercategories**:

| Class ID | Supercategory | Real-World SKU Examples | Physical Packaging Substrate | Recyclability Classification |
| :---: | :--- | :--- | :--- | :--- |
| `0` | `drink` | Soda cans, tea bottles, mineral water | Polyethylene Terephthalate (PET) / Aluminum | Resin Code 1 / Infinitely Recyclable Metal |
| `1` | `puffed_food` | Potato chips, crisps, crackers | Metallized BOPP Foil Barrier Film | Multi-layer Flexible Plastic Laminate |
| `2` | `instant_noodles`| Cup ramen, noodle bowls | Expanded Polystyrene (EPS) / Kraftboard | Coated Cellulose Composite |
| `3` | `dessert` | Pastries, cookies, yogurt cups | Solid Bleached Sulfate (SBS) / PP | Paperboard / Resin Code 5 |
| `4` | `canned_food` | Canned beans, fish, meat tins | Tinplate Steel / Drawn Aluminum | 100% Infinitely Recyclable Metal |
| `5` | `stationery` | Notebooks, glue sticks, pen packs | Bleached Wood Pulp / Polystyrene | Biodegradable Fiber / Polymer |
| `6` | `dried_food` | Dried mushrooms, nuts, seeds | Kraft Polyethylene Stand-Up Pouch | High-Density Polyethylene Film |
| `7` | `chocolate` | Candy bars, boxed chocolates | Cold-Seal Metallized PP Foil | Lightweight Barrier Polymer Film |
| `8` | `candy` | Hard candy bags, mint packs | Cast Polypropylene (CPP) Film | Polymer Packaging Film |
| `9` | `milk` | Milk jugs, yogurt drinks | High-Density Polyethylene (HDPE) / Tetra Pak | Resin Code 2 / Aseptic Composite Pulp |

### Dataset Partitioning & Support Counts
* **Total Cluttered Scenes:** 450 images (sampled exclusively from multi-product checkout scenes).
* **Total Supervised Annotations:** 3,565 bounding boxes.
* **Train Split (70%):** 315 images (2,504 bounding boxes).
* **Validation Split (15%):** 67 images (534 bounding boxes).
* **Test Split (15%):** 68 images (527 bounding boxes).

---

## 4. Mathematical Formulations

### 1. Normalized Shelf Fill Factor ($\Phi_{\text{shelf}}$)
Quantifies overall shelf stocking level as a bounded percentage:
$$\Phi_{\text{shelf}} = \frac{\sum_{k=1}^K \min(C_k, T_{\text{cap}})}{K \cdot T_{\text{cap}}} \times 100\%$$
Where $K = 10$, $C_k$ is the detected unit count of category $k$, and $T_{\text{cap}} = 4$ is the nominal target capacity per facing.

### 2. Restock Urgency Index ($U_k$)
Maps facing deficits to a prioritized replenishment scale:
$$U_k = \max\left(0.0, \, \frac{T_{\text{cap}} - C_k}{T_{\text{cap}}}\right)$$
* Depleted ($C_k = 0$): $U_k = 1.00$ (Critical replenishment priority).
* Low Stock ($C_k = 2, T_{\text{cap}} = 4$): $U_k = 0.50$ (Scheduled replenishment).
* Fully Stocked ($C_k \ge T_{\text{cap}}$): $U_k = 0.00$ (Zero urgency).

### 3. Mean Absolute Error (MAE) for Inventory Counting
Measures count discrepancies against ground truth:
$$\text{MAE}_{\text{system}} = \frac{1}{K \cdot N} \sum_{k=0}^{K-1} \sum_{i=1}^N \left\vert{} \hat{y}_{i, k} - y_{i, k} \right\vert{}$$

### 4. Background False Positive Rate under Covariate Shift
Models background false detections caused by domain shift:
$$P(\text{FP} \mid \tau_{\text{conf}}) = 1 - \Phi\left(\frac{\ln\left(\frac{\tau_{\text{conf}}}{1 - \tau_{\text{conf}}}\right) - \mu_{\text{shelf}}}{\sigma_{\text{shelf}}}\right)$$
Raising $\tau_{\text{conf}}$ to $0.32$ suppresses the distribution tail, eliminating background shelf noise.

---

## 5. Model Training & Iterative Evolution

The vision engine was developed and evaluated across two major architectural iterations:

| Metric / Parameter | Model 1 (Baseline: Turntable Hybrid) | Model 2 (Production: Multi-Product Clutter) | Engineering Impact |
| :--- | :---: | :---: | :--- |
| **Data Composition** | 400 images (Single items on turntable) | 450 images (Dense multi-product scenes) | Eliminated clean background bias |
| **Annotated Boxes** | 782 boxes | **3,565 boxes** | **+355.8% instance volume increase** |
| **Supercategory Scope**| 6 departments | **10 departments** | Expanded vocabulary coverage |
| **`puffed_food` MAE** | 0.92 items (Severe underfitting) | **0.03 items** | Successfully learned crumpled foil textures |
| **`dessert` MAE** | 1.14 items (Misclassification) | **0.40 items** | Generalized across boxes, tubs, and pouches |
| **Overall System MAE** | 0.84 items | **0.19 items** | Production-ready counting fidelity (~93.7%) |
| **Inference Latency** | 12.4 ms (Tesla T4) / 85 ms (CPU) | **11.8 ms (Tesla T4) / 74 ms (CPU)** | Sub-100 ms real-time responsiveness |

---

## 6. Directory Structure
```
IADAI201-1000406-AJeyaditya/
├── .streamlit/
│   └── config.toml             # Theme enforcement (light mode)
├── data/
│   ├── dataset/                # 450 images, YOLO annotations, data.yaml
│   │   ├── images/             # train / val / test splits
│   │   ├── labels/             # train / val / test YOLO txt annotations
│   │   └── data.yaml           # nc: 10 configuration mapping
│   └── sample_images/          # Benchmark retail shelf test imagery
├── models/
│   └── best.pt                 # 10-class trained model weights (~6 MB)
├── notebooks/
│   └── StockSense_Pro.ipynb    # Training workflow and evaluation notebook
├── src/
│   ├── __init__.py
│   ├── build_cluttered_dataset.py # Cluttered scene extraction engine
│   ├── stock_logic.py          # 3-tier triage & packaging substrate profiler
│   └── validate_dataset.py     # Dynamic dataset schema verification engine
├── tests/
│   ├── __init__.py
│   └── test_stock_logic.py     # Automated pytest business logic test suite
├── .gitignore                  # Excludes venv, pycache, and build artifacts
├── app.py                      # Main Streamlit web application
├── repair_and_validate.py      # Automated schema synchronization utility
├── requirements.txt            # Production Python package manifest
└── README.md                   # Comprehensive system documentation
```

---

## 7. Installation & Local Execution

### Prerequisites
* Python 3.10 or higher
* Git

### Step-by-Step Setup
1. **Clone the repository:**

2. **Create and activate a virtual environment:**
   ```bash
   # Windows Command Prompt
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install production dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the automated unit test suite:**
   ```bash
   pytest tests/
   ```

5. **Launch the Streamlit web dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 8. Real-World Engineering Challenges & Debugging Journey
Developing StockSense Pro highlighted critical computer vision and software architecture edge cases:

### 1. The "Turntable Trap" & Packaging Geometry Failure
* **Failure Mode:** Model 1 detected rigid soda cans and tins accurately, but consistently missed snack bags (`puffed_food`), pastries (`dessert`), and stationery.
* **Root Cause Diagnosis:** Inspection of RPC's `train2019` revealed that training images showed single products sitting upright on a white turntable. Rigid cylinders (soda cans) look identical upright or tilted on a shelf. However, deformable foil bags change their 2D silhouette entirely when stacked or lying flat. The network learned turntable background edges rather than packaging textures.
* **Resolution:** `src/build_cluttered_dataset.py` was rebuilt to sample exclusively from multi-product checkout scenes (`val2019`). Training bounding boxes increased from 782 to 3,565, driving `puffed_food` counting error down to $\text{MAE} = 0.03$.

### 2. Color Inversion from Color Space Mismatches
* **Failure Mode:** Pepsi bottles and blue yogurt tubs appeared bright orange in early dashboard iterations.
* **Root Cause Diagnosis:** OpenCV processes image buffers in BGR (Blue-Green-Red) order, whereas Pillow and web browsers render standard sRGB. Ingesting raw arrays caused an inversion of red and blue channels:
  $$\begin{bmatrix} R \\ G \\ B \end{bmatrix}_{\text{displayed}} = \begin{bmatrix} B \\ G \\ R \end{bmatrix}_{\text{memory}}$$
* **Resolution:** An explicit bidirectional color alignment step (`cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)` before inference, and `cv2.COLOR_BGR2RGB` before display) resolved the issue.

### 3. Invisible Text via Dark-Mode Streamlit Inheritance
* **Failure Mode:** Custom CSS set a light slate background (`#F8FAFC`), but running Streamlit in dark mode inherited `#FFFFFF` font colors, rendering text invisible.
* **Root Cause Diagnosis:** The contrast ratio dropped to $1.05:1$, violating WCAG accessibility minimums ($4.5:1$):
  $$CR = \frac{1.0 + 0.05}{0.95 + 0.05} = 1.05:1$$
* **Resolution:** Created `.streamlit/config.toml` enforcing `base = "light"`, paired with forced `!important` CSS rules targeting `#0F172A` text colors to restore an accessible $15.38:1$ contrast ratio.

### 4. The 1,076 Validation Error Schema Desynchronization
* **Failure Mode:** Expanding the training configuration from 6 to 10 classes produced 1,076 validation errors on the first build.
* **Root Cause Diagnosis:** Bounding box annotation frequencies for the four newly added categories (`dried_food: 198` + `chocolate: 337` + `candy: 241` + `milk: 300`) summed to exactly 1,076. The annotations were correct; the validation script was still checking against an outdated 6-class predicate ($k < 6$).
* **Resolution:** Updated `src/validate_dataset.py` to read class counts dynamically from `data.yaml`, eliminating schema validation mismatches.

### 5. Division by Zero ($\text{NaN}$) in Test Split Evaluation
* **Failure Mode:** Colab evaluation output printed `Mean Absolute Error: nan` across all categories alongside `RuntimeWarning: invalid value encountered in divide`.
* **Root Cause Diagnosis:** Hardcoded paths (`/content/data/data/dataset`) failed after folder restructuring, yielding an empty test list ($N = 0$). Dividing by zero evaluated to the IEEE 754 indeterminate form $\frac{0}{0} = \text{NaN}$.
* **Resolution:** Replaced static paths with dynamic filesystem discovery (`rglob("images/test")`), restoring reliable metric evaluation across all 68 test images.

### 6. Deprecated UI Parameter Migration
* **Failure Mode:** Streamlit raised layout warnings: `Please replace use_container_width with width`.
* **Root Cause Diagnosis:** Streamlit unified its component sizing API, replacing boolean flags with polymorphic container parameters (`width="stretch"` and `width="content"`).
* **Resolution:** Replaced all legacy `use_container_width=True` calls with `width="stretch"` across the visual inspection columns.

### 7. Virtual Environment Repository Bloat
* **Failure Mode:** Staging the project directory committed the local Windows virtual environment (`venv/`) to Git, introducing thousands of compiled binaries.
* **Root Cause Diagnosis:** Pushing Windows-specific dynamic link libraries (`.dll`, `.pyd`) caused cross-platform ABI mismatches on Streamlit Cloud's Linux runtime (`OSError: Exec format error`).
* **Resolution:** Untracked the virtual environment (`git rm -r --cached venv`), added explicit `.gitignore` rules, and ensured deployments rely entirely on clean builds via `requirements.txt`.

---

## 9. Verification & Automated Unit Testing
The business logic layer is validated using an automated Pytest test suite in `tests/test_stock_logic.py`:
* `test_classify_stock_status_boundaries`: Verifies exact cutoffs for Depleted (0), Low Stock (1–3), and Optimal (>3).
* `test_negative_count_raises_exception`: Confirms negative inventory inputs raise explicit `ValueError` exceptions.
* `test_parse_detections_filtering`: Verifies that low-confidence bounding boxes below $\tau_{\text{conf}}$ are excluded.
* `test_priority_queue_sorting`: Ensures replenishment queues sort strictly by restock urgency.
* `test_material_metadata_integrity`: Confirms all 10 retail classes map to valid packaging substrates.

---

## 10. Submission Checklist
- [x] GitHub Repository structured with naming format `IADAI201-1000406-AJeyaditya`.
- [x] Collaborator `ai.assignments@wacpinternational.org` invited.
- [x] Trained 10-class weights saved in `models/best.pt`.
- [x] Streamlit web application running with 3-column triage board and material profiling.
- [x] Test evaluation documented with confusion matrix and counting MAE.
