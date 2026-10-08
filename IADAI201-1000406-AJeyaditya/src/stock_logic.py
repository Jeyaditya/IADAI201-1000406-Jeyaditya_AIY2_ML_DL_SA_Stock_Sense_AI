"""
StockSense Pro - Retail Stock & Material Intelligence Engine
10-Class Supercategory Architecture & Packaging Substrate Intelligence
"""

from typing import Dict, List, Any

# Complete 10 Target Retail Supercategories
DEFAULT_CATEGORIES = [
    "drink",            # 0
    "puffed_food",      # 1
    "instant_noodles",  # 2
    "dessert",          # 3
    "canned_food",      # 4
    "stationery",       # 5
    "dried_food",       # 6
    "chocolate",        # 7
    "candy",            # 8
    "milk",             # 9
]

# Physical Packaging Substrate Profiling
MATERIAL_METADATA = {
    "canned_food": {
        "material": "Tinplate Steel / Aluminum",
        "form": "Rigid Can",
        "recyclability": "100% Infinitely Recyclable Metal",
        "badge_color": "#475569",
    },
    "drink": {
        "material": "Polyethylene Terephthalate (PET) / Aluminum",
        "form": "Rigid Bottle / Can",
        "recyclability": "Code 1 PET / Aluminum",
        "badge_color": "#0284C7",
    },
    "puffed_food": {
        "material": "Metallized BOPP Foil Barrier Film",
        "form": "Flexible Pouch",
        "recyclability": "Multi-layer Plastic Laminate",
        "badge_color": "#D97706",
    },
    "instant_noodles": {
        "material": "Expanded Polystyrene (EPS) / Kraftboard",
        "form": "Thermal Insulated Tub",
        "recyclability": "Coated Cellulose Composite",
        "badge_color": "#EA580C",
    },
    "dessert": {
        "material": "Solid Bleached Sulfate (SBS) / Polypropylene",
        "form": "Folding Carton / PP Tub",
        "recyclability": "Paperboard / Resin Code 5",
        "badge_color": "#9333EA",
    },
    "stationery": {
        "material": "Cellulose Wood Pulp / Polystyrene",
        "form": "Bound Fiber / Molded Plastic",
        "recyclability": "Biodegradable Pulp / Polymer",
        "badge_color": "#059669",
    },
    "dried_food": {
        "material": "Kraft Polyethylene Barrier Pouch",
        "form": "Flexible Stand-Up Bag",
        "recyclability": "High-Density Polyethylene Film",
        "badge_color": "#78350F",
    },
    "chocolate": {
        "material": "Cold-Seal Metallized Polypropylene Foil",
        "form": "Foil Wrapper",
        "recyclability": "Lightweight Polymer Film",
        "badge_color": "#713F12",
    },
    "candy": {
        "material": "Cast Polypropylene (CPP) Film",
        "form": "Twist / Pouch Wrap",
        "recyclability": "Polymer Film",
        "badge_color": "#BE185D",
    },
    "milk": {
        "material": "High-Density Polyethylene (HDPE) / Tetra Pak",
        "form": "Aseptic Carton / Jug",
        "recyclability": "Aseptic Composite Pulp",
        "badge_color": "#2563EB",
    },
}

STATUS_NAMES = {
    "OOS": "Depleted Facing",
    "LOW": "Critical Low Stock",
    "IN_STOCK": "Optimal Inventory",
}


def classify_stock_status(count: int, low_threshold: int = 3) -> str:
    if count < 0:
        raise ValueError(f"Inventory count cannot be negative: {count}")
    if count == 0:
        return STATUS_NAMES["OOS"]
    elif 1 <= count <= low_threshold:
        return STATUS_NAMES["LOW"]
    else:
        return STATUS_NAMES["IN_STOCK"]


def parse_detections(
    predictions: Any,
    category_names: List[str] = DEFAULT_CATEGORIES,
    conf_threshold: float = 0.20,
) -> Dict[str, int]:
    counts = {cat: 0 for cat in category_names}
    if predictions is None:
        return counts

    result_obj = predictions[0] if isinstance(predictions, list) else predictions
    if not hasattr(result_obj, "boxes") or result_obj.boxes is None:
        return counts

    boxes = result_obj.boxes
    if len(boxes) == 0:
        return counts

    confidences = boxes.conf.cpu().numpy()
    class_indices = boxes.cls.cpu().numpy().astype(int)

    for cls_idx, conf in zip(class_indices, confidences):
        if conf >= conf_threshold:
            if 0 <= cls_idx < len(category_names):
                cat_name = category_names[cls_idx]
                counts[cat_name] += 1

    return counts


def generate_triaged_inventory(
    counts: Dict[str, int],
    low_threshold: int = 3,
    target_capacity: int = 4,
) -> Dict[str, Any]:
    depleted_items = []
    low_stock_items = []
    optimal_items = []

    total_effective_fill = 0
    total_max_capacity = len(counts) * target_capacity

    for cat_name, count in counts.items():
        status = classify_stock_status(count, low_threshold=low_threshold)
        meta = MATERIAL_METADATA.get(
            cat_name,
            {
                "material": "Commercial Packaging Material",
                "form": "Retail Container",
                "recyclability": "Standard Recyclable",
                "badge_color": "#64748B",
            },
        )

        urgency_score = max(0.0, (target_capacity - count) / target_capacity) if target_capacity > 0 else 0.0
        total_effective_fill += min(count, target_capacity)

        item_card = {
            "category": cat_name,
            "display_name": cat_name.replace("_", " ").title(),
            "count": count,
            "status": status,
            "urgency_score": round(urgency_score, 2),
            "material": meta["material"],
            "form": meta["form"],
            "recyclability": meta["recyclability"],
            "badge_color": meta["badge_color"],
        }

        if status == STATUS_NAMES["OOS"]:
            depleted_items.append(item_card)
        elif status == STATUS_NAMES["LOW"]:
            low_stock_items.append(item_card)
        else:
            optimal_items.append(item_card)

    depleted_items.sort(key=lambda x: x["category"])
    low_stock_items.sort(key=lambda x: (-x["urgency_score"], x["category"]))
    optimal_items.sort(key=lambda x: (-x["count"], x["category"]))

    fill_factor = (total_effective_fill / total_max_capacity * 100.0) if total_max_capacity > 0 else 0.0

    return {
        "depleted": depleted_items,
        "low_stock": low_stock_items,
        "optimal": optimal_items,
        "fill_factor": round(fill_factor, 1),
    }
