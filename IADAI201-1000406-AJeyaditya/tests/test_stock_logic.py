"""
StockSense Pro - Unit Test Suite for Retail Stock Logic
Executes rigorous boundary checks and edge case validation.
"""

import pytest
from src.stock_logic import (
    classify_stock_status,
    generate_inventory_report,
    get_restock_priority_list,
    STATUS_COLORS,
)


def test_classify_stock_status_boundaries():
    """Verify exact threshold boundary transitions."""
    # Out of Stock
    assert classify_stock_status(0) == "Out of Stock"

    # Low Stock (1 to 3)
    assert classify_stock_status(1) == "Low Stock"
    assert classify_stock_status(2) == "Low Stock"
    assert classify_stock_status(3) == "Low Stock"

    # In Stock (> 3)
    assert classify_stock_status(4) == "In Stock"
    assert classify_stock_status(10) == "In Stock"


def test_classify_negative_count_raises_error():
    """Verify negative inventory counts raise an explicit ValueError."""
    with pytest.raises(ValueError):
        classify_stock_status(-1)


def test_custom_threshold_configuration():
    """Verify that configurable threshold parameters behave correctly."""
    # If store manager raises low_threshold to 5
    assert classify_stock_status(4, low_threshold=5) == "Low Stock"
    assert classify_stock_status(5, low_threshold=5) == "Low Stock"
    assert classify_stock_status(6, low_threshold=5) == "In Stock"


def test_generate_inventory_report_structure():
    """Verify report formatting, colors, and urgency sorting."""
    sample_counts = {
        "drink": 0,
        "dessert": 5,
        "canned_food": 2,
    }

    report = generate_inventory_report(sample_counts, low_threshold=3, target_capacity=4)

    assert len(report) == 3

    # Primary urgency item must be 'drink' (count 0)
    assert report[0]["category"] == "drink"
    assert report[0]["status"] == "Out of Stock"
    assert report[0]["color"] == STATUS_COLORS["Out of Stock"]
    assert report[0]["urgency_score"] == 1.0

    # Second urgency item must be 'canned_food' (count 2)
    assert report[1]["category"] == "canned_food"
    assert report[1]["status"] == "Low Stock"
    assert report[1]["color"] == STATUS_COLORS["Low Stock"]

    # Final item must be 'dessert' (count 5)
    assert report[2]["category"] == "dessert"
    assert report[2]["status"] == "In Stock"
    assert report[2]["color"] == STATUS_COLORS["In Stock"]
    assert report[2]["urgency_score"] == 0.0


def test_get_restock_priority_list_filters_in_stock():
    """Verify that In Stock items are excluded from the restocking queue."""
    sample_counts = {
        "drink": 0,
        "dessert": 6,
        "puffed_food": 2,
    }
    report = generate_inventory_report(sample_counts)
    priority_queue = get_restock_priority_list(report)

    categories_in_queue = [item["category"] for item in priority_queue]
    assert "drink" in categories_in_queue
    assert "puffed_food" in categories_in_queue
    assert "dessert" not in categories_in_queue
