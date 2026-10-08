"""Shared scope and terminology for the Inventory-Aware Evaluation Framework."""

FRAMEWORK_NAME = "Inventory-Aware Evaluation Framework"
COMPONENTS = (
    "item", "sum", "share", "alloc", "scaled", "cap", "asym", "delta", "zero",
)
COMPONENT_LABELS = {
    "item": "Item", "sum": "Sum", "share": "Share", "alloc": "Allocation",
    "scaled": "Scaled", "cap": "Cap",
    "asym": "Asymmetric", "delta": "Delta", "zero": "Zero",
}
EXCLUDED_COMPONENTS = {"weighted_item", "rel", "robust", "sparse", "total", "neg", "tv", "int"}
