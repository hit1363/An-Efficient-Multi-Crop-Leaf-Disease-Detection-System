"""Canonical labels and explicit aliases for the multi-crop benchmark.

The label map intentionally uses exact source folder names. Unknown labels are
rejected instead of being guessed with fuzzy matching, so every taxonomy
decision is auditable.
"""

_CANONICAL_LABELS = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Banana___Cordana",
    "Banana___healthy",
    "Banana___Pestalotiopsis",
    "Banana___Sigatoka",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Guava___Anthracnose",
    "Guava___healthy",
    "Guava___Insect_bite",
    "Guava___Scorch",
    "Guava___Yellowing",
    "Invalid",
    "Jackfruit___Algal_Leaf_Spot",
    "Jackfruit___Black_Spot",
    "Jackfruit___healthy",
    "Mango___Anthracnose",
    "Mango___Bacterial_Canker",
    "Mango___Cutting_Weevil",
    "Mango___Die_Back",
    "Mango___Gall_Midge",
    "Mango___healthy",
    "Mango___Powdery_Mildew",
    "Mango___Sooty_Mould",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_Blight",
    "Potato___healthy",
    "Potato___Late_Blight",
    "Pumpkin___Bacterial_Leaf_Spot",
    "Pumpkin___Downy_Mildew",
    "Pumpkin___healthy",
    "Pumpkin___Mosaic",
    "Pumpkin___Powdery_Mildew",
    "Rice___Brown_Spot",
    "Rice___healthy",
    "Rice___Hispa",
    "Rice___Leaf_Blast",
    "Strawberry___healthy",
    "Strawberry___Leaf_Scorch",
    "Sugarcane___Bacterial_Blights",
    "Sugarcane___healthy",
    "Sugarcane___Mosaic",
    "Sugarcane___Red_Rot",
    "Sugarcane___Rust",
    "Sugarcane___Yellowing",
    "Tomato___Bacterial_Spot",
    "Tomato___Early_Blight",
    "Tomato___healthy",
    "Tomato___Late_Blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Wheat___Brown_Rust",
    "Wheat___healthy",
    "Wheat___Yellow_Rust",
)
# Keep the reserved reject/invalid category at output index 71 (class 72).
CANONICAL_CLASSES = tuple(
    [label for label in sorted(_CANONICAL_LABELS) if label != "Invalid"] + ["Invalid"]
)

# Explicit spelling/case/separator variants seen in the source collection.
LABEL_ALIASES = {
    "Apple___Healthy": "Apple___healthy",
    "Banana___Healthy": "Banana___healthy",
    "Cherry_(including_sour)___Healthy": "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Healthy": "Corn_(maize)___healthy",
    "Grape___Healthy": "Grape___healthy",
    "Guava___Healthy": "Guava___healthy",
    "Jackfruit___Healthy": "Jackfruit___healthy",
    "Mango__Anthracnose": "Mango___Anthracnose",
    "Mango___Healthy": "Mango___healthy",
    "Peach___Healthy": "Peach___healthy",
    "Pepper,_bell___Healthy": "Pepper,_bell___healthy",
    "Potato___Healthy": "Potato___healthy",
    "Pumpkin___Healthy": "Pumpkin___healthy",
    "Rice___Healthy": "Rice___healthy",
    "Sugarcane___Healthy": "Sugarcane___healthy",
    "Strawberry___Leaf_scorch": "Strawberry___Leaf_Scorch",
    "Tomato___Bacterial_spot": "Tomato___Bacterial_Spot",
    "Tomato___Early_blight": "Tomato___Early_Blight",
    "Tomato___Late_blight": "Tomato___Late_Blight",
    "Tomato___Healthy": "Tomato___healthy",
    "Wheat___Healthy": "Wheat___healthy",
}

EXPECTED_CLASS_COUNT = 72
_CASEFOLD_LABELS = {}
for _label in CANONICAL_CLASSES:
    _CASEFOLD_LABELS.setdefault(_label.casefold(), []).append(_label)


def canonicalize_label(source_label):
    """Map a source folder label to its canonical label or reject it."""
    if source_label in LABEL_ALIASES:
        return LABEL_ALIASES[source_label]
    if source_label in CANONICAL_CLASSES:
        return source_label
    matches = _CASEFOLD_LABELS.get(str(source_label).casefold(), [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous case-insensitive source label: {source_label!r}")
    raise ValueError(f"Unknown source label: {source_label!r}")


def taxonomy_document():
    """Return a JSON-serializable taxonomy record for generated datasets."""
    return {
        "version": 1,
        "expected_class_count": EXPECTED_CLASS_COUNT,
        "canonical_classes": list(CANONICAL_CLASSES),
        "source_aliases": dict(sorted(LABEL_ALIASES.items())),
        "unknown_label_policy": "error; case-insensitive aliases must be unambiguous",
    }
