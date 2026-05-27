"""Class-hierarchy bookkeeping.

The 100 fine-grained classes are partitioned into 4 superclasses by contiguous
id range. These helpers convert between three label spaces used across the
pipeline:

    original id   integer in [0, 99]   — the dataset's own label
    router label  integer in [0,  3]   — superclass index (food/flowers/cars/planes)
    local label   integer in [0, 24]   — index within a specialist's superclass
"""

SUPERCLASS_RANGES = {
    "food":    (0, 24),
    "flowers": (25, 49),
    "cars":    (50, 74),
    "planes":  (75, 99),
}

SUPERCLASS_TO_ROUTER_LABEL = {
    "food": 0,
    "flowers": 1,
    "cars": 2,
    "planes": 3,
}

ROUTER_LABEL_TO_SUPERCLASS = {v: k for k, v in SUPERCLASS_TO_ROUTER_LABEL.items()}


def original_id_to_superclass(original_id: int) -> str:
    for superclass, (start, end) in SUPERCLASS_RANGES.items():
        if start <= original_id <= end:
            return superclass
    raise ValueError(f"Original class id {original_id} is not in any superclass range.")


def original_id_to_router_label(original_id: int) -> int:
    return SUPERCLASS_TO_ROUTER_LABEL[original_id_to_superclass(original_id)]


def original_id_to_local(original_id: int) -> tuple[str, int]:
    superclass = original_id_to_superclass(original_id)
    start, _ = SUPERCLASS_RANGES[superclass]
    return superclass, original_id - start


def local_to_original_id(superclass: str, local_label: int) -> int:
    start, _ = SUPERCLASS_RANGES[superclass]
    return start + local_label
