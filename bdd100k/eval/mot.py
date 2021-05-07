"""BDD100K tracking evaluation with CLEAR MOT metrics."""


SUPER_CLASSES = {
    "HUMAN": ["pedestrian", "rider"],
    "VEHICLE": ["car", "truck", "bus", "train"],
    "BIKE": ["motorcycle", "bicycle"],
}
CLASSES = [c for cs in SUPER_CLASSES.values() for c in cs]
IGNORE_CLASSES = ["trailer", "other person", "other vehicle"]
