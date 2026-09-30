"""Application-level configuration constants for VexCAM API."""

# Session time-to-live in seconds (1 hour)
SESSION_TTL_SECONDS: int = 3600

# Work-offset string → enum key mapping order
WORK_OFFSET_ORDER: list[str] = ["G54", "G55", "G56", "G57", "G58", "G59"]

# Default safe-retract height added above stock top (mm)
DEFAULT_SAFE_RETRACT_HEIGHT: float = 5.0

# Default stock margin used when no explicit config provided (mm)
DEFAULT_STOCK_MARGIN: float = 5.0

# Supported CAD upload extensions
SUPPORTED_CAD_EXTENSIONS: tuple[str, ...] = (".step", ".stp", ".stl")

# Candidate setup orientations for auto-generation
AUTO_SETUP_CANDIDATES: list[tuple[str, list[float], str, str]] = [
    ("top",    [0.0,   0.0, 0.0], "Top Milling",     "G54"),
    ("bottom", [180.0, 0.0, 0.0], "Bottom Flip",     "G55"),
    ("side_x", [0.0,  90.0, 0.0], "Side Milling +X", "G56"),
    ("side_y", [90.0,  0.0, 0.0], "End Face +Y",     "G57"),
]
