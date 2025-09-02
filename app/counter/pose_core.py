from __future__ import annotations
import math
from typing import Tuple

# Utility math

def angle_3pt(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Return angle ABC in degrees with B as vertex."""
    try:
        ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        )
        ang = abs(ang)
        if ang > 180:
            ang = 360 - ang
        return ang
    except Exception:
        return 0.0
