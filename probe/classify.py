from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HsvRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int


def fractions(pixels_bgr: np.ndarray, ranges: Sequence[HsvRange], mask: np.ndarray | None = None) -> tuple[float, ...]:
    if not ranges or pixels_bgr.size == 0:
        return ()
    hsv = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1] if mask is None else cv2.countNonZero(mask)
    if total == 0:
        return (0.0,) * len(ranges)
    return tuple(_fraction(hsv, bounds, total, mask) for bounds in ranges)


def _fraction(hsv: np.ndarray, bounds: HsvRange, total: int, mask: np.ndarray | None) -> float:
    lower = np.array([bounds.h_min, bounds.s_min, bounds.v_min], dtype=np.uint8)
    upper = np.array([bounds.h_max, bounds.s_max, bounds.v_max], dtype=np.uint8)
    in_range = cv2.inRange(hsv, lower, upper)
    if mask is not None:
        in_range = cv2.bitwise_and(in_range, mask)
    return cv2.countNonZero(in_range) / total
