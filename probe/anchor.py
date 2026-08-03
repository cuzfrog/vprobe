from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AnchorMatch:
    x: int
    y: int
    scale: float
    score: float


def find_anchor(image: np.ndarray, template: np.ndarray, scale: float) -> AnchorMatch | None:
    template_height, template_width = template.shape[:2]
    image_height, image_width = image.shape[:2]
    scaled_width = int(round(template_width * scale))
    scaled_height = int(round(template_height * scale))
    if scaled_width >= image_width or scaled_height >= image_height or scaled_width < 2 or scaled_height < 2:
        return None
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled_template = cv2.resize(template, (scaled_width, scaled_height), interpolation=interpolation)
    result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
    _, max_score, _, max_loc = cv2.minMaxLoc(result)
    return AnchorMatch(int(max_loc[0]), int(max_loc[1]), scale, float(max_score))
