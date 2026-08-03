from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np

from vprobe import classify
from vprobe.anchor import find_anchor
from vprobe.ocr import RapidRecognizer, Recognizer
from vprobe.protocol import (
    AnnulusMask,
    ColorMatchItem,
    ColorMatchResult,
    CropRect,
    Item,
    MatchItem,
    MatchResult,
    OcrItem,
    OcrResult,
    RectResult,
    Result,
)

DEFAULT_SCALE = 1.0
DEFAULT_MATCH_THRESHOLD = 0.8

Executor = Callable[[Sequence[Item], Sequence[bytes]], Sequence[Result]]

log = logging.getLogger("vprobe.analyze")


class ImageTable:
    def __init__(self, payloads: Sequence[bytes]) -> None:
        self._payloads = payloads
        self._decoded: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def color(self, index: int) -> np.ndarray:
        with self._lock:
            decoded = self._decoded.get(index)
            if decoded is None:
                decoded = _decode_color(self._payloads[index])
                self._decoded[index] = decoded
            return decoded


def execute_match(item: MatchItem, table: ImageTable) -> MatchResult:
    template = _decode_color(item.template, "could not decode template image")
    image = table.color(item.image)
    threshold = DEFAULT_MATCH_THRESHOLD if item.threshold is None else item.threshold
    scale = DEFAULT_SCALE if item.scale is None else item.scale
    match = find_anchor(image, template, scale)
    if match is None:
        return MatchResult(found=False)
    if match.score < threshold:
        return MatchResult(found=False, score=match.score)
    height, width = template.shape[:2]
    rect = RectResult(match.x, match.y, int(round(width * scale)), int(round(height * scale)))
    return MatchResult(found=True, rect=rect, scale=scale, score=match.score)


def execute_ocr(item: OcrItem, recognizer: Recognizer, table: ImageTable) -> OcrResult:
    crop = _slice(table.color(item.image), item.rect)
    return OcrResult(tuple(recognizer.recognize(crop, upscale=item.upscale if item.upscale is not None else True)))


def execute_color_match(item: ColorMatchItem, table: ImageTable) -> ColorMatchResult:
    crop = _slice(table.color(item.image), item.rect)
    mask = None if item.mask is None else _annulus_mask(crop.shape[0], crop.shape[1], item.mask)
    return ColorMatchResult(classify.fractions(crop, item.ranges, mask))


def build_executor(gpu: bool = False) -> Executor:
    cv2.setNumThreads(1)
    start = time.perf_counter()
    recognizer = RapidRecognizer(gpu=gpu)
    log.info("ocr models loaded in %s ms", int(round((time.perf_counter() - start) * 1000)))
    pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

    def execute(items: Sequence[Item], images: Sequence[bytes]) -> Sequence[Result]:
        table = ImageTable(images)
        return tuple(pool.map(partial(_run_item, recognizer=recognizer, table=table), range(len(items)), items))

    return execute


def _run_item(index: int, item: Item, recognizer: Recognizer, table: ImageTable) -> Result:
    op = _op_name(item)
    start = time.perf_counter()
    try:
        result = _execute_item(item, recognizer, table)
    except Exception as exc:
        raise ValueError(f"item {index} ({op}) failed: {exc}") from exc
    ms = int(round((time.perf_counter() - start) * 1000))
    log.debug("item index=%s op=%s ms=%s", index, op, ms)
    return result


def _execute_item(item: Item, recognizer: Recognizer, table: ImageTable) -> Result:
    if isinstance(item, MatchItem):
        return execute_match(item, table)
    if isinstance(item, OcrItem):
        return execute_ocr(item, recognizer, table)
    return execute_color_match(item, table)


def _op_name(item: Item) -> str:
    if isinstance(item, MatchItem):
        return "match"
    if isinstance(item, OcrItem):
        return "ocr"
    return "colorMatch"


def _decode_color(image_bytes: bytes, failure: str = "could not decode png image") -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(failure)
    return image


def _slice(image: np.ndarray, rect: CropRect | None) -> np.ndarray:
    if rect is None:
        return image
    height, width = image.shape[:2]
    if rect.x + rect.w > width or rect.y + rect.h > height:
        raise ValueError(f"rect {rect.x},{rect.y},{rect.w},{rect.h} exceeds image {width}x{height}")
    return image[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]


def _annulus_mask(height: int, width: int, mask: AnnulusMask) -> np.ndarray:
    rows, cols = np.ogrid[:height, :width]
    squared = (cols - mask.cx) ** 2 + (rows - mask.cy) ** 2
    return np.where((squared > mask.inner**2) & (squared <= mask.outer**2), 255, 0).astype(np.uint8)
