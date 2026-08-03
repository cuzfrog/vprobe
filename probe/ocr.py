from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil
from typing import Protocol

import cv2
import numpy as np

MIN_OCR_HEIGHT = 320


@dataclass(frozen=True)
class OcrLine:
    text: str
    confidence: float
    x: float
    y: float
    width: float
    height: float


class Recognizer(Protocol):
    def recognize(self, crop_bgr: np.ndarray, upscale: bool = True) -> list[OcrLine]: ...


class RapidRecognizer:
    def __init__(self, gpu: bool = False) -> None:
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion, RapidOCR

        params = {
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": ModelType.MOBILE,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": LangRec.EN,
            "Rec.model_type": ModelType.MOBILE,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            "EngineConfig.onnxruntime.use_dml": gpu,
        }
        self._ocr = RapidOCR(params=params)
        self._lock = threading.Lock()

    def recognize(self, crop_bgr: np.ndarray, upscale: bool = True) -> list[OcrLine]:
        height, width = crop_bgr.shape[:2]
        scale = upscale_factor(height) if upscale else 1
        if scale == 1:
            scaled = crop_bgr
        else:
            scaled = cv2.resize(crop_bgr, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)
        with self._lock:
            output = self._ocr(scaled, use_cls=False)
        return _to_lines(output.boxes, output.txts, output.scores, scale)


def upscale_factor(height: int) -> int:
    if height <= 0:
        return 1
    return max(1, ceil(MIN_OCR_HEIGHT / height))


def _to_lines(
    boxes: Sequence[np.ndarray] | None,
    txts: Sequence[str] | None,
    scores: Sequence[float] | None,
    scale: int,
) -> list[OcrLine]:
    if boxes is None or txts is None or scores is None:
        return []
    lines: list[OcrLine] = []
    for polygon, text, score in zip(boxes, txts, scores, strict=True):
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        lines.append(
            OcrLine(
                text=text,
                confidence=float(score),
                x=min(xs) / scale,
                y=min(ys) / scale,
                width=(max(xs) - min(xs)) / scale,
                height=(max(ys) - min(ys)) / scale,
            )
        )
    return lines
