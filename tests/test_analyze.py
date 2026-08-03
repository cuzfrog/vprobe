import logging
import re
import threading
import time

import cv2
import numpy as np
import pytest

from probe.analyze import (
    DEFAULT_SCALE,
    ImageTable,
    build_executor,
    execute_color_match,
    execute_match,
    execute_ocr,
)
from probe.classify import HsvRange
from probe.classify import fractions as unmasked_fractions
from probe.ocr import OcrLine
from probe.protocol import (
    AnnulusMask,
    ColorMatchItem,
    ColorMatchResult,
    CropRect,
    MatchItem,
    MatchResult,
    OcrItem,
    OcrResult,
    RectResult,
)

TEMPLATE = np.random.default_rng(0).integers(40, 220, size=(26, 25), dtype=np.uint8)
TEMPLATE_H, TEMPLATE_W = TEMPLATE.shape
RED_RANGE = HsvRange(0, 5, 200, 255, 200, 255)
GREEN_RANGE = HsvRange(55, 65, 200, 255, 200, 255)


def png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def table_of(*images: np.ndarray) -> ImageTable:
    return ImageTable([png_bytes(image) for image in images])


def bgr_pixel(h: int, s: int, v: int) -> np.ndarray:
    return cv2.cvtColor(np.full((1, 1, 3), (h, s, v), dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]


def scene(scale: float = 1.0, x: int = 80, y: int = 50) -> np.ndarray:
    rw, rh = int(round(TEMPLATE_W * scale)), int(round(TEMPLATE_H * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    block = cv2.resize(TEMPLATE, (rw, rh), interpolation=interp)
    canvas = np.full((200, 300, 3), 12, dtype=np.uint8)
    canvas[y : y + rh, x : x + rw] = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
    return canvas


def noise_scene() -> np.ndarray:
    gray = np.random.default_rng(1).integers(0, 30, size=(200, 300), dtype=np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def striped_cell(red_rows: int, green_rows: int) -> np.ndarray:
    cell = np.zeros((red_rows + green_rows, 10, 3), dtype=np.uint8)
    if red_rows:
        cell[:red_rows] = bgr_pixel(0, 255, 255)
    if green_rows:
        cell[red_rows:] = bgr_pixel(60, 255, 255)
    return cell


def tricolor_canvas() -> np.ndarray:
    canvas = np.zeros((21, 21, 3), dtype=np.uint8)
    canvas[0:7] = bgr_pixel(0, 255, 255)
    canvas[7:14] = bgr_pixel(60, 255, 255)
    return canvas


def rep_icon(color: np.ndarray, side: int = 33) -> np.ndarray:
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    margin = 3
    canvas[margin : side - margin, margin : side - margin] = color
    center = side // 2
    bar_half = 3
    canvas[center - bar_half : center + bar_half + 1, side // 4 : side - side // 4] = (255, 255, 255)
    return canvas


def icon_scene(icon: np.ndarray, x: int = 40, y: int = 30) -> np.ndarray:
    canvas = np.zeros((120, 120, 3), dtype=np.uint8)
    h, w = icon.shape[:2]
    canvas[y : y + h, x : x + w] = icon
    return canvas


class RecordingRecognizer:
    def __init__(self, lines):
        self._lines = lines
        self.calls = []

    def recognize(self, crop_bgr, upscale=True):
        self.calls.append((crop_bgr.shape, upscale))
        return self._lines


@pytest.fixture(autouse=True)
def fake_recognizer(monkeypatch):
    monkeypatch.setattr("probe.analyze.RapidRecognizer", lambda gpu=False: RecordingRecognizer([]))


def test_match_finds_template_at_image_relative_rect():
    result = execute_match(MatchItem(template=png_bytes(TEMPLATE), image=0), table_of(scene(1.0, 80, 50)))
    assert result.found is True
    assert result.rect == RectResult(80, 50, TEMPLATE_W, TEMPLATE_H)
    assert result.scale == pytest.approx(1.0, abs=0.03)
    assert result.score > 0.9


def test_match_uses_configured_scale_and_image_relative_rect():
    result = execute_match(MatchItem(template=png_bytes(TEMPLATE), image=0, scale=1.5), table_of(scene(1.5, 30, 70)))
    assert result.found is True
    assert result.scale == 1.5
    assert (result.rect.x, result.rect.y) == (30, 70)
    assert result.rect.w == int(round(TEMPLATE_W * 1.5))
    assert result.rect.h == int(round(TEMPLATE_H * 1.5))


def test_match_absent_template_reports_low_score_without_rect():
    result = execute_match(MatchItem(template=png_bytes(TEMPLATE), image=0), table_of(noise_scene()))
    assert result.found is False
    assert result.rect is None
    assert result.scale is None
    assert result.score < 0.8


def test_match_below_threshold_reports_not_found_with_score():
    gray = cv2.cvtColor(scene(1.0, 80, 50), cv2.COLOR_BGR2GRAY)
    blurred = cv2.cvtColor(cv2.GaussianBlur(gray, (15, 15), 0), cv2.COLOR_GRAY2BGR)
    item = MatchItem(template=png_bytes(TEMPLATE), image=0, threshold=0.95)
    result = execute_match(item, table_of(blurred))
    assert result.found is False
    assert result.score < 0.95


def test_match_red_template_finds_same_color_icon():
    red = bgr_pixel(0, 255, 255)
    template = rep_icon(red)
    result = execute_match(MatchItem(template=png_bytes(template), image=0), table_of(icon_scene(rep_icon(red))))
    assert result.found is True
    assert result.score > 0.9


def test_match_red_template_rejects_luminance_twin_gray_icon():
    red = bgr_pixel(0, 255, 255)
    gray = bgr_pixel(0, 0, 150)
    template = rep_icon(red)
    result = execute_match(MatchItem(template=png_bytes(template), image=0), table_of(icon_scene(rep_icon(gray))))
    assert result.found is False
    assert result.score < 0.8


def test_match_defaults_absent_scale_to_one(monkeypatch):
    calls = []

    def fake_find_anchor(image, template, scale):
        calls.append(scale)
        return None

    monkeypatch.setattr("probe.analyze.find_anchor", fake_find_anchor)
    result = execute_match(MatchItem(template=png_bytes(TEMPLATE), image=0), table_of(scene()))
    assert result == MatchResult(found=False)
    assert calls == [DEFAULT_SCALE]


def test_match_passes_explicit_scale_to_anchor(monkeypatch):
    calls = []

    def fake_find_anchor(image, template, scale):
        calls.append(scale)
        return None

    monkeypatch.setattr("probe.analyze.find_anchor", fake_find_anchor)
    execute_match(MatchItem(template=png_bytes(TEMPLATE), image=0, scale=1.1, threshold=0.99), table_of(scene()))
    assert calls == [1.1]


def test_match_rejects_undecodable_template():
    with pytest.raises(ValueError, match="could not decode template image"):
        execute_match(MatchItem(template=b"garbage", image=0), table_of(scene()))


def test_match_rejects_undecodable_image():
    with pytest.raises(ValueError, match="could not decode png image"):
        execute_match(MatchItem(template=png_bytes(TEMPLATE), image=0), ImageTable([b"garbage"]))


def test_ocr_defaults_upscale_to_true_and_passes_lines_through():
    lines = [OcrLine(text="70", confidence=0.9, x=1.0, y=2.0, width=10.0, height=5.0)]
    recognizer = RecordingRecognizer(lines)
    result = execute_ocr(OcrItem(image=0), recognizer, table_of(np.zeros((40, 100, 3), dtype=np.uint8)))
    assert result == OcrResult(lines=tuple(lines))
    assert recognizer.calls == [((40, 100, 3), True)]


def test_ocr_passes_explicit_upscale_false():
    recognizer = RecordingRecognizer([])
    execute_ocr(OcrItem(image=0, upscale=False), recognizer, table_of(np.zeros((40, 100, 3), dtype=np.uint8)))
    assert recognizer.calls[0][1] is False


def test_ocr_slices_the_rect_before_recognition():
    recognizer = RecordingRecognizer([])
    execute_ocr(OcrItem(image=0, rect=CropRect(10, 5, 20, 30)), recognizer, table_of(np.zeros((40, 100, 3), dtype=np.uint8)))
    assert recognizer.calls == [((30, 20, 3), True)]


def test_ocr_rejects_rect_beyond_image_bounds():
    with pytest.raises(ValueError, match="exceeds image"):
        execute_ocr(OcrItem(image=0, rect=CropRect(90, 0, 20, 10)), RecordingRecognizer([]), table_of(np.zeros((40, 100, 3), dtype=np.uint8)))


def test_ocr_rejects_undecodable_image():
    with pytest.raises(ValueError, match="could not decode png image"):
        execute_ocr(OcrItem(image=0), RecordingRecognizer([]), ImageTable([b"garbage"]))


def test_color_match_measures_fractions_positionally():
    item = ColorMatchItem(image=0, ranges=(RED_RANGE, GREEN_RANGE))
    assert execute_color_match(item, table_of(striped_cell(4, 6))) == ColorMatchResult(fractions=(0.4, 0.6))


def test_color_match_slices_the_rect_before_sampling():
    canvas = np.zeros((20, 20, 3), dtype=np.uint8)
    canvas[5:15, 5:15] = bgr_pixel(0, 255, 255)
    item = ColorMatchItem(image=0, ranges=(RED_RANGE,), rect=CropRect(5, 5, 10, 10))
    assert execute_color_match(item, table_of(canvas)) == ColorMatchResult(fractions=(1.0,))


def test_color_match_mask_normalizes_by_mask_pixels():
    item = ColorMatchItem(image=0, ranges=(RED_RANGE, GREEN_RANGE), mask=AnnulusMask(cx=10, cy=10, outer=2, inner=0))
    result = execute_color_match(item, table_of(tricolor_canvas()))
    assert result.fractions == (0.0, 1.0)


def test_color_match_masked_fractions_equal_legacy_black_fill_factor_correction():
    canvas = tricolor_canvas()
    outer, inner, center = 8, 3, 10
    rows, cols = np.ogrid[:21, :21]
    squared = (cols - center) ** 2 + (rows - center) ** 2
    inside = (squared > inner**2) & (squared <= outer**2)
    item = ColorMatchItem(image=0, ranges=(RED_RANGE, GREEN_RANGE), mask=AnnulusMask(center, center, outer, inner))
    result = execute_color_match(item, table_of(canvas))
    filled = canvas.copy()
    filled[~inside] = 0
    factor = (21 * 21) / int(inside.sum())
    legacy = tuple(fraction * factor for fraction in unmasked_fractions(filled, (RED_RANGE, GREEN_RANGE)))
    assert result.fractions == pytest.approx(legacy, abs=1e-12)


def test_color_match_rejects_rect_beyond_image_bounds():
    item = ColorMatchItem(image=0, ranges=(RED_RANGE,), rect=CropRect(5, 5, 10, 10))
    with pytest.raises(ValueError, match="exceeds image"):
        execute_color_match(item, table_of(striped_cell(4, 6)))


def test_color_match_rejects_undecodable_image():
    with pytest.raises(ValueError, match="could not decode png image"):
        execute_color_match(ColorMatchItem(image=0, ranges=(RED_RANGE,)), ImageTable([b"garbage"]))


def test_image_table_decodes_each_index_once(monkeypatch):
    payloads = [png_bytes(striped_cell(4, 6))]
    decodes = []
    decode = cv2.imdecode

    def counting(buffer, flags):
        decodes.append(1)
        return decode(buffer, flags)

    monkeypatch.setattr("probe.analyze.cv2.imdecode", counting)
    execute = build_executor()
    items = [
        ColorMatchItem(image=0, ranges=(RED_RANGE,)),
        ColorMatchItem(image=0, ranges=(GREEN_RANGE,)),
        ColorMatchItem(image=0, ranges=(RED_RANGE,)),
    ]
    results = execute(items, payloads)
    assert decodes == [1]
    assert [result.fractions for result in results] == [(0.4,), (0.6,), (0.4,)]


def test_build_executor_dispatches_per_op_and_keeps_item_order():
    execute = build_executor()
    items = [
        ColorMatchItem(image=0, ranges=(RED_RANGE, GREEN_RANGE)),
        MatchItem(template=png_bytes(TEMPLATE), image=1),
    ]
    results = execute(items, [png_bytes(striped_cell(4, 6)), png_bytes(scene())])
    assert results[0] == ColorMatchResult(fractions=(0.4, 0.6))
    assert isinstance(results[1], MatchResult)
    assert results[1].found is True


def test_build_executor_decodes_a_shared_match_image_once(monkeypatch):
    decodes = []
    decode = cv2.imdecode

    def counting(buffer, flags):
        decodes.append(1)
        return decode(buffer, flags)

    monkeypatch.setattr("probe.analyze.cv2.imdecode", counting)
    execute = build_executor()
    items = [MatchItem(template=png_bytes(TEMPLATE), image=0) for _ in range(3)]
    results = execute(items, [png_bytes(scene(1.0, 80, 50))])
    assert [result.found for result in results] == [True, True, True]
    assert len(decodes) == 4


def test_build_executor_runs_items_across_multiple_threads(monkeypatch):
    idents = []
    lock = threading.Lock()

    def fake_find_anchor(image, template, scale):
        with lock:
            idents.append(threading.get_ident())
        time.sleep(0.02)
        return None

    monkeypatch.setattr("probe.analyze.find_anchor", fake_find_anchor)
    execute = build_executor()
    items = [MatchItem(template=png_bytes(TEMPLATE), image=0) for _ in range(4)]
    results = execute(items, [png_bytes(scene())])
    assert [result for result in results] == [MatchResult(found=False)] * 4
    assert len(set(idents)) >= 2


def test_build_executor_wraps_failures_with_item_index_and_op():
    execute = build_executor()
    items = [
        ColorMatchItem(image=0, ranges=(RED_RANGE,)),
        MatchItem(template=b"garbage", image=0),
    ]
    expected = re.escape("item 1 (match) failed: could not decode template image")
    with pytest.raises(ValueError, match=expected):
        execute(items, [png_bytes(striped_cell(4, 6))])


def test_build_executor_names_ocr_in_failure_context():
    execute = build_executor()
    with pytest.raises(ValueError, match=re.escape("item 0 (ocr) failed: could not decode png image")):
        execute([OcrItem(image=0)], [b"garbage"])


def test_build_executor_logs_per_item_timing_at_debug(caplog):
    execute = build_executor()
    with caplog.at_level(logging.DEBUG, logger="probe.analyze"):
        execute([ColorMatchItem(image=0, ranges=(RED_RANGE,))], [png_bytes(striped_cell(4, 6))])
    assert any(record.message.startswith("item index=0 op=colorMatch ms=") for record in caplog.records)


def test_build_executor_builds_recognizer_eagerly_and_logs_load_time(monkeypatch, caplog):
    builds = []

    def factory(gpu=False):
        builds.append(gpu)
        return RecordingRecognizer([])

    monkeypatch.setattr("probe.analyze.RapidRecognizer", factory)
    with caplog.at_level(logging.INFO, logger="probe.analyze"):
        build_executor()
    assert builds == [False]
    assert any(record.message.startswith("ocr models loaded in") and record.message.endswith(" ms") for record in caplog.records)


def test_build_executor_passes_gpu_flag_to_recognizer(monkeypatch):
    builds = []

    def factory(gpu=False):
        builds.append(gpu)
        return RecordingRecognizer([])

    monkeypatch.setattr("probe.analyze.RapidRecognizer", factory)
    build_executor(gpu=True)
    assert builds == [True]
