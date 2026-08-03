"""Real-model recognition guards over the committed synthetic fixtures."""

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from generate_fixtures import DIGIT_LINES, word_lines
from vprobe.analyze import build_executor
from vprobe.classify import HsvRange
from vprobe.ocr import OcrLine
from vprobe.protocol import (
    AnnulusMask,
    ColorMatchItem,
    ColorMatchResult,
    CropRect,
    MatchItem,
    MatchResult,
    OcrItem,
    OcrResult,
)

FIXTURES = Path(__file__).parent / "fixtures"
MIN_CONFIDENCE = 0.8
BUDGET_SECONDS = 60
UNIVERSAL_RANGE = HsvRange(0, 180, 0, 255, 0, 255)
RED_RANGE = HsvRange(0, 10, 100, 255, 100, 255)
GREEN_RANGE = HsvRange(50, 70, 100, 255, 100, 255)
BLUE_RANGE = HsvRange(110, 130, 100, 255, 100, 255)


@pytest.fixture(scope="module")
def execute():
    return build_executor()


def png(relative: str) -> bytes:
    return (FIXTURES / relative).read_bytes()


def run_ocr(execute, payload: bytes, upscale: bool = False) -> OcrResult:
    (result,) = execute([OcrItem(0, upscale=upscale)], [payload])
    assert isinstance(result, OcrResult)
    return result


def texts_by_row(lines: tuple[OcrLine, ...]) -> list[str]:
    return [line.text for line in sorted(lines, key=lambda line: line.y)]


def test_ocr_reads_digit_lines_top_to_bottom(execute):
    result = run_ocr(execute, png("text/lines.png"))
    assert texts_by_row(result.lines) == list(DIGIT_LINES)
    assert all(line.confidence >= MIN_CONFIDENCE for line in result.lines)


def test_ocr_reads_every_rendered_word(execute):
    result = run_ocr(execute, png("text/words.png"))
    detected = Counter(word for line in result.lines for word in line.text.split())
    expected = Counter(word for row in word_lines() for word in row.split())
    assert detected == expected
    assert all(line.confidence >= MIN_CONFIDENCE for line in result.lines)


def test_ocr_upscale_reads_a_short_digit_strip(execute):
    result = run_ocr(execute, png("text/small-strip.png"), upscale=True)
    assert texts_by_row(result.lines) == ["70", "25"]
    assert all(line.confidence >= MIN_CONFIDENCE for line in result.lines)


def test_ocr_on_a_blank_image_finds_nothing(execute):
    _, encoded = cv2.imencode(".png", np.zeros((300, 300, 3), dtype=np.uint8))
    result = run_ocr(execute, encoded.tobytes())
    assert result.lines == ()


def test_match_locates_the_exact_patch(execute):
    (result,) = execute([MatchItem(png("match/template.png"), 0)], [png("match/canvas.png")])
    assert isinstance(result, MatchResult)
    assert result.found
    assert (result.rect.x, result.rect.y, result.rect.w, result.rect.h) == (123, 234, 48, 48)
    assert result.scale == 1.0
    assert result.score > 0.99


def test_match_locates_a_doubled_patch_at_caller_scale(execute):
    (result,) = execute([MatchItem(png("match/template.png"), 0, scale=2.0)], [png("match/canvas-double.png")])
    assert result.found
    assert (result.rect.x, result.rect.y, result.rect.w, result.rect.h) == (400, 60, 96, 96)
    assert result.scale == 2.0
    assert result.score > 0.99


def test_match_reports_a_miss_with_its_score(execute):
    (result,) = execute([MatchItem(png("match/template.png"), 0)], [png("match/canvas-miss.png")])
    assert not result.found
    assert result.rect is None
    assert result.score is not None and result.score < 0.8


def test_color_match_fractions_on_solid_quadrants(execute):
    top_left = CropRect(0, 0, 100, 100)
    red_on_red, blue_on_red, red_on_white = execute(
        [
            ColorMatchItem(0, (RED_RANGE,), top_left),
            ColorMatchItem(0, (BLUE_RANGE,), top_left),
            ColorMatchItem(0, (RED_RANGE,), CropRect(100, 100, 100, 100)),
        ],
        [png("color/swatches.png")],
    )
    assert red_on_red == ColorMatchResult(fractions=(1.0,))
    assert blue_on_red == ColorMatchResult(fractions=(0.0,))
    assert red_on_white == ColorMatchResult(fractions=(0.0,))


def test_color_match_annulus_follows_the_ring(execute):
    on_ring, inside_ring = execute(
        [
            ColorMatchItem(0, (GREEN_RANGE,), None, AnnulusMask(100, 100, 65, 55)),
            ColorMatchItem(0, (GREEN_RANGE,), None, AnnulusMask(100, 100, 40, 0)),
        ],
        [png("color/ring.png")],
    )
    assert on_ring.fractions[0] >= 0.9
    assert inside_ring == ColorMatchResult(fractions=(0.0,))


def test_batch_results_are_positional_across_ops(execute):
    ocr_result, match, colors = execute(
        [
            OcrItem(0, upscale=True),
            MatchItem(png("match/template.png"), 1),
            ColorMatchItem(2, (RED_RANGE,), CropRect(0, 0, 100, 100)),
        ],
        [png("text/small-strip.png"), png("match/canvas.png"), png("color/swatches.png")],
    )
    assert isinstance(ocr_result, OcrResult)
    assert texts_by_row(ocr_result.lines) == ["70", "25"]
    assert isinstance(match, MatchResult) and match.found
    assert isinstance(colors, ColorMatchResult) and colors.fractions == (1.0,)


def test_large_mixed_batch_meets_the_wall_budget(execute):
    items = [
        OcrItem(1, CropRect(0, 0, 120, 106), True),
        OcrItem(1, CropRect(0, 0, 120, 106), True),
    ]
    ranges = (RED_RANGE, GREEN_RANGE, BLUE_RANGE, UNIVERSAL_RANGE)
    crops = [(x, y) for y in range(0, 720 - 25, 47) for x in range(0, 1280 - 25, 31)]
    for x, y in crops[:407]:
        items.append(ColorMatchItem(0, ranges, CropRect(x, y, 25, 25)))
    started = time.perf_counter()
    results = execute(items, [png("match/canvas-wide.png"), png("text/small-strip.png")])
    elapsed = time.perf_counter() - started
    assert elapsed < BUDGET_SECONDS, f"batch of {len(items)} items took {elapsed:.1f} s"
    assert all(isinstance(result, ColorMatchResult) and result.fractions[3] == 1.0 for result in results[2:])
    for result in results[:2]:
        assert isinstance(result, OcrResult)
        assert texts_by_row(result.lines) == ["70", "25"]
