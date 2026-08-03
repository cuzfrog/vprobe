import numpy as np

from probe.ocr import OcrLine, _to_lines, upscale_factor


def test_upscale_factor_reaches_min_height():
    assert upscale_factor(90) == 4
    assert upscale_factor(160) == 2
    assert upscale_factor(320) == 1
    assert upscale_factor(1000) == 1
    assert upscale_factor(0) == 1


def test_to_lines_reports_box_top_left_and_size_in_sent_image_px():
    boxes = [np.array([[3.0, 8.0], [104.0, 17.0], [101.0, 62.0], [4.0, 53.0]])]
    assert _to_lines(boxes, ("*70",), (0.99,), 4) == [OcrLine(text="*70", confidence=0.99, x=0.75, y=2.0, width=25.25, height=13.5)]


def test_to_lines_at_scale_one_keeps_raw_box():
    boxes = [np.array([[10.0, 20.0], [60.0, 20.0], [60.0, 44.0], [10.0, 44.0]])]
    assert _to_lines(boxes, ("25",), (0.87,), 1) == [OcrLine(text="25", confidence=0.87, x=10.0, y=20.0, width=50.0, height=24.0)]


def test_to_lines_no_text_detected():
    assert _to_lines(None, None, None, 1) == []
    assert _to_lines([], (), (), 1) == []
