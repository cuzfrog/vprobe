import cv2
import numpy as np

from vprobe.anchor import find_anchor

TEMPLATE = np.random.default_rng(0).integers(40, 220, size=(26, 25), dtype=np.uint8)


def _canvas_with_template(scale, x, y):
    h, w = TEMPLATE.shape
    rw, rh = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    block = cv2.resize(TEMPLATE, (rw, rh), interpolation=interp)
    canvas = np.full((200, 300), 12, dtype=np.uint8)
    canvas[y : y + rh, x : x + rw] = block
    return canvas


def test_finds_template_at_scale_one():
    canvas = _canvas_with_template(1.0, 80, 50)
    match = find_anchor(canvas, TEMPLATE, 1.0)
    assert match is not None
    assert (match.x, match.y) == (80, 50)
    assert match.scale == 1.0
    assert match.score > 0.99


def test_finds_template_at_configured_scale():
    canvas = _canvas_with_template(1.5, 30, 70)
    match = find_anchor(canvas, TEMPLATE, 1.5)
    assert match is not None
    assert (match.x, match.y) == (30, 70)
    assert match.scale == 1.5
    assert match.score > 0.99


def test_reports_low_score_when_template_absent():
    canvas = np.random.default_rng(1).integers(0, 30, size=(200, 300), dtype=np.uint8)
    match = find_anchor(canvas, TEMPLATE, 1.0)
    assert match is not None
    assert match.score < 0.8


def test_returns_none_when_scaled_template_exceeds_image():
    canvas = np.full((20, 20), 12, dtype=np.uint8)
    assert find_anchor(canvas, TEMPLATE, 1.0) is None


def test_returns_none_when_scaled_template_degenerates():
    canvas = np.full((200, 300), 12, dtype=np.uint8)
    assert find_anchor(canvas, TEMPLATE, 0.01) is None


def test_echoes_the_configured_scale():
    canvas = _canvas_with_template(1.1, 40, 60)
    match = find_anchor(canvas, TEMPLATE, 1.1)
    assert match is not None
    assert match.scale == 1.1


def test_runs_exactly_one_match_template(monkeypatch):
    calls = []
    match_template = cv2.matchTemplate

    def counting(image, templ, method):
        calls.append(1)
        return match_template(image, templ, method)

    monkeypatch.setattr("vprobe.anchor.cv2.matchTemplate", counting)
    canvas = _canvas_with_template(1.0, 80, 50)
    match = find_anchor(canvas, TEMPLATE, 1.0)
    assert match is not None
    assert len(calls) == 1
