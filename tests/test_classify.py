import cv2
import numpy as np

from vprobe.classify import HsvRange, fractions

RED = HsvRange(0, 5, 200, 255, 200, 255)
GREEN = HsvRange(55, 65, 200, 255, 200, 255)


def bgr_pixel(h: int, s: int, v: int) -> np.ndarray:
    return cv2.cvtColor(np.full((1, 1, 3), (h, s, v), dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]


def solid_cell(h: int, s: int, v: int, side: int = 10) -> np.ndarray:
    return np.full((side, side, 3), bgr_pixel(h, s, v), dtype=np.uint8)


def striped_cell(red_rows: int, green_rows: int, black_rows: int = 0) -> np.ndarray:
    cell = np.zeros((red_rows + green_rows + black_rows, 10, 3), dtype=np.uint8)
    if red_rows:
        cell[:red_rows] = bgr_pixel(0, 255, 255)
    if green_rows:
        cell[red_rows : red_rows + green_rows] = bgr_pixel(60, 255, 255)
    return cell


def test_solid_color_is_fully_inside_its_window_and_outside_others():
    assert fractions(solid_cell(0, 255, 255), (RED, GREEN)) == (1.0, 0.0)
    assert fractions(solid_cell(60, 255, 255), (RED, GREEN)) == (0.0, 1.0)


def test_color_outside_every_window_measures_zero():
    assert fractions(solid_cell(0, 0, 0), (RED, GREEN)) == (0.0, 0.0)


def test_range_built_from_canonical_hsv_matches_exactly():
    pixel = np.full((1, 1, 3), (200, 100, 50), dtype=np.uint8)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    assert fractions(pixel, (HsvRange(int(h), int(h), int(s), int(s), int(v), int(v)),)) == (1.0,)


def test_partial_mask_counts_only_matching_rows():
    assert fractions(striped_cell(4, 6), (RED, GREEN)) == (0.4, 0.6)
    assert fractions(striped_cell(3, 3, 4), (RED, GREEN)) == (0.3, 0.3)


def test_fractions_are_positional_to_ranges():
    cell = solid_cell(0, 255, 255)
    assert fractions(cell, (GREEN, RED)) == (0.0, 1.0)


def test_no_ranges_returns_empty_tuple():
    assert fractions(solid_cell(0, 255, 255), ()) == ()


def test_empty_pixels_returns_empty_tuple():
    assert fractions(np.zeros((0, 0, 3), dtype=np.uint8), (RED,)) == ()


def test_masked_fractions_normalize_by_mask_pixel_count():
    cell = striped_cell(4, 6)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, :] = 255
    assert fractions(cell, (RED, GREEN), mask) == (0.8, 0.2)


def test_full_mask_matches_unmasked_fractions():
    cell = striped_cell(4, 6)
    mask = np.full((10, 10), 255, dtype=np.uint8)
    assert fractions(cell, (RED, GREEN), mask) == fractions(cell, (RED, GREEN))


def test_zero_pixel_mask_yields_zero_fractions():
    cell = striped_cell(4, 6)
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert fractions(cell, (RED, GREEN), mask) == (0.0, 0.0)
