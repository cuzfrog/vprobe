"""Regenerate the synthetic recognition fixtures committed under tests/fixtures."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 20260801
FIXTURES = Path(__file__).parent / "fixtures"
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/DejaVuSans-Bold.ttf",
)
DIGIT_LINES = ("70", "25", "12345", "4096")
WORDS = (
    "alpha", "bravo", "delta", "echo", "foxtrot", "golf", "hotel", "india",
    "juliet", "kilo", "lima", "mike", "oscar", "papa", "romeo", "sierra",
    "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu",
)
TEMPLATE_X = 123
TEMPLATE_Y = 234
TEMPLATE_SIZE = 48
DOUBLE_X = 400
DOUBLE_Y = 60
DOUBLE_SIZE = 96
SWATCH_SIZE = 100
RED_BGR = (0, 0, 255)
GREEN_BGR = (0, 255, 0)
BLUE_BGR = (255, 0, 0)
WHITE_BGR = (255, 255, 255)


def word_lines() -> tuple[str, ...]:
    picked = random.Random(SEED).sample(WORDS, 9)
    return tuple(" ".join(picked[row * 3 : row * 3 + 3]) for row in range(3))


def main() -> None:
    args = argparse.ArgumentParser(prog="generate_fixtures")
    args.add_argument("--font", default=None)
    options = args.parse_args()
    font_path = resolve_font(options.font)
    write_text(FIXTURES / "text" / "lines.png", DIGIT_LINES, font_path, size=48, width=480, height=320)
    write_text(FIXTURES / "text" / "words.png", word_lines(), font_path, size=40, width=640, height=300)
    write_text(FIXTURES / "text" / "small-strip.png", ("70", "25"), font_path, size=24, width=120, height=106, top=6, step=48)
    write_match()
    write_color()
    for path in sorted(FIXTURES.rglob("*.png")):
        print(path.relative_to(FIXTURES.parent.parent), path.stat().st_size)


def resolve_font(explicit: str | None) -> Path:
    candidates = (Path(explicit),) if explicit is not None else tuple(Path(path) for path in FONT_CANDIDATES)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise SystemExit(f"no bold TTF font found (searched: {searched}); pass --font PATH")


def write_text(
    path: Path, lines: tuple[str, ...], font_path: Path, size: int,
    width: int, height: int, top: int | None = None, step: int | None = None,
) -> None:
    font = ImageFont.truetype(str(font_path), size)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = top if top is not None else int(size * 0.6)
    row_step = step if step is not None else int(size * 1.8)
    for line in lines:
        draw.text((int(size * 0.5), y), line, font=font, fill="black")
        y += row_step
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_match() -> None:
    rng = np.random.default_rng(SEED)
    canvas = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    template = canvas[TEMPLATE_Y : TEMPLATE_Y + TEMPLATE_SIZE, TEMPLATE_X : TEMPLATE_X + TEMPLATE_SIZE].copy()
    doubled = canvas.copy()
    doubled[DOUBLE_Y : DOUBLE_Y + DOUBLE_SIZE, DOUBLE_X : DOUBLE_X + DOUBLE_SIZE] = cv2.resize(
        template, (DOUBLE_SIZE, DOUBLE_SIZE), interpolation=cv2.INTER_LINEAR,
    )
    miss = np.random.default_rng(SEED + 1).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    wide = np.random.default_rng(SEED + 2).integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    match_dir = FIXTURES / "match"
    match_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(match_dir / "canvas.png"), canvas)
    cv2.imwrite(str(match_dir / "canvas-double.png"), doubled)
    cv2.imwrite(str(match_dir / "canvas-miss.png"), miss)
    cv2.imwrite(str(match_dir / "canvas-wide.png"), wide)
    cv2.imwrite(str(match_dir / "template.png"), template)


def write_color() -> None:
    swatches = np.zeros((SWATCH_SIZE * 2, SWATCH_SIZE * 2, 3), dtype=np.uint8)
    swatches[:SWATCH_SIZE, :SWATCH_SIZE] = RED_BGR
    swatches[:SWATCH_SIZE, SWATCH_SIZE:] = GREEN_BGR
    swatches[SWATCH_SIZE:, :SWATCH_SIZE] = BLUE_BGR
    swatches[SWATCH_SIZE:, SWATCH_SIZE:] = WHITE_BGR
    ring = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(ring, (100, 100), 60, GREEN_BGR, thickness=10)
    color_dir = FIXTURES / "color"
    color_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(color_dir / "swatches.png"), swatches)
    cv2.imwrite(str(color_dir / "ring.png"), ring)


if __name__ == "__main__":
    main()
