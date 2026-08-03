from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO

import msgpack

from probe.classify import HsvRange
from probe.ocr import OcrLine

PROTOCOL_VERSION = 5
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

_HEADER = struct.Struct(">I")


class ProtocolError(ValueError):
    pass


class BatchParseError(ValueError):
    def __init__(self, batch_id: int | None, message: str) -> None:
        super().__init__(message)
        self.batch_id = batch_id


@dataclass(frozen=True)
class MatchItem:
    template: bytes
    image: int
    scale: float | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class CropRect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class AnnulusMask:
    cx: int
    cy: int
    outer: int
    inner: int


@dataclass(frozen=True)
class OcrItem:
    image: int
    rect: CropRect | None = None
    upscale: bool | None = None


@dataclass(frozen=True)
class ColorMatchItem:
    image: int
    ranges: tuple[HsvRange, ...]
    rect: CropRect | None = None
    mask: AnnulusMask | None = None


Item = MatchItem | OcrItem | ColorMatchItem


@dataclass(frozen=True)
class RectResult:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class MatchResult:
    found: bool
    rect: RectResult | None = None
    scale: float | None = None
    score: float | None = None


@dataclass(frozen=True)
class OcrResult:
    lines: tuple[OcrLine, ...]


@dataclass(frozen=True)
class ColorMatchResult:
    fractions: tuple[float, ...]


Result = MatchResult | OcrResult | ColorMatchResult


@dataclass(frozen=True)
class BatchRequest:
    id: int
    items: tuple[Item, ...]
    images: tuple[bytes, ...] = ()


def read_message(stream: BinaryIO) -> bytes | None:
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    if len(header) < _HEADER.size:
        raise ProtocolError("connection closed in frame header")
    (length,) = _HEADER.unpack(header)
    if length > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload length {length} exceeds the {MAX_PAYLOAD_BYTES} byte cap")
    payload = _read_exact(stream, length)
    if payload is None or len(payload) < length:
        raise ProtocolError("connection closed in frame payload")
    return payload


def write_message(stream: BinaryIO, framed_bytes: bytes) -> None:
    stream.write(framed_bytes)
    stream.flush()


def parse_message(payload: bytes) -> BatchRequest:
    try:
        document = msgpack.unpackb(
            payload,
            raw=False,
            strict_map_key=False,
            max_str_len=MAX_PAYLOAD_BYTES,
            max_bin_len=MAX_PAYLOAD_BYTES,
            max_array_len=MAX_PAYLOAD_BYTES,
            max_map_len=MAX_PAYLOAD_BYTES,
        )
    except ValueError as exc:
        raise BatchParseError(None, f"invalid msgpack: {exc}") from exc
    if not isinstance(document, dict):
        raise BatchParseError(None, "batch must be a map")
    batch_id = document.get("id")
    if isinstance(batch_id, bool) or not isinstance(batch_id, int):
        raise BatchParseError(None, "batch id must be an integer")
    raw_items = document.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise BatchParseError(batch_id, "batch items must be a non-empty list")
    try:
        images = _parse_images(document.get("images"))
        items = _parse_items(raw_items, len(images))
    except ValueError as exc:
        raise BatchParseError(batch_id, str(exc)) from exc
    return BatchRequest(batch_id, items, images)


def format_ready() -> bytes:
    return _frame(msgpack.packb({"type": "ready", "v": PROTOCOL_VERSION}, use_bin_type=True))


def format_results(req_id: int, results: Sequence[Result]) -> bytes:
    document = {"id": req_id, "ok": True, "results": [_result_dict(result) for result in results]}
    return _frame(msgpack.packb(document, use_bin_type=True))


def format_error(req_id: int, message: str) -> bytes:
    return _frame(msgpack.packb({"id": req_id, "ok": False, "error": message}, use_bin_type=True))


def _read_exact(stream: BinaryIO, count: int) -> bytes | None:
    if count == 0:
        return b""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if not chunks:
        return None
    return b"".join(chunks)


def _frame(payload: bytes) -> bytes:
    return _HEADER.pack(len(payload)) + payload


def _result_dict(result: Result) -> dict[str, object]:
    if isinstance(result, MatchResult):
        payload: dict[str, object] = {"found": result.found}
        if result.rect is not None:
            payload["rect"] = {"x": result.rect.x, "y": result.rect.y, "w": result.rect.w, "h": result.rect.h}
        if result.scale is not None:
            payload["scale"] = result.scale
        if result.score is not None:
            payload["score"] = result.score
        return payload
    if isinstance(result, OcrResult):
        return {"lines": [_line_dict(line) for line in result.lines]}
    return {"fractions": list(result.fractions)}


def _line_dict(line: OcrLine) -> dict[str, object]:
    return {
        "text": line.text,
        "x": int(round(line.x)),
        "y": int(round(line.y)),
        "w": int(round(line.width)),
        "h": int(round(line.height)),
        "confidence": line.confidence,
    }


def _parse_images(raw: object) -> tuple[bytes, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("batch images must be a list of binary payloads")
    for image_index, entry in enumerate(raw):
        if not isinstance(entry, bytes):
            raise ValueError(f"batch image {image_index} must be binary")
    return tuple(raw)


def _parse_items(raw_items: list[object], image_count: int) -> tuple[Item, ...]:
    items: list[Item] = []
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            raise ValueError(f"item {index} must be a map")
        op = raw_item.get("op")
        if op == "match":
            items.append(_parse_match_item(index, raw_item, image_count))
        elif op == "ocr":
            items.append(_parse_ocr_item(index, raw_item, image_count))
        elif op == "colorMatch":
            items.append(_parse_color_match_item(index, raw_item, image_count))
        else:
            raise ValueError(f'item {index} has unsupported op "{op}"')
    return tuple(items)


def _parse_match_item(index: int, raw_item: dict[object, object], image_count: int) -> MatchItem:
    template = raw_item.get("template")
    if not isinstance(template, bytes):
        raise ValueError(f"item {index} template must be binary")
    return MatchItem(
        template,
        _required_index(index, raw_item, image_count),
        scale=_optional_number(raw_item.get("scale"), index, "scale"),
        threshold=_optional_number(raw_item.get("threshold"), index, "threshold"),
    )


def _parse_ocr_item(index: int, raw_item: dict[object, object], image_count: int) -> OcrItem:
    upscale = raw_item.get("upscale")
    if upscale is not None and not isinstance(upscale, bool):
        raise ValueError(f"item {index} upscale must be a bool")
    return OcrItem(_required_index(index, raw_item, image_count), _optional_rect(index, raw_item), upscale)


def _parse_color_match_item(index: int, raw_item: dict[object, object], image_count: int) -> ColorMatchItem:
    raw_ranges = raw_item.get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError(f"item {index} ranges must be a non-empty list")
    ranges = tuple(_parse_range(entry, index, range_index) for range_index, entry in enumerate(raw_ranges))
    return ColorMatchItem(_required_index(index, raw_item, image_count), ranges, _optional_rect(index, raw_item), _optional_mask(index, raw_item))


def _required_index(index: int, raw_item: dict[object, object], image_count: int) -> int:
    image = raw_item.get("image")
    if isinstance(image, bool) or not isinstance(image, int) or image < 0 or image >= image_count:
        raise ValueError(f"item {index} image must be an image index")
    return image


def _optional_rect(index: int, raw_item: dict[object, object]) -> CropRect | None:
    raw = raw_item.get("rect")
    if raw is None:
        return None
    if not _is_int_list(raw, 4):
        raise ValueError(f"item {index} rect must be four integers [x,y,w,h]")
    x, y, w, h = raw
    if x < 0 or y < 0:
        raise ValueError(f"item {index} rect origin must not be negative")
    if w < 1 or h < 1:
        raise ValueError(f"item {index} rect sides must be at least 1")
    return CropRect(x, y, w, h)


def _optional_mask(index: int, raw_item: dict[object, object]) -> AnnulusMask | None:
    raw = raw_item.get("mask")
    if raw is None:
        return None
    if not _is_int_list(raw, 4):
        raise ValueError(f"item {index} mask must be four integers [cx,cy,outer,inner]")
    cx, cy, outer, inner = raw
    if outer < 1:
        raise ValueError(f"item {index} mask outer radius must be at least 1")
    if inner < 0 or inner >= outer:
        raise ValueError(f"item {index} mask inner radius must satisfy 0 <= inner < outer")
    return AnnulusMask(cx, cy, outer, inner)


def _is_int_list(raw: object, length: int) -> bool:
    return isinstance(raw, list) and len(raw) == length and all(isinstance(entry, int) and not isinstance(entry, bool) for entry in raw)


def _optional_number(value: object, index: int, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"item {index} {name} must be a number")
    return value


def _parse_range(entry: object, index: int, range_index: int) -> HsvRange:
    prefix = f"item {index} range {range_index}"
    malformed = not isinstance(entry, list) or len(entry) != 6 or any(isinstance(value, bool) or not isinstance(value, int) for value in entry)
    if malformed:
        raise ValueError(f"{prefix} must be six integers [h0,h1,s0,s1,v0,v1]")
    h_min, h_max, s_min, s_max, v_min, v_max = entry
    if not (0 <= h_min <= 180 and 0 <= h_max <= 180):
        raise ValueError(f"{prefix} hue must be 0-180")
    if not (0 <= s_min <= 255 and 0 <= s_max <= 255 and 0 <= v_min <= 255 and 0 <= v_max <= 255):
        raise ValueError(f"{prefix} saturation and value must be 0-255")
    if h_min > h_max or s_min > s_max or v_min > v_max:
        raise ValueError(f"{prefix} channel mins must not exceed maxes")
    return HsvRange(h_min, h_max, s_min, s_max, v_min, v_max)
