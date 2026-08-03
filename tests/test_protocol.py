import struct
from io import BytesIO

import msgpack
import pytest

from vprobe.classify import HsvRange
from vprobe.ocr import OcrLine
from vprobe.protocol import (
    MAX_PAYLOAD_BYTES,
    PROTOCOL_VERSION,
    AnnulusMask,
    BatchParseError,
    BatchRequest,
    ColorMatchItem,
    ColorMatchResult,
    CropRect,
    MatchItem,
    MatchResult,
    OcrItem,
    OcrResult,
    ProtocolError,
    RectResult,
    format_error,
    format_ready,
    format_results,
    parse_message,
    read_message,
    write_message,
)

PNG = b"\x89PNG-not-really"
PNG_TWO = b"\x89PNG-second"
TEMPLATE = b"template-bytes"


def frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def pack(document: object) -> bytes:
    return msgpack.packb(document, use_bin_type=True)


def deframe(framed: bytes) -> object:
    (length,) = struct.unpack(">I", framed[:4])
    assert length == len(framed) - 4
    return msgpack.unpackb(framed[4:], raw=False)


def batch(items: list[object], req_id: int = 7, images: list[bytes] | None = None, **extra: object) -> bytes:
    document: dict[str, object] = {"id": req_id, "items": items}
    if images is not None:
        document["images"] = images
    document.update(extra)
    return pack(document)


def test_framing_round_trip_over_stream():
    payload = batch([{"op": "ocr", "image": 0}], images=[PNG])
    stream = BytesIO()
    write_message(stream, frame(payload))
    stream.seek(0)
    assert read_message(stream) == payload


def test_multiple_messages_read_back_in_order():
    payloads = [b"first", b"second", b"third"]
    stream = BytesIO(b"".join(frame(payload) for payload in payloads))
    assert [read_message(stream) for _ in payloads] == payloads
    assert read_message(stream) is None


def test_clean_eof_before_header_returns_none():
    assert read_message(BytesIO(b"")) is None


def test_zero_length_frame_reads_as_empty_payload():
    assert read_message(BytesIO(struct.pack(">I", 0))) == b""


def test_length_over_cap_raises_protocol_error():
    stream = BytesIO(struct.pack(">I", MAX_PAYLOAD_BYTES + 1) + b"\x00" * 16)
    with pytest.raises(ProtocolError, match="cap"):
        read_message(stream)


def test_truncated_payload_raises_protocol_error():
    stream = BytesIO(struct.pack(">I", 16) + b"short")
    with pytest.raises(ProtocolError, match="payload"):
        read_message(stream)


def test_partial_header_raises_protocol_error():
    with pytest.raises(ProtocolError, match="header"):
        read_message(BytesIO(b"\x00\x00"))


def test_cap_is_64_mib():
    assert MAX_PAYLOAD_BYTES == 67108864


def test_format_ready_announces_protocol_version_5():
    assert PROTOCOL_VERSION == 5
    assert deframe(format_ready()) == {"type": "ready", "v": 5}


def test_format_error_frames_id_ok_and_message():
    assert deframe(format_error(-1, "boom")) == {"id": -1, "ok": False, "error": "boom"}


def test_format_results_match_found_includes_rect_and_scale():
    framed = format_results(7, [MatchResult(found=True, rect=RectResult(1, 2, 3, 4), scale=1.5)])
    assert deframe(framed) == {"id": 7, "ok": True, "results": [{"found": True, "rect": {"x": 1, "y": 2, "w": 3, "h": 4}, "scale": 1.5}]}


def test_format_results_match_not_found_omits_rect_and_scale():
    assert deframe(format_results(3, [MatchResult(found=False)])) == {"id": 3, "ok": True, "results": [{"found": False}]}


def test_format_results_match_includes_score_when_present():
    framed = format_results(7, [MatchResult(found=False, score=0.74)])
    assert deframe(framed) == {"id": 7, "ok": True, "results": [{"found": False, "score": 0.74}]}


def test_format_results_ocr_rounds_line_boxes_to_ints():
    lines = (OcrLine(text="hi", confidence=0.9, x=1.4, y=2.6, width=10.6, height=3.2),)
    document = deframe(format_results(1, [OcrResult(lines=lines)]))
    assert document == {"id": 1, "ok": True, "results": [{"lines": [{"text": "hi", "x": 1, "y": 3, "w": 11, "h": 3, "confidence": 0.9}]}]}


def test_format_results_color_match_lists_fractions_positionally():
    document = deframe(format_results(2, [ColorMatchResult(fractions=(0.5, 1.0))]))
    assert document == {"id": 2, "ok": True, "results": [{"fractions": [0.5, 1.0]}]}


def test_parse_images_table_absent_defaults_to_empty():
    with pytest.raises(ValueError, match="item 0 image must be an image index"):
        parse_message(batch([{"op": "match", "template": TEMPLATE, "image": 0}]))


def test_parse_images_table_round_trips_positionally():
    request = parse_message(batch([{"op": "ocr", "image": 1}], images=[PNG, PNG_TWO]))
    assert request.images == (PNG, PNG_TWO)


def test_parse_rejects_non_list_images_table():
    with pytest.raises(ValueError, match="batch images must be a list of binary payloads"):
        parse_message(batch([{"op": "match", "template": TEMPLATE, "image": 0}], images=PNG))


def test_parse_rejects_non_binary_images_table_entry():
    with pytest.raises(ValueError, match="batch image 1 must be binary"):
        parse_message(batch([{"op": "ocr", "image": 0}], images=[PNG, "base64-ish"]))


def test_parse_match_item_minimal():
    request = parse_message(batch([{"op": "match", "template": TEMPLATE, "image": 0}], images=[PNG]))
    assert request == BatchRequest(id=7, items=(MatchItem(template=TEMPLATE, image=0),), images=(PNG,))


def test_parse_match_item_with_optional_params():
    item = {"op": "match", "template": TEMPLATE, "image": 0, "scale": 1.1, "threshold": 0.9}
    request = parse_message(batch([item], images=[PNG]))
    assert request.items == (MatchItem(TEMPLATE, 0, scale=1.1, threshold=0.9),)


def test_parse_match_accepts_int_number_params():
    item = {"op": "match", "template": TEMPLATE, "image": 0, "scale": 2, "threshold": 1}
    request = parse_message(batch([item], images=[PNG]))
    assert (request.items[0].scale, request.items[0].threshold) == (2, 1)


def test_parse_ocr_item_index_rect_and_upscale():
    request = parse_message(batch([{"op": "ocr", "image": 0}, {"op": "ocr", "image": 1, "rect": [1, 2, 3, 4], "upscale": False}], images=[PNG, PNG_TWO]))
    assert request.items == (OcrItem(image=0), OcrItem(image=1, rect=CropRect(1, 2, 3, 4), upscale=False))


def test_parse_color_match_item_index_rect_mask_and_ranges():
    item = {"op": "colorMatch", "image": 0, "rect": [5, 6, 7, 8], "mask": [3, 3, 4, 1], "ranges": [[0, 5, 200, 255, 200, 255], [55, 65, 0, 255, 100, 200]]}
    request = parse_message(batch([item], images=[PNG]))
    ranges = (HsvRange(0, 5, 200, 255, 200, 255), HsvRange(55, 65, 0, 255, 100, 200))
    assert request.items == (ColorMatchItem(image=0, ranges=ranges, rect=CropRect(5, 6, 7, 8), mask=AnnulusMask(3, 3, 4, 1)),)


def test_parse_mask_center_may_be_negative_for_clipped_rects():
    item = {"op": "colorMatch", "image": 0, "mask": [-3, -4, 10, 2], "ranges": [[0, 5, 200, 255, 200, 255]]}
    request = parse_message(batch([item], images=[PNG]))
    assert request.items[0].mask == AnnulusMask(-3, -4, 10, 2)


def test_parse_mixed_batch_preserves_order():
    items = [
        {"op": "ocr", "image": 0},
        {"op": "match", "template": TEMPLATE, "image": 0},
        {"op": "colorMatch", "image": 0, "ranges": [[0, 1, 0, 1, 0, 1]]},
    ]
    request = parse_message(batch(items, images=[PNG]))
    assert [type(item) for item in request.items] == [OcrItem, MatchItem, ColorMatchItem]
    assert request.id == 7
    assert request.images == (PNG,)


def test_parse_ignores_unknown_extra_keys_at_every_level():
    item = {"op": "ocr", "image": 0, "future": {"nested": [1, 2]}, "flag": True}
    request = parse_message(batch([item], images=[PNG], layout="junk", imagePos={"x": 1}))
    assert request == BatchRequest(id=7, items=(OcrItem(image=0),), images=(PNG,))


def test_parse_rejects_invalid_msgpack_byte():
    with pytest.raises(ValueError, match="invalid msgpack:"):
        parse_message(b"\xc1")


def test_parse_rejects_empty_payload():
    with pytest.raises(ValueError, match="invalid msgpack:"):
        parse_message(b"")


def test_parse_rejects_trailing_bytes():
    with pytest.raises(ValueError, match="invalid msgpack:"):
        parse_message(batch([{"op": "ocr", "image": 0}], images=[PNG]) + b"\x00")


def test_parse_rejects_non_map_document():
    with pytest.raises(ValueError, match="batch must be a map"):
        parse_message(pack([{"op": "ocr"}]))


@pytest.mark.parametrize("bad_id", [None, True, "7", 1.5], ids=["missing", "bool", "string", "float"])
def test_parse_rejects_bad_id(bad_id):
    document: dict[str, object] = {"items": [{"op": "match", "template": TEMPLATE, "image": 0}]}
    if bad_id is not None:
        document["id"] = bad_id
    with pytest.raises(ValueError, match="batch id must be an integer"):
        parse_message(pack(document))


@pytest.mark.parametrize("bad_items", [None, [], "x", 3], ids=["missing", "empty", "string", "int"])
def test_parse_rejects_bad_items(bad_items):
    document: dict[str, object] = {"id": 7}
    if bad_items is not None:
        document["items"] = bad_items
    with pytest.raises(ValueError, match="batch items must be a non-empty list"):
        parse_message(pack(document))


def test_parse_error_carries_the_batch_id_once_the_id_is_usable():
    with pytest.raises(BatchParseError) as caught:
        parse_message(pack({"id": 5}))
    assert caught.value.batch_id == 5
    assert str(caught.value) == "batch items must be a non-empty list"


def test_parse_error_carries_the_batch_id_for_item_failures():
    with pytest.raises(BatchParseError) as caught:
        parse_message(batch([{"op": "scan", "image": 0}], req_id=9, images=[PNG]))
    assert caught.value.batch_id == 9
    assert str(caught.value) == 'item 0 has unsupported op "scan"'


@pytest.mark.parametrize("payload", [b"\xc1", pack(["not-a-map"]), pack({"items": [{"op": "ocr", "image": 0}]})], ids=["msgpack", "non-map", "bad-id"])
def test_parse_error_has_no_batch_id_before_the_id_is_usable(payload):
    with pytest.raises(BatchParseError) as caught:
        parse_message(payload)
    assert caught.value.batch_id is None


def test_parse_rejects_non_map_item():
    with pytest.raises(ValueError, match="item 0 must be a map"):
        parse_message(batch(["match"]))


def test_parse_reports_item_index_in_errors():
    items = [{"op": "ocr", "image": 0}, "nope"]
    with pytest.raises(ValueError, match="item 1 must be a map"):
        parse_message(batch(items, images=[PNG]))


def test_parse_rejects_missing_op_as_none():
    with pytest.raises(ValueError, match=r'item 0 has unsupported op "None"'):
        parse_message(batch([{"image": 0}], images=[PNG]))


def test_parse_rejects_unknown_op():
    with pytest.raises(ValueError, match='item 0 has unsupported op "scan"'):
        parse_message(batch([{"op": "scan", "image": 0}], images=[PNG]))


@pytest.mark.parametrize("bad_template", ["anchor.png", 5], ids=["string", "int"])
def test_parse_rejects_non_binary_template(bad_template):
    with pytest.raises(ValueError, match="item 0 template must be binary"):
        parse_message(batch([{"op": "match", "template": bad_template, "image": 0}], images=[PNG]))


def test_parse_rejects_missing_template():
    with pytest.raises(ValueError, match="item 0 template must be binary"):
        parse_message(batch([{"op": "match", "image": 0}], images=[PNG]))


@pytest.mark.parametrize("op", ["match", "ocr", "colorMatch"])
@pytest.mark.parametrize("bad_index", [None, -1, 2, "0", 0.5, True], ids=["missing", "negative", "out-of-range", "string", "float", "bool"])
def test_parse_rejects_bad_image_index(op, bad_index):
    item: dict[str, object] = {"op": op}
    if bad_index is not None:
        item["image"] = bad_index
    if op == "colorMatch":
        item["ranges"] = [[0, 1, 0, 1, 0, 1]]
    if op == "match":
        item["template"] = TEMPLATE
    with pytest.raises(ValueError, match="item 0 image must be an image index"):
        parse_message(batch([item], images=[PNG, PNG_TWO]))


@pytest.mark.parametrize("name", ["scale", "threshold"])
@pytest.mark.parametrize("bad_number", ["0.8", True, [0.8]], ids=["string", "bool", "list"])
def test_parse_rejects_bad_numbers(name, bad_number):
    item = {"op": "match", "template": TEMPLATE, "image": 0, name: bad_number}
    with pytest.raises(ValueError, match=f"item 0 {name} must be a number"):
        parse_message(batch([item], images=[PNG]))


@pytest.mark.parametrize("bad_upscale", [1, "yes"], ids=["int", "string"])
def test_parse_rejects_bad_upscale(bad_upscale):
    with pytest.raises(ValueError, match="item 0 upscale must be a bool"):
        parse_message(batch([{"op": "ocr", "image": 0, "upscale": bad_upscale}], images=[PNG]))


@pytest.mark.parametrize("op", ["ocr", "colorMatch"])
def test_parse_rect_valid_for_indexed_ops(op):
    item: dict[str, object] = {"op": op, "image": 0, "rect": [0, 0, 1, 1]}
    if op == "colorMatch":
        item["ranges"] = [[0, 1, 0, 1, 0, 1]]
    request = parse_message(batch([item], images=[PNG]))
    assert request.items[0].rect == CropRect(0, 0, 1, 1)


@pytest.mark.parametrize("op", ["ocr", "colorMatch"])
@pytest.mark.parametrize("bad_rect", [[1, 2, 3], [1, 2, 3, 4, 5], [1.0, 2, 3, 4], [True, 2, 3, 4], "rect"], ids=["short", "long", "float-entry", "bool-entry", "not-a-list"])
def test_parse_rejects_malformed_rect_shape(op, bad_rect):
    item: dict[str, object] = {"op": op, "image": 0, "rect": bad_rect}
    if op == "colorMatch":
        item["ranges"] = [[0, 1, 0, 1, 0, 1]]
    with pytest.raises(ValueError, match="item 0 rect must be four integers"):
        parse_message(batch([item], images=[PNG]))


@pytest.mark.parametrize("bad_rect", [[-1, 2, 3, 4], [1, -2, 3, 4]], ids=["negative-x", "negative-y"])
def test_parse_rejects_negative_rect_origin(bad_rect):
    with pytest.raises(ValueError, match="item 0 rect origin must not be negative"):
        parse_message(batch([{"op": "ocr", "image": 0, "rect": bad_rect}], images=[PNG]))


@pytest.mark.parametrize("bad_rect", [[1, 2, 0, 4], [1, 2, 3, 0], [1, 2, -3, 4]], ids=["zero-w", "zero-h", "negative-w"])
def test_parse_rejects_non_positive_rect_sides(bad_rect):
    with pytest.raises(ValueError, match="item 0 rect sides must be at least 1"):
        parse_message(batch([{"op": "ocr", "image": 0, "rect": bad_rect}], images=[PNG]))


@pytest.mark.parametrize("bad_mask", [[1, 2, 3], [1, 2, 3, 4, 5], [1.5, 2, 3, 4], [True, 2, 3, 4], "mask"], ids=["short", "long", "float-entry", "bool-entry", "not-a-list"])
def test_parse_rejects_malformed_mask_shape(bad_mask):
    item = {"op": "colorMatch", "image": 0, "mask": bad_mask, "ranges": [[0, 1, 0, 1, 0, 1]]}
    with pytest.raises(ValueError, match="item 0 mask must be four integers"):
        parse_message(batch([item], images=[PNG]))


def test_parse_rejects_zero_mask_outer_radius():
    item = {"op": "colorMatch", "image": 0, "mask": [0, 0, 0, 0], "ranges": [[0, 1, 0, 1, 0, 1]]}
    with pytest.raises(ValueError, match="item 0 mask outer radius must be at least 1"):
        parse_message(batch([item], images=[PNG]))


@pytest.mark.parametrize("bad_mask", [[0, 0, 5, -1], [0, 0, 5, 5], [0, 0, 5, 9]], ids=["negative-inner", "inner-equals-outer", "inner-above-outer"])
def test_parse_rejects_bad_mask_inner_radius(bad_mask):
    item = {"op": "colorMatch", "image": 0, "mask": bad_mask, "ranges": [[0, 1, 0, 1, 0, 1]]}
    with pytest.raises(ValueError, match="item 0 mask inner radius must satisfy 0 <= inner < outer"):
        parse_message(batch([item], images=[PNG]))


@pytest.mark.parametrize("bad_ranges", [None, [], "0-5"], ids=["missing", "empty", "string"])
def test_parse_rejects_bad_ranges(bad_ranges):
    item: dict[str, object] = {"op": "colorMatch", "image": 0}
    if bad_ranges is not None:
        item["ranges"] = bad_ranges
    with pytest.raises(ValueError, match="item 0 ranges must be a non-empty list"):
        parse_message(batch([item], images=[PNG]))


@pytest.mark.parametrize(
    "bad_range",
    [
        [0, 5, 200, 255, 200],
        [0, 5, 200, 255, 200, 255, 9],
        [0, 5.0, 200, 255, 200, 255],
        [True, 5, 200, 255, 200, 255],
        [0, 5, 200, 255, 200, "255"],
        "0-5",
    ],
    ids=["short", "long", "float-entry", "bool-entry", "string-entry", "not-a-list"],
)
def test_parse_rejects_malformed_range_shape(bad_range):
    with pytest.raises(ValueError) as caught:
        parse_message(batch([{"op": "colorMatch", "image": 0, "ranges": [bad_range]}], images=[PNG]))
    assert str(caught.value) == "item 0 range 0 must be six integers [h0,h1,s0,s1,v0,v1]"


def test_parse_reports_range_index_in_errors():
    ranges = [[0, 5, 200, 255, 200, 255], [0, 5, 200, 255]]
    with pytest.raises(ValueError, match="item 0 range 1 must be six integers"):
        parse_message(batch([{"op": "colorMatch", "image": 0, "ranges": ranges}], images=[PNG]))


@pytest.mark.parametrize("hue", [-1, 181])
def test_parse_rejects_hue_out_of_bounds(hue):
    with pytest.raises(ValueError, match="item 0 range 0 hue must be 0-180"):
        parse_message(batch([{"op": "colorMatch", "image": 0, "ranges": [[hue, 5, 200, 255, 200, 255]]}], images=[PNG]))


@pytest.mark.parametrize("bad_range", [[0, 5, -1, 255, 200, 255], [0, 5, 200, 256, 200, 255], [0, 5, 200, 255, 200, 999]], ids=["s-min", "s-max", "v-max"])
def test_parse_rejects_saturation_or_value_out_of_bounds(bad_range):
    with pytest.raises(ValueError, match="item 0 range 0 saturation and value must be 0-255"):
        parse_message(batch([{"op": "colorMatch", "image": 0, "ranges": [bad_range]}], images=[PNG]))


@pytest.mark.parametrize("bad_range", [[5, 0, 200, 255, 200, 255], [0, 5, 255, 200, 200, 255], [0, 5, 200, 255, 255, 200]], ids=["hue", "saturation", "value"])
def test_parse_rejects_min_above_max(bad_range):
    with pytest.raises(ValueError, match="item 0 range 0 channel mins must not exceed maxes"):
        parse_message(batch([{"op": "colorMatch", "image": 0, "ranges": [bad_range]}], images=[PNG]))
