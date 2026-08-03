import logging
import socket
import struct
import threading
import time
from io import BytesIO

import msgpack
import pytest

import vprobe.serve
from vprobe.ocr import OcrLine
from vprobe.protocol import (
    MAX_PAYLOAD_BYTES,
    ColorMatchResult,
    MatchItem,
    MatchResult,
    OcrItem,
    OcrResult,
    RectResult,
    read_message,
)
from vprobe.serve import _configure_logging, _log_level, main, run_session, run_tcp

MATCH_ITEM = {"op": "match", "template": b"tpl", "image": 0}
OCR_ITEM = {"op": "ocr", "image": 0}
COLOR_ITEM = {"op": "colorMatch", "image": 0, "ranges": [[0, 5, 200, 255, 200, 255]]}
IMAGES = [b"img"]


def frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def pack(document: object) -> bytes:
    return msgpack.packb(document, use_bin_type=True)


def batch(items: list[object], req_id: int = 7, images: list[bytes] = IMAGES) -> bytes:
    return pack({"id": req_id, "items": items, "images": images})


def frames(data: bytes) -> list[object]:
    stream = BytesIO(data)
    documents = []
    while True:
        payload = read_message(stream)
        if payload is None:
            return documents
        documents.append(msgpack.unpackb(payload, raw=False))


def constant_executor(results):
    return lambda items, images: results


def test_mixed_batch_results_are_positional():
    results = [
        MatchResult(found=False),
        OcrResult(lines=(OcrLine(text="70", confidence=0.9, x=1.0, y=2.0, width=3.0, height=4.0),)),
        ColorMatchResult(fractions=(0.5,)),
    ]
    output = BytesIO()
    run_session(constant_executor(results), BytesIO(frame(batch([MATCH_ITEM, OCR_ITEM, COLOR_ITEM]))), output)
    (doc,) = frames(output.getvalue())
    expected = {
        "id": 7,
        "ok": True,
        "results": [
            {"found": False},
            {"lines": [{"text": "70", "x": 1, "y": 2, "w": 3, "h": 4, "confidence": 0.9}]},
            {"fractions": [0.5]},
        ],
    }
    assert doc == expected


def test_executor_receives_parsed_items_and_images():
    seen = []
    seen_images = []

    def execute(items, images):
        seen.extend(items)
        seen_images.append(tuple(images))
        return [MatchResult(found=False)] * len(items)

    run_session(execute, BytesIO(frame(batch([MATCH_ITEM, OCR_ITEM], req_id=1))), BytesIO())
    assert [type(item) for item in seen] == [MatchItem, OcrItem]
    assert seen[1].image == 0
    assert seen_images == [(b"img",)]


def test_executor_failure_fails_only_the_batch_and_session_continues():
    state = {"raised": False}

    def execute(items, images):
        if not state["raised"]:
            state["raised"] = True
            raise ValueError("item 0 (match) failed: could not decode png image")
        return [MatchResult(found=False)]

    input = BytesIO(frame(batch([MATCH_ITEM], req_id=7)) + frame(batch([MATCH_ITEM], req_id=8)))
    output = BytesIO()
    run_session(execute, input, output)
    error, ok = frames(output.getvalue())
    assert error == {"id": 7, "ok": False, "error": "item 0 (match) failed: could not decode png image"}
    assert ok == {"id": 8, "ok": True, "results": [{"found": False}]}


def test_bad_payload_answers_id_minus_one_and_session_continues():
    input = BytesIO(frame(b"\xc1") + frame(batch([MATCH_ITEM], req_id=9)))
    output = BytesIO()
    run_session(constant_executor([MatchResult(found=False)]), input, output)
    error, ok = frames(output.getvalue())
    assert error["id"] == -1
    assert error["ok"] is False
    assert "invalid msgpack" in error["error"]
    assert ok == {"id": 9, "ok": True, "results": [{"found": False}]}


def test_schema_violation_answers_with_the_batch_id_and_session_continues():
    input = BytesIO(frame(pack({"id": 5})) + frame(batch([MATCH_ITEM], req_id=6)))
    output = BytesIO()
    run_session(constant_executor([MatchResult(found=False)]), input, output)
    error, ok = frames(output.getvalue())
    assert error == {"id": 5, "ok": False, "error": "batch items must be a non-empty list"}
    assert ok["id"] == 6 and ok["ok"] is True


def test_bad_op_answers_with_the_batch_id_and_session_continues():
    input = BytesIO(frame(batch([{"op": "bogus"}], req_id=4)) + frame(batch([MATCH_ITEM], req_id=6)))
    output = BytesIO()
    run_session(constant_executor([MatchResult(found=False)]), input, output)
    error, ok = frames(output.getvalue())
    assert error == {"id": 4, "ok": False, "error": 'item 0 has unsupported op "bogus"'}
    assert ok["id"] == 6 and ok["ok"] is True


def test_over_cap_frame_answers_id_minus_one_and_ends_session():
    tail = frame(batch([MATCH_ITEM], req_id=9))
    input = BytesIO(struct.pack(">I", MAX_PAYLOAD_BYTES + 1) + b"\x00" * 8 + tail)
    output = BytesIO()
    run_session(constant_executor([MatchResult(found=False)]), input, output)
    (error,) = frames(output.getvalue())
    assert error["id"] == -1
    assert error["ok"] is False
    assert "cap" in error["error"]


def test_short_read_answers_id_minus_one_and_ends_session():
    output = BytesIO()
    run_session(constant_executor([]), BytesIO(struct.pack(">I", 100) + b"short"), output)
    (error,) = frames(output.getvalue())
    assert error["id"] == -1
    assert "payload" in error["error"]


def test_clean_eof_ends_session_silently():
    output = BytesIO()
    run_session(constant_executor([]), BytesIO(b""), output)
    assert output.getvalue() == b""


def test_eof_after_batches_answers_each_batch():
    input = BytesIO(frame(batch([MATCH_ITEM], req_id=1)) + frame(batch([MATCH_ITEM], req_id=2)))
    output = BytesIO()
    run_session(constant_executor([MatchResult(found=False)]), input, output)
    assert [doc["id"] for doc in frames(output.getvalue())] == [1, 2]


def test_batch_summary_is_logged_at_info(caplog):
    results = [MatchResult(found=False), OcrResult(lines=()), ColorMatchResult((0.1,)), ColorMatchResult((0.2,))]
    input = BytesIO(frame(batch([MATCH_ITEM, OCR_ITEM, COLOR_ITEM, COLOR_ITEM], req_id=7)))
    with caplog.at_level(logging.INFO, logger="vprobe"):
        run_session(constant_executor(results), input, BytesIO())
    summary = next(record.message for record in caplog.records if record.message.startswith("batch id=7"))
    assert "items=4" in summary
    assert "bytes=" in summary
    assert "total_ms=" in summary
    assert "match=1" in summary
    assert "ocr=1" in summary
    assert "color_match=2" in summary


def test_main_stdio_writes_framed_ready_then_answers_one_batch():
    input = BytesIO(frame(batch([MATCH_ITEM], req_id=3)))
    output = BytesIO()
    builds = []

    def factory():
        builds.append(1)
        return constant_executor([MatchResult(found=True, rect=RectResult(1, 2, 3, 4), scale=1.0)])

    main(["serve", "--stdio"], executor_factory=factory, input=input, output=output)

    ready, response = frames(output.getvalue())
    assert ready == {"type": "ready", "v": 5}
    assert response == {"id": 3, "ok": True, "results": [{"found": True, "rect": {"x": 1, "y": 2, "w": 3, "h": 4}, "scale": 1.0}]}
    assert builds == [1]


def test_serve_requires_a_transport():
    with pytest.raises(SystemExit):
        main(["serve"], executor_factory=lambda: constant_executor([]), input=BytesIO(), output=BytesIO())


def test_stdio_and_tcp_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main(["serve", "--stdio", "--tcp"], executor_factory=lambda: constant_executor([]), input=BytesIO(), output=BytesIO())


def test_main_builds_executor_with_gpu_off_by_default(monkeypatch):
    seen = []

    def fake_build(gpu):
        seen.append(gpu)
        return constant_executor([])

    monkeypatch.setattr("vprobe.serve.build_executor", fake_build)
    main(["serve", "--stdio"], input=BytesIO(b""), output=BytesIO())
    assert seen == [False]


def test_main_gpu_flag_builds_executor_with_gpu_on(monkeypatch):
    seen = []

    def fake_build(gpu):
        seen.append(gpu)
        return constant_executor([])

    monkeypatch.setattr("vprobe.serve.build_executor", fake_build)
    main(["serve", "--stdio", "--gpu"], input=BytesIO(b""), output=BytesIO())
    assert seen == [True]


def test_log_level_defaults_to_info_when_unset():
    assert _log_level(None) == logging.INFO


def test_log_level_maps_vocabulary_case_insensitively():
    assert _log_level("SILLY") == logging.DEBUG
    assert _log_level("trace") == logging.DEBUG
    assert _log_level("Debug") == logging.DEBUG
    assert _log_level("info") == logging.INFO
    assert _log_level("warn") == logging.WARNING
    assert _log_level("ERROR") == logging.ERROR
    assert _log_level("fatal") == logging.CRITICAL


def test_log_level_falls_back_to_info_on_invalid_value():
    assert _log_level("loud") == logging.INFO


def test_configure_logging_warns_on_stderr_for_invalid_level(monkeypatch, capsys):
    monkeypatch.setenv("VPROBE_LOG_LEVEL", "loud")
    _configure_logging()
    assert 'invalid VPROBE_LOG_LEVEL "loud"' in capsys.readouterr().err


def test_tcp_serves_across_idle_polls_partial_frames_and_reconnects():
    vprobe.serve.INTERRUPT_POLL_SECONDS = 0.05
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    thread = threading.Thread(target=run_tcp, args=(constant_executor([MatchResult(found=False)]), "127.0.0.1", port), daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            assert read_socket_message(client) == {"type": "ready", "v": 5}
            time.sleep(0.2)
            framed = frame(batch([MATCH_ITEM], req_id=3))
            client.sendall(framed[:2])
            time.sleep(0.2)
            client.sendall(framed[2:])
            assert read_socket_message(client) == {"id": 3, "ok": True, "results": [{"found": False}]}
        time.sleep(0.2)
        with socket.create_connection(("127.0.0.1", port), timeout=2) as second:
            assert read_socket_message(second) == {"type": "ready", "v": 5}
    finally:
        vprobe.serve.INTERRUPT_POLL_SECONDS = 0.5


def test_tcp_serves_a_second_client_while_the_first_session_is_busy():
    vprobe.serve.INTERRUPT_POLL_SECONDS = 0.05
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()
    release = threading.Event()
    entered = threading.Event()
    calls = []

    def busy_executor(items, images):
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            release.wait(5)
        return [MatchResult(found=False)]

    thread = threading.Thread(target=run_tcp, args=(busy_executor, "127.0.0.1", port), daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as first:
            assert read_socket_message(first) == {"type": "ready", "v": 5}
            first.sendall(frame(batch([MATCH_ITEM], req_id=1)))
            assert entered.wait(5)
            with socket.create_connection(("127.0.0.1", port), timeout=5) as second:
                assert read_socket_message(second) == {"type": "ready", "v": 5}
                second.sendall(frame(batch([MATCH_ITEM], req_id=2)))
                assert read_socket_message(second) == {"id": 2, "ok": True, "results": [{"found": False}]}
            release.set()
            assert read_socket_message(first) == {"id": 1, "ok": True, "results": [{"found": False}]}
    finally:
        vprobe.serve.INTERRUPT_POLL_SECONDS = 0.5
        release.set()


def test_keyboard_interrupt_in_tcp_shuts_down_cleanly(caplog):
    def interrupting_factory():
        raise KeyboardInterrupt

    with caplog.at_level(logging.INFO, logger="vprobe"):
        main(["serve", "--tcp"], executor_factory=interrupting_factory, input=BytesIO(), output=BytesIO())
    assert any(record.message == "interrupted, shutting down" for record in caplog.records)


def test_keyboard_interrupt_in_stdio_shuts_down_cleanly():
    def interrupting_executor(items, images):
        raise KeyboardInterrupt

    main(["serve", "--stdio"], executor_factory=lambda: interrupting_executor, input=BytesIO(frame(batch([MATCH_ITEM]))), output=BytesIO())


def read_socket_message(sock):
    header = b""
    while len(header) < 4:
        header += sock.recv(4 - len(header))
    (length,) = struct.unpack(">I", header)
    payload = b""
    while len(payload) < length:
        payload += sock.recv(length - len(payload))
    return msgpack.unpackb(payload, raw=False)
