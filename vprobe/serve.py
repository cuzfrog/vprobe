from __future__ import annotations

import argparse
import logging
import os
import select
import socket
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import BinaryIO

from vprobe.analyze import Executor, build_executor
from vprobe.protocol import (
    BatchParseError,
    Item,
    MatchItem,
    OcrItem,
    ProtocolError,
    format_error,
    format_ready,
    format_results,
    parse_message,
    read_message,
    write_message,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 51883
INTERRUPT_POLL_SECONDS = 0.5

_LEVELS = {
    "SILLY": logging.DEBUG,
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
}

log = logging.getLogger("vprobe")


def run_session(executor: Executor, input: BinaryIO, output: BinaryIO) -> None:
    while True:
        try:
            payload = read_message(input)
        except ProtocolError as exc:
            log.error("framing failed: %s", exc)
            _write_best_effort(output, format_error(-1, str(exc)))
            return
        if payload is None:
            return
        try:
            batch = parse_message(payload)
        except BatchParseError as exc:
            log.error("invalid batch: %s", exc)
            write_message(output, format_error(-1 if exc.batch_id is None else exc.batch_id, str(exc)))
            continue
        try:
            start = time.perf_counter()
            results = executor(batch.items, batch.images)
            total_ms = int(round((time.perf_counter() - start) * 1000))
        except Exception as exc:
            log.exception("batch failed id=%s", batch.id)
            write_message(output, format_error(batch.id, str(exc)))
            continue
        write_message(output, format_results(batch.id, results))
        log.info("batch id=%s items=%s bytes=%s total_ms=%s %s", batch.id, len(batch.items), len(payload), total_ms, _op_summary(batch.items))


def run_tcp(executor: Executor, host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        server.settimeout(INTERRUPT_POLL_SECONDS)
        log.info("listening on %s:%s", host, port)
        while True:
            try:
                connection, address = server.accept()
            except TimeoutError:
                continue
            log.info("client connected from %s", address)
            threading.Thread(target=_serve_client, args=(executor, connection), daemon=True).start()


def main(
    argv: list[str] | None = None,
    *,
    executor_factory: Callable[[], Executor] | None = None,
    input: BinaryIO = sys.stdin.buffer,
    output: BinaryIO = sys.stdout.buffer,
) -> None:
    _configure_logging()
    args = _parse_args(argv)
    try:
        executor = executor_factory() if executor_factory is not None else _default_executor(args.gpu)
        if args.tcp:
            run_tcp(executor, args.host, args.port)
        else:
            write_message(output, format_ready())
            run_session(executor, input, output)
    except KeyboardInterrupt:
        log.info("interrupted, shutting down")


def _serve_client(executor: Executor, connection: socket.socket) -> None:
    try:
        reader = connection.makefile("rb")
        writer = connection.makefile("wb")
        try:
            write_message(writer, format_ready())
            run_session(executor, _InterruptibleReader(connection, reader, INTERRUPT_POLL_SECONDS), writer)
        except ConnectionError:
            log.info("client connection lost")
        finally:
            _close_quietly(reader)
            _close_quietly(writer)
    finally:
        connection.close()
        log.info("client disconnected")


def _configure_logging() -> None:
    raw = os.environ.get("VPROBE_LOG_LEVEL")
    if raw is not None and raw.strip().upper() not in _LEVELS:
        print(f'invalid VPROBE_LOG_LEVEL "{raw}", falling back to INFO', file=sys.stderr)
    logging.basicConfig(stream=sys.stderr, level=_log_level(raw), format="%(levelname)s %(name)s %(message)s")


def _log_level(raw: str | None) -> int:
    if raw is None:
        return logging.INFO
    return _LEVELS.get(raw.strip().upper(), logging.INFO)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="vprobe")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    transport = serve.add_mutually_exclusive_group(required=True)
    transport.add_argument("--stdio", action="store_true")
    transport.add_argument("--tcp", action="store_true")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--gpu", action="store_true")
    return parser.parse_args(argv)


def _default_executor(gpu: bool) -> Executor:
    return build_executor(gpu=gpu)


def _write_best_effort(output: BinaryIO, framed_bytes: bytes) -> None:
    try:
        write_message(output, framed_bytes)
    except OSError:
        pass


def _op_summary(items: Sequence[Item]) -> str:
    counts = {"match": 0, "ocr": 0, "color_match": 0}
    for item in items:
        if isinstance(item, MatchItem):
            counts["match"] += 1
        elif isinstance(item, OcrItem):
            counts["ocr"] += 1
        else:
            counts["color_match"] += 1
    return " ".join(f"{op}={count}" for op, count in counts.items() if count)


def _close_quietly(stream: BinaryIO) -> None:
    try:
        stream.close()
    except OSError:
        pass


class _InterruptibleReader:
    def __init__(self, connection: socket.socket, reader: BinaryIO, poll_seconds: float) -> None:
        self._connection = connection
        self._reader = reader
        self._poll_seconds = poll_seconds

    def read(self, count: int) -> bytes:
        while not select.select([self._connection], [], [], self._poll_seconds)[0]:
            pass
        return self._reader.read1(count)
