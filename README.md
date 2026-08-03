# probe

A local, stateless, client-agnostic recognition daemon (Python/uv). It offers three computer-vision primitives and nothing else: a template `match` at a caller-supplied scale (OpenCV `matchTemplate`), RapidOCR text recognition (`ocr`), and HSV colour-range fraction sampling (`colorMatch`). It contains no knowledge of its clients — no config file, no labels, no template directory on disk. Templates, HSV ranges and the match scale arrive as bytes inside each request, and every recipe and threshold lives in the caller. The wire protocol is specified below.

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

## Install

From the repository root:

```bash
uv sync
```

## Start the daemon

```bash
uv run python -m probe serve --tcp [--host 127.0.0.1] [--port 51883]
```

There are no `--config`, `--images` or `--layout` flags: the daemon takes no inputs at startup beyond its transport. Flags:

- `--tcp` — run a long-lived daemon on `127.0.0.1` (default port 51883). Each client is served on its own thread, so multiple clients can connect at once; models load once and stay shared across connections, so client restarts and additional clients do not cost a model reload. Items within a batch run across a thread pool; OCR runs are serialized on the recognizer, `match`/`colorMatch` run concurrently.
- `--stdio` — alternative transport for tests: one session over stdin/stdout, ends on stdin EOF. Mutually exclusive with `--tcp`.
- `--host` / `--port` — tcp bind address and port (defaults `127.0.0.1` / `51883`).
- `--gpu` — run ONNX Runtime inference via DirectML (Windows 10+); on other platforms it logs a warning and stays on CPU, the default.

Clients connect to the chosen address and receive a framed `{"type":"ready","v":5}` on connection, then exchange framed msgpack messages. Request timeouts are a client-side setting; the daemon only owns its listen address (`--host`/`--port`, default `127.0.0.1:51883`). Logs go to stderr; the protocol stream is binary-framed msgpack only. Ctrl+C stops the daemon at any time — even idle between clients, on Windows consoles too — and exits cleanly; a `--stdio` session ends on stdin EOF (Ctrl+D on Unix, Ctrl+Z followed by Enter on Windows consoles).

## Protocol

Each message on the wire is a 4-byte big-endian payload length followed by one msgpack document, in both directions, capped at 64 MiB. On startup (stdio) or on each new connection (tcp) the daemon sends the handshake `{"type":"ready","v":5}` framed; a client must reject any `v` it does not speak.

A request is a batch: `{"id":<int>,"images":[<bin png>,...],"items":[...]}` with a non-empty item list; `images` is optional (absent ≡ empty) and is a batch-level table — the daemon decodes each referenced index lazily, at most once per batch, however many items share it. Items:

- `match` — `{"op":"match","template":<bin png>,"image":<int index>,"scale"?,"threshold"?}`. A single `matchTemplate` at `scale` (default 1; the template is resized by that factor before matching), threshold default 0.8. The image references the batch `images` table like the other ops (decoded at most once per batch, shared across items); the template is item-specific and stays inline. Result: `{"found":<bool>,"rect"?:"{x,y,w,h}","scale"?,"score"?}`; `rect` is image-relative and `w`/`h` equal `round(template dims * scale)`; `rect` and `scale` are omitted when not found; `score` is the normalized correlation, present on found and not-found results alike unless the scaled template could not be matched at all.
- `ocr` — `{"op":"ocr","image":<int index>,"rect"?:[x,y,w,h],"upscale"?}`. RapidOCR (PP-OCRv5 mobile, ONNX Runtime) over the indexed image, sliced to an image-relative `rect` when present (x/y >= 0, w/h >= 1); `upscale` (default true) grows a short crop to a 320 px minimum height first. Result: `{"lines":[{"text","x","y","w","h","confidence"}]}`, integer boxes, top-left origin, relative to the item's `rect`.
- `colorMatch` — `{"op":"colorMatch","image":<int index>,"rect"?,"ranges":[[h0,h1,s0,s1,v0,v1],...],"mask"?:[cx,cy,outer,inner]}`. Fraction of in-range HSV pixels per range, in order (H 0-180, S/V 0-255). The optional `mask` is a rect-relative annulus — a pixel counts when `inner² < (x-cx)² + (y-cy)² <= outer²` (outer >= 1, 0 <= inner < outer; cx/cy may be negative for clipped crops) — and fractions are normalized by the mask pixel count, so a zero-pixel mask yields zero fractions. Result: `{"fractions":[<float>,...]}`.

A success response is `{"id":<n>,"ok":true,"results":[...]}`, positional, one result per item in the same order. A failure is `{"id":<n>,"ok":false,"error":<string>}` for the whole batch, with `id:-1` when the framed message is unusable before the batch id: a parse failure or an executor exception (e.g. a `rect` beyond the decoded image, an undecodable png) fails the batch but the session continues; a framing violation (a length over the cap, a disconnect mid-frame) writes an error where it still can and then closes the connection. A clean EOF on stdio ends the session silently. There is no `analyze` operation and no layout the daemon reads.

## First startup

On the very first startup the RapidOCR ONNX models (~15 MB) are downloaded into the probe venv (`rapidocr/models/` under site-packages); startup then loads det/cls/rec before serving, and the daemon only starts accepting once they are loaded (`ocr models loaded in <ms> ms` at INFO). Every request thereafter runs against warm models — there is no lazy path.

## Running as a background service

The daemon is a plain foreground console process with no self-daemonization; use your OS's service manager to keep it alive.

Linux (systemd user unit, `~/.config/systemd/user/probe.service`):

```ini
[Unit]
Description=Probe daemon

[Service]
WorkingDirectory=/path/to/repo
ExecStart=/usr/bin/env uv run python -m probe serve --tcp
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now probe
```

Windows: register the same `uv run python -m probe serve --tcp` command with [NSSM](https://nssm.cc/) or a Task Scheduler task set to run at logon.

## Logging

Configure the level via env var `PROBE_LOG_LEVEL`. Accepted values (case-insensitive): `SILLY`, `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `FATAL` (SILLY and TRACE map to DEBUG). When unset it logs at INFO; an invalid value falls back to INFO with a stderr warning. At INFO the daemon logs a one-line per-batch summary plus connect/disconnect and the OCR model load; per-item timing is at DEBUG.

## Development

```bash
uv run pytest
```

Each `tests/test_<module>.py` tests `probe/<module>.py`. Every test file except `tests/test_recognition.py` runs offline and model-free — none of them construct a real recognizer; the executor tests patch `RapidRecognizer`. `tests/test_recognition.py` guards the real models end to end: it runs the unpatched executor against the synthetic fixtures committed under `tests/fixtures/` (random-noise canvases for `match`, solid colours and a painted ring for `colorMatch`, rendered digit and word strips for `ocr`). Those fixtures are generated ahead of time and checked in, and the tests read them (save one deliberately blank canvas synthesized in memory). The very first run downloads the ONNX models (~15 MB) into the probe venv, like the daemon's first startup. Regenerate the fixtures only when they must change:

```bash
uv run python tests/generate_fixtures.py
```
