# vprobe

A cross-platform, stateless, client-agnostic vision recognition daemon via TCP or stdio.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12)

## Install & Run

Install from PyPI as a uv tool:

```bash
uv tool install vprobe
vprobe serve
```

Upgrade:

```bash
uv tool upgrade vprobe
```

One-shot run without installing:

```bash
uvx vprobe serve
```

Flags:
- `--stdio` — alternative transport for tests: one session over stdin/stdout, ends on stdin EOF. Mutually exclusive with `--tcp`.
- `--gpu` — run ONNX Runtime inference via DirectML (Windows 10+); on other platforms it logs a warning and stays on CPU, the default.
- `--host` — default `127.0.0.1`.
- `--port` — default `23883`.

## First startup

On the very first startup the RapidOCR ONNX models (~15 MB) are downloaded into the vprobe venv (`rapidocr/models/` under site-packages); startup then loads det/cls/rec before serving, and the daemon only starts accepting once they are loaded (`ocr models loaded in <ms> ms` at INFO).

## Running as a background service

Linux (systemd user unit, `~/.config/systemd/user/vprobe.service`):

```ini
[Unit]
Description=Vprobe daemon

[Service]
WorkingDirectory=/path/to/repo
ExecStart=/usr/bin/env uv run vprobe serve --tcp
Restart=on-failure

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload && systemctl --user enable --now vprobe
```

If vprobe was installed via `uv tool install`, `ExecStart=/home/<user>/.local/bin/vprobe serve --tcp` works without the repo checkout.

Windows: register the same `uv run vprobe serve --tcp` command with [NSSM](https://nssm.cc/) or a Task Scheduler task set to run at logon.

## Logging

Configure the level via env var `VPROBE_LOG_LEVEL`. Accepted values (case-insensitive): `DEBUG`, `INFO`, `WARN`, `ERROR`, `FATAL`.

## Wire protocol
See [Protocol](doc/PROTOCOL.md)

## Development

```bash
uv run pytest
```

Each `tests/test_<module>.py` tests `vprobe/<module>.py`. Every test file except `tests/test_recognition.py` runs offline and model-free — none of them construct a real recognizer; the executor tests patch `RapidRecognizer`. `tests/test_recognition.py` guards the real models end to end: it runs the unpatched executor against the synthetic fixtures committed under `tests/fixtures/` (random-noise canvases for `match`, solid colours and a painted ring for `colorMatch`, rendered digit and word strips for `ocr`). Those fixtures are generated ahead of time and checked in, and the tests read them (save one deliberately blank canvas synthesized in memory). The very first run downloads the ONNX models (~15 MB) into the vprobe venv. 

Regenerate the fixtures:
```bash
uv run python tests/generate_fixtures.py
```

### Start the daemon (from source)

```bash
uv sync
uv run vprobe serve --tcp [--host 127.0.0.1] [--port 23883]
```

## License
MIT

## Author
Cause Chung (cuzfrog@gmail.com)
