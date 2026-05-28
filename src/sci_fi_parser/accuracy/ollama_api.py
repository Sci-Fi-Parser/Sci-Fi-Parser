"""Ollama daemon HTTP client + disk-space preflight.

Talks to the local ollama daemon over its stable HTTP endpoints via stdlib
``urllib`` -- the ``ollama`` Python package's response shapes change between
releases and this module only needs list/pull/delete plus the on-disk size,
so we sidestep that volatility.

Used by :mod:`sci_fi_parser.accuracy.vlm_compare` for cross-model runs. Kept
separate so the comparison runner stays free of HTTP/daemon concerns.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
def _ollama_url(path: str) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return f"{host}{path}"


def _ollama_request(method: str, path: str, body: dict | None = None,
                    *, timeout: float | None = 30.0, stream: bool = False):
    """Return a response object (caller-closed if streaming) or the parsed JSON.

    Raises :class:`OSError` if the daemon is unreachable.
    """
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        _ollama_url(path), method=method, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    resp = urllib.request.urlopen(req, timeout=timeout)  # pylint: disable=consider-using-with
    if stream:
        return resp
    try:
        raw = resp.read()
        return json.loads(raw) if raw else {}
    finally:
        resp.close()


def list_local_models() -> dict[str, int]:
    """Map of locally-pulled tag -> size in bytes. Empty dict if daemon down."""
    try:
        data = _ollama_request("GET", "/api/tags", timeout=5)
    except (OSError, urllib.error.URLError):
        return {}
    out: dict[str, int] = {}
    for m in data.get("models", []):
        name = m.get("name") or m.get("model")
        if name:
            out[name] = int(m.get("size") or 0)
    return out


def _handle_pull_event(tag: str, raw: bytes, last_status: str) -> str:
    """Parse one pull-stream event; print status if it changed. Returns new last."""
    try:
        evt = json.loads(raw)
    except json.JSONDecodeError:
        return last_status
    if "error" in evt:
        raise RuntimeError(f"ollama pull {tag}: {evt['error']}")
    status = evt.get("status", "")
    if status and status != last_status:
        print(f"    {status}", flush=True)
        return status
    return last_status


def pull_model(tag: str) -> None:
    """Stream-pull a model, printing the status line as it changes."""
    print(f"  pulling {tag} ...", flush=True)
    resp = _ollama_request("POST", "/api/pull",
                           {"name": tag, "stream": True},
                           timeout=None, stream=True)
    try:
        last_status = ""
        for raw in resp:
            last_status = _handle_pull_event(tag, raw, last_status)
    finally:
        resp.close()


def delete_model(tag: str) -> None:
    """Remove a model from the local ollama store."""
    _ollama_request("DELETE", "/api/delete", {"name": tag}, timeout=30)


# --------------------------------------------------------------------------- #
# Preflight: classify each unique tag as local/missing, render a report.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ModelStatus:
    tag: str
    local: bool
    size_bytes: int | None  # known only for local models


def _human_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} PB"  # unreachable, keeps linter happy


def _ollama_models_dir() -> Path:
    env = os.environ.get("OLLAMA_MODELS")
    return Path(env) if env else Path.home() / ".ollama" / "models"


def _disk_free(path: Path) -> int:
    """Free bytes on the FS holding `path`; 0 if the path doesn't exist."""
    probe = path if path.exists() else path.parent
    if not probe.exists():
        probe = Path.home()
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def inspect_preflight(tags: Iterable[str]) -> list[ModelStatus]:
    """One ModelStatus per *unique* tag, in first-seen order.

    Caller passes the iterable of tags the run wants (e.g.
    ``e.profile.model for e in entries``) -- this module doesn't depend on the
    comparison config types so it stays usable for any caller that has tags.
    """
    local = list_local_models()
    seen: dict[str, ModelStatus] = {}
    for tag in tags:
        if tag in seen:
            continue
        seen[tag] = ModelStatus(tag=tag, local=tag in local,
                                size_bytes=local.get(tag))
    return list(seen.values())


def format_preflight(statuses: list[ModelStatus],
                     mode: str | None = None) -> str:
    """Human-readable summary printed before the run."""
    name_w = max((len(s.tag) for s in statuses), default=10) + 2
    lines = ["Preflight:"]
    local_total = 0
    for s in statuses:
        if s.local:
            local_total += s.size_bytes or 0
            sz = _human_bytes(s.size_bytes)
            label = "LOCAL"
        else:
            sz = "(pull required)"
            label = "MISSING"
        lines.append(f"  {s.tag.ljust(name_w)} {label:<8} {sz}")
    free = _disk_free(_ollama_models_dir())
    lines.append("")
    lines.append(f"  local total: {_human_bytes(local_total)}")
    lines.append(f"  free disk:   {_human_bytes(free)}  "
                 f"({_ollama_models_dir()})")
    if mode:
        lines.append(f"  mode:        {mode}")
    return "\n".join(lines)
