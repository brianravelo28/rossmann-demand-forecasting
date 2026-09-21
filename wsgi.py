"""
WSGI entry point for hosting (gunicorn wsgi:application).

The dashboard loads its data at import time, which can take a while on a small
shared-CPU instance. Hosts like Render mark a deploy failed if nothing answers HTTP
quickly, so this wrapper answers right away (health check + a "loading" page) and
imports the real app in a background thread, then hands requests over.

The loader is started per *process* (keyed on os.getpid()): a thread started before a
fork does not exist in the child, so relying on import-time startup alone can leave a
worker serving the loading page forever.
"""
import json
import os
import sys
import threading
import time

_lock = threading.Lock()
_state = {"app": None, "error": None, "started": time.time(), "pid": None}

_LOADING = (
    b"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=4>"
    b"<title>Loading</title><body style=\"font:16px system-ui;margin:15vh auto;max-width:28em;"
    b"color:#333\"><h2>Rossmann-style demand forecasting</h2>"
    b"<p>The dashboard is starting up - this page refreshes automatically.</p>"
)


def _load(pid):
    try:
        import app as dash_app

        if _state["pid"] == pid:
            _state["app"] = dash_app.server
        print(f"[wsgi] pid {pid}: app ready after {time.time() - _state['started']:.1f}s", file=sys.stderr, flush=True)
    except BaseException as exc:  # noqa: BLE001 - surface any startup failure in the logs
        _state["error"] = exc
        import traceback

        traceback.print_exc()


def _ensure_loader():
    pid = os.getpid()
    if _state["pid"] == pid:
        return
    with _lock:
        if _state["pid"] == pid:
            return
        _state.update(app=None, error=None, started=time.time(), pid=pid)
        print(f"[wsgi] pid {pid}: starting loader thread", file=sys.stderr, flush=True)
        threading.Thread(target=_load, args=(pid,), daemon=True).start()


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            return int(next(l.split()[1] for l in f if l.startswith("VmRSS"))) // 1024
    except (OSError, StopIteration):
        return None


_ensure_loader()


def application(environ, start_response):
    _ensure_loader()
    path = environ.get("PATH_INFO")
    if path == "/healthz":
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]
    if path == "/debugz":
        info = {
            "pid": os.getpid(),
            "loaded": _state["app"] is not None,
            "error": repr(_state["error"]) if _state["error"] else None,
            "seconds_since_loader_start": round(time.time() - _state["started"], 1),
            "rss_mb": _rss_mb(),
            "threads": threading.active_count(),
        }
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps(info).encode()]
    if _state["app"] is not None:
        return _state["app"](environ, start_response)
    if _state["error"] is not None:
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        return [f"Startup failed: {_state['error']!r}".encode()]
    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Retry-After", "4")])
    return [_LOADING]
