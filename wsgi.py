"""
WSGI entry point for hosting (gunicorn wsgi:application).

The dashboard loads its data at import time, which can take a minute on a small
shared-CPU instance. Hosts like Render mark a deploy failed if nothing answers HTTP
quickly, so this wrapper starts answering right away (health check + a "loading"
page) and imports the real app in a background thread, then hands requests over.
"""
import sys
import threading
import time

_state = {"app": None, "error": None, "started": time.time()}

_LOADING = (
    b"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=4>"
    b"<title>Loading</title><body style=\"font:16px system-ui;margin:15vh auto;max-width:28em;"
    b"color:#333\"><h2>Rossmann-style demand forecasting</h2>"
    b"<p>The dashboard is starting up - this page refreshes automatically.</p>"
)


def _load():
    try:
        import app as dash_app

        _state["app"] = dash_app.server
        print(f"[wsgi] app ready after {time.time() - _state['started']:.1f}s", file=sys.stderr, flush=True)
    except BaseException as exc:  # noqa: BLE001 - surface any startup failure in the logs
        _state["error"] = exc
        import traceback

        traceback.print_exc()


threading.Thread(target=_load, daemon=True).start()


def application(environ, start_response):
    if environ.get("PATH_INFO") == "/healthz":
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]
    if _state["app"] is not None:
        return _state["app"](environ, start_response)
    if _state["error"] is not None:
        start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
        return [f"Startup failed: {_state['error']!r}".encode()]
    start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Retry-After", "4")])
    return [_LOADING]
