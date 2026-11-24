"""Start the code-use namespace import while the browser is starting.

The namespace module pulls in the data-science stack, the action registry, and
the framework bindings. That import costs about 0.5 s on a cold interpreter,
which is close to the cost of launching Chrome. Both costs are unavoidable, so
run the import on a worker thread alongside browser startup instead of after
it.

The module itself imports nothing heavy, so callers can start the fetch before
they need the namespace.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_THREAD: threading.Thread | None = None
_ERROR: BaseException | None = None


def _import_namespace() -> None:
    global _ERROR
    try:
        import openbrowser.code_use.namespace  # noqa: F401 - imported for its side effect
    except BaseException as error:  # pragma: no cover - re-raised by the caller
        _ERROR = error


def start_namespace_import() -> threading.Thread:
    """Start the namespace import once and return its thread.

    Repeated calls return the same thread. The import itself is cached by
    Python, so joining a finished thread and importing normally costs nothing.
    """
    global _THREAD
    with _LOCK:
        if _THREAD is None:
            _THREAD = threading.Thread(
                target=_import_namespace,
                name="openbrowser-namespace-prewarm",
                daemon=True,
            )
            _THREAD.start()
        return _THREAD


def join_namespace_import(thread: threading.Thread) -> None:
    """Wait for a prewarm thread and re-raise its import error, if any."""
    thread.join()
    if _ERROR is not None:
        error = _ERROR
        globals()["_ERROR"] = None
        raise ImportError(f"Failed to import openbrowser.code_use.namespace: {error}") from error


def reset_prewarm_for_tests() -> None:
    """Clear the cached thread so a test can observe a cold import."""
    global _THREAD, _ERROR
    with _LOCK:
        _THREAD = None
        _ERROR = None
