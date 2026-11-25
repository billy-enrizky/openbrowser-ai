"""Console entry point for openbrowser.

The CLI keeps its fast paths before heavy imports so `--mcp`, `-c`, HTTP-only
scrapes, and daemon commands do not load the agent stack. This wrapper executes
`openbrowser.cli` as `__main__`, preserving that ordering without making a
normal library import of `openbrowser.cli` execute a command.
"""

import sys

from openbrowser.cli_modes import CliMode, select_cli_mode


class _LazyRunpy:
    """Resolve ``runpy`` only when a path that needs it runs."""

    def run_module(self, *args, **kwargs):
        """Import ``runpy`` and delegate to ``run_module``."""
        import runpy

        return runpy.run_module(*args, **kwargs)


def run_http_scrape_sync(*args, **kwargs):
    """Resolve the HTTP scrape runner on first use."""
    from openbrowser.scrape.http import run_http_scrape_sync as implementation

    return implementation(*args, **kwargs)


def emit_scrape_payload(*args, **kwargs):
    """Resolve the scrape payload writer on first use."""
    from openbrowser.scrape.output import emit_scrape_payload as implementation

    return implementation(*args, **kwargs)


def read_scrape_urls_file(*args, **kwargs):
    """Resolve the URL file reader on first use."""
    from openbrowser.scrape.output import read_scrape_urls_file as implementation

    return implementation(*args, **kwargs)


runpy = _LazyRunpy()

_SCRAPE_VALUE_OPTIONS = {
    "--format",
    "--selector",
    "--engine",
    "--wait-for",
    "--wait-until",
    "--timeout-ms",
    "--max-chars",
    "--concurrency",
    "--output",
    "--output-format",
    "--urls-file",
    "--render-mode",
    "--include-tag",
    "--exclude-tag",
}
_SCRAPE_FLAG_OPTIONS = {
    "--extract-links",
    "--lightweight",
    "--jsonl",
    "--json",
    "--only-main-content",
    "--text-only",
    "--no-css",
    "--css",
    "--no-images",
    "--images",
    "--no-fonts",
    "--fonts",
    "--no-media",
    "--media",
    "--no-browser",
}
_FORMATS = {"markdown", "text", "html", "links", "json"}
_OUTPUT_FORMATS = {"json", "jsonl"}


def _daemon_stop_fast_path(argv: list[str]) -> bool:
    """Stop or force-stop the daemon without importing the Click tree."""
    if select_cli_mode(argv[1:]) is not CliMode.DAEMON:
        return False
    args = argv[1:]
    if len(args) != 2 or args[1] not in {"stop", "force-stop"}:
        return False
    import asyncio

    from openbrowser.daemon.client import DaemonClient

    client = DaemonClient()
    operation = client.stop if args[1] == "stop" else client.force_stop
    response = asyncio.run(operation())
    if not response.success:
        print(response.output or response.error, file=sys.stderr)
        if response.error_code:
            print(f"Error code: {response.error_code}", file=sys.stderr)
        if response.hint:
            print(f"Hint: {response.hint}", file=sys.stderr)
        raise SystemExit(1)
    print(response.output or ("Daemon stopped" if args[1] == "stop" else "Daemon force-stopped"))
    return True


def _execute_code_fast_path(argv: list[str]) -> bool:
    """Run `-c` against a warm daemon without importing the Click tree."""
    if select_cli_mode(argv[1:]) is not CliMode.EXECUTE_CODE:
        return False
    try:
        code_index = argv.index("-c")
    except ValueError:
        return False
    code = argv[code_index + 1] if code_index + 1 < len(argv) else None
    if code is None:
        return False
    if code == "-":
        code = sys.stdin.read().strip()

    from openbrowser.daemon.thin_client import execute_code_sync

    response = execute_code_sync(code)
    if response is None:
        # No warm daemon. Start it through the full client without loading the
        # Click command tree or the agent stack.
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        import asyncio

        from openbrowser.daemon.client import execute_code_via_daemon

        response = asyncio.run(execute_code_via_daemon(code))
    if response.success:
        if response.output:
            print(response.output)
        return True
    print(response.output or response.error, file=sys.stderr)
    if response.error_code:
        print(f"Error code: {response.error_code}", file=sys.stderr)
    if response.hint:
        print(f"Hint: {response.hint}", file=sys.stderr)
    raise SystemExit(1)


def _parse_http_scrape_args(args: list[str]) -> dict | None:
    """Parse the documented HTTP scrape flags, or return None to use Click.

    The fast path only accepts flags it understands. Any unknown or malformed
    option returns None so the full Click command owns validation and help.
    """
    values: dict[str, object] = {}
    tag_values: dict[str, list[str]] = {"--include-tag": [], "--exclude-tag": []}
    urls: list[str] = []
    positional_only = False
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            positional_only = True
            index += 1
            continue
        if not positional_only and arg.startswith("-"):
            name, separator, inline_value = arg.partition("=")
            if name == "-o":
                name = "--output"
            if separator:
                if name not in _SCRAPE_VALUE_OPTIONS:
                    return None
                values[name] = inline_value
                index += 1
                continue
            if arg in _SCRAPE_FLAG_OPTIONS:
                values[arg] = True
                index += 1
                continue
            if name in _SCRAPE_VALUE_OPTIONS:
                if index + 1 >= len(args):
                    return None
                if name in tag_values:
                    tag_values[name].append(args[index + 1])
                else:
                    values[name] = args[index + 1]
                index += 2
                continue
            return None
        urls.append(arg)
        index += 1

    engine = values.get("--engine")
    if engine not in (None, "http", "auto"):
        return None
    if values.get("--no-browser"):
        engine = "http"
    urls_file = values.get("--urls-file")
    if urls_file:
        try:
            urls.extend(read_scrape_urls_file(urls_file))  # type: ignore[arg-type]
        except ValueError as error:
            raise SystemExit(f"Scrape failed: {error}") from error
    urls = [url.strip() for url in urls if url.strip()]
    if not urls:
        return None
    if engine != "http" and (len(urls) > 1 or urls_file):
        # Batch scraping over HTTP only exists for the explicit engine. Fall
        # back to the Click command, which owns the browser batch path.
        return None
    output_format = str(values.get("--format") or "markdown")
    if output_format not in _FORMATS:
        return None
    batch_output_format = values.get("--output-format")
    if batch_output_format is not None and batch_output_format not in _OUTPUT_FORMATS:
        return None
    try:
        concurrency = int(str(values.get("--concurrency") or 4))
        timeout_ms = int(str(values["--timeout-ms"])) if "--timeout-ms" in values else None
        max_chars = int(str(values["--max-chars"])) if "--max-chars" in values else None
    except (TypeError, ValueError):
        return None
    if concurrency < 1 or (timeout_ms is not None and timeout_ms < 1) or (max_chars is not None and max_chars < 0):
        return None
    return {
        "urls": urls,
        "batch_mode": len(urls) > 1 or bool(urls_file),
        "output_format": output_format,
        "selector": values.get("--selector"),
        "extract_links": bool(values.get("--extract-links")),
        "timeout_ms": timeout_ms,
        "max_chars": max_chars,
        "concurrency": concurrency,
        "jsonl": bool(values.get("--jsonl")),
        "batch_output_format": batch_output_format,
        "output": values.get("--output"),
        "as_json": bool(values.get("--json")),
        "engine": str(engine or "auto"),
        "only_main_content": bool(values.get("--only-main-content")),
        "include_tags": list(tag_values["--include-tag"]),
        "exclude_tags": list(tag_values["--exclude-tag"]),
    }


def _execute_scrape_fast_path(argv: list[str]) -> bool:
    """Run an explicit HTTP-only scrape without importing the Click tree."""
    if not argv or argv[0] != "scrape":
        return False
    parsed = _parse_http_scrape_args(argv[1:])
    if parsed is None:
        return False
    try:
        payload = run_http_scrape_sync(
            parsed["urls"],
            batch_mode=parsed["batch_mode"],
            output_format=parsed["output_format"],
            selector=parsed["selector"],
            extract_links=parsed["extract_links"],
            timeout_ms=parsed["timeout_ms"],
            max_chars=parsed["max_chars"],
            concurrency=parsed["concurrency"],
            engine=parsed["engine"],
            only_main_content=parsed["only_main_content"],
            include_tags=parsed["include_tags"] or None,
            exclude_tags=parsed["exclude_tags"] or None,
        )
        if payload is None:
            # The page needs a browser. The full command owns that path.
            return False
        emit_scrape_payload(
            payload,
            batch_mode=parsed["batch_mode"],
            jsonl=parsed["jsonl"],
            batch_output_format=parsed["batch_output_format"],
            output=parsed["output"],
            as_json=parsed["as_json"],
        )
    except SystemExit:
        raise
    except Exception as error:
        print(f"Scrape failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    return True


def cli() -> None:
    if _daemon_stop_fast_path(sys.argv):
        return
    if _execute_code_fast_path(sys.argv):
        return
    if _execute_scrape_fast_path(sys.argv[1:]):
        return
    runpy.run_module("openbrowser.cli", run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    cli()
