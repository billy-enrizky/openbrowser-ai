import asyncio
import base64
import importlib.util
import io
import json
import stat
import sys
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks/navigation_compare.py"
spec = importlib.util.spec_from_file_location(
    "navigation_compare_screenshot_tests", SCRIPT
)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def screenshot_png():
    image = Image.new("RGB", (128, 128))
    for y in range(128):
        for x in range(128):
            if x < 64 and y < 64:
                color = (255, 0, 0)
            elif x >= 64 and y < 64:
                color = (0, 255, 0)
            elif x < 64:
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            image.putpixel((x, y), color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def screenshot_request(arm="openbrowser"):
    run_id = "a" * 32
    nonce = f"{run_id}-{arm}-0"
    return {
        "op": "capture_screenshot",
        "schema": "openbrowser-screenshot-request/1",
        "attempt_id": f"{run_id}:{arm}:0",
        "url": f"http://127.0.0.1:12345/screenshot?nonce={nonce}",
        "nonce": nonce,
        "selector": "#proof",
        "output": "result.png",
    }


class FakeArm:
    def __init__(self, name="openbrowser", *, fail_capture=False):
        self.name = name
        self.calls = []
        self.fail_capture = fail_capture
        self.stopped = False

    async def start(self):
        self.calls.append("start")
        return {"owned": []}

    async def capture_screenshot(self, request):
        self.calls.append(request)
        if self.fail_capture:
            raise RuntimeError("screenshot capture failed")
        Path(request.output).write_bytes(screenshot_png())

    async def stop(self):
        self.calls.append("stop")
        self.stopped = True
        return {"errors": [], "identities": []}


def replies_from(stdout):
    return [
        json.loads(line[len(benchmark.PREFIX) :])
        for line in stdout.getvalue().splitlines()
        if line.startswith(benchmark.PREFIX)
    ]


async def run_worker(tmp_path, monkeypatch, messages, arm=None):
    arm = arm or FakeArm()
    run_root = tmp_path / "screenshot-worker"
    run_root.mkdir(mode=0o700)
    run_root.chmod(0o700)
    monkeypatch.chdir(run_root)
    stdin = io.StringIO("".join(json.dumps(message) + "\n" for message in messages))
    stdout = io.StringIO()
    monkeypatch.setattr(benchmark.sys, "stdin", stdin)
    monkeypatch.setattr(benchmark.sys, "stdout", stdout)
    monkeypatch.setattr(benchmark, "Arm", lambda _args: arm)
    args = SimpleNamespace(
        worker=arm.name,
        worker_mode=benchmark.WorkerMode.SCREENSHOT,
        run_root=str(run_root),
    )
    await benchmark.worker(args)
    return run_root, replies_from(stdout), arm


def test_capture_screenshot_rejects_extra_fields(tmp_path, monkeypatch):
    request = {**screenshot_request(), "duration_ns": 10}
    _run_root, replies, arm = asyncio.run(
        run_worker(
            tmp_path,
            monkeypatch,
            [{"op": "start"}, request, {"op": "stop"}],
        )
    )

    assert replies[0]["ok"] is True
    assert replies[1]["ok"] is False
    assert "unsupported fields" in replies[1]["error"]
    assert replies[2]["ok"] is True
    assert arm.calls == ["start", "stop"]
    assert arm.stopped


def test_capture_screenshot_reply_matches_request(tmp_path, monkeypatch):
    request = screenshot_request()
    run_root, replies, _arm = asyncio.run(
        run_worker(
            tmp_path,
            monkeypatch,
            [{"op": "start"}, request, {"op": "stop"}],
        )
    )
    parsed_request = benchmark.parse_screenshot_request("openbrowser", request)
    parsed_reply = benchmark.parse_screenshot_reply(parsed_request, replies[1])

    assert replies[1]["ok"] is True
    assert asdict(parsed_reply) == replies[1]["result"]
    assert parsed_reply.attempt_id == parsed_request.attempt_id
    assert parsed_reply.arm == parsed_request.arm
    assert parsed_reply.nonce == parsed_request.nonce
    assert parsed_reply.ready_marker == parsed_request.nonce
    assert parsed_reply.output == "result.png"
    assert (run_root / "result.png").is_file()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_id", "wrong:openbrowser:0"),
        ("arm", "raw-cdp"),
        ("nonce", "wrong-nonce"),
    ],
)
def test_capture_screenshot_rejects_reply_attempt_arm_or_nonce_mismatch(field, value):
    request = benchmark.parse_screenshot_request("openbrowser", screenshot_request())
    result = {
        "schema": "openbrowser-screenshot-reply/1",
        "attempt_id": request.attempt_id,
        "arm": request.arm,
        "nonce": request.nonce,
        "ready_marker": request.nonce,
        "output": request.output,
        "byte_count": 123,
        "sha256": "a" * 64,
    }
    result[field] = value

    with pytest.raises(ValueError):
        benchmark.parse_screenshot_reply(request, {"ok": True, "result": result})


def test_capture_screenshot_writes_verified_png_with_mode_0600(tmp_path, monkeypatch):
    request = screenshot_request()
    run_root, replies, _arm = asyncio.run(
        run_worker(
            tmp_path,
            monkeypatch,
            [{"op": "start"}, request, {"op": "stop"}],
        )
    )
    output = run_root / "result.png"

    assert replies[1]["ok"] is True
    assert stat.S_ISREG(output.lstat().st_mode)
    assert stat.S_IMODE(output.lstat().st_mode) == 0o600
    assert replies[1]["result"]["byte_count"] == output.stat().st_size
    assert replies[1]["result"]["sha256"] == benchmark.sha256(output)


def test_capture_screenshot_is_stopped_after_failure(tmp_path, monkeypatch):
    request = screenshot_request()
    arm = FakeArm(fail_capture=True)
    _run_root, replies, arm = asyncio.run(
        run_worker(
            tmp_path,
            monkeypatch,
            [{"op": "start"}, request, {"op": "stop"}],
            arm,
        )
    )

    assert replies[1]["ok"] is False
    assert "screenshot capture failed" in replies[1]["error"]
    assert replies[2]["ok"] is True
    assert arm.calls == [
        "start",
        benchmark.parse_screenshot_request("openbrowser", request),
        "stop",
    ]
    assert arm.stopped


def test_openbrowser_screenshot_uses_element_capture_after_ready_marker(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    request = benchmark.parse_screenshot_request("openbrowser", screenshot_request())
    calls = []

    class Browser:
        async def navigate_to(self, url, *, wait_until):
            calls.append(("navigate", url, wait_until))

        async def screenshot_element(self, selector, *, path, format):
            calls.append(("screenshot_element", selector, path, format))
            content = screenshot_png()
            Path(path).write_bytes(content)
            return content

    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.name = "openbrowser"
    arm.mode = benchmark.WorkerMode.SCREENSHOT
    arm.browser = Browser()

    async def ready(nonce):
        calls.append(("ready", nonce))

    arm.wait_for_screenshot_ready = ready
    content = asyncio.run(arm.capture_screenshot(request))

    assert len(content) > 0
    assert calls == [
        ("navigate", request.url, "load"),
        ("ready", request.nonce),
        ("screenshot_element", "#proof", "result.png", "png"),
    ]
    assert Path("result.png").read_bytes() == content


def test_screenshot_readiness_waits_for_matching_marker_and_loaded_image(monkeypatch):
    observations = [
        {"readyState": "complete", "readyMarker": "old-nonce", "imageReady": True},
        {"readyState": "complete", "readyMarker": "expected-nonce", "imageReady": True},
    ]
    calls = []
    arm = benchmark.Arm.__new__(benchmark.Arm)

    async def evaluate(expression):
        calls.append(expression)
        return json.dumps(observations.pop(0))

    async def no_delay(_seconds):
        return None

    arm.evaluate = evaluate
    monkeypatch.setattr(benchmark.asyncio, "sleep", no_delay)

    asyncio.run(arm.wait_for_screenshot_ready("expected-nonce"))

    assert len(calls) == 2
    assert calls == [benchmark.SCREENSHOT_OBSERVE, benchmark.SCREENSHOT_OBSERVE]


def test_raw_cdp_screenshot_sets_fixed_viewport_and_clip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    request = benchmark.parse_screenshot_request(
        "raw-cdp", screenshot_request("raw-cdp")
    )
    calls = []
    content = screenshot_png()

    class Cdp:
        async def send(self, method, params=None, session=None):
            calls.append((method, params, session))
            if method == "Page.captureScreenshot":
                return {"data": base64.b64encode(content).decode("ascii")}
            return {}

    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.name = "raw-cdp"
    arm.mode = benchmark.WorkerMode.SCREENSHOT
    arm.cdp = Cdp()
    arm.session = "worker-session"

    async def navigate(url):
        calls.append(("navigate", url))

    async def ready(nonce):
        calls.append(("ready", nonce))

    arm.navigate = navigate
    arm.wait_for_screenshot_ready = ready
    captured = asyncio.run(arm.capture_screenshot(request))

    assert captured == content
    assert Path("result.png").read_bytes() == content
    assert calls == [
        (
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 128,
                "height": 128,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
            "worker-session",
        ),
        ("navigate", request.url),
        ("ready", request.nonce),
        (
            "Page.captureScreenshot",
            {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": False,
                "clip": {"x": 0, "y": 0, "width": 128, "height": 128, "scale": 1},
            },
            "worker-session",
        ),
    ]
