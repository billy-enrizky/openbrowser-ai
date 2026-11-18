from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from enum import StrEnum
from datetime import datetime, timezone
import ipaddress
import json
import html
import hashlib
import logging
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

from PIL import Image
from urllib.parse import parse_qs, quote, urlsplit

import latest_benchmark_preflight as benchmark_preflight
import navigation_compare as navigation


CLI_DESCRIPTION = 'Run and verify a local screenshot capability check without measuring latency.'
LOG = logging.getLogger('screenshot-capability')
SMOKE_ARMS = ('openbrowser', 'raw-cdp')
SCREENSHOT_ARMS = ('openbrowser', 'raw-cdp', 'lightpanda')
RUN_SCHEMA = 'openbrowser-screenshot-capability-run/1'
SMOKE_SCHEMA = 'openbrowser-screenshot-smoke/1'


class ScreenshotRunState(StrEnum):
    PREPARING = 'preparing'
    RUNNING = 'running'
    CLEANING = 'cleaning'
    COMPLETE = 'complete'
    INCOMPLETE = 'incomplete'


class ScreenshotAttemptState(StrEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    PASSED = 'passed'
    PIXEL_MISMATCH = 'pixel_mismatch'
    FAILED = 'failed'
    INCOMPLETE = 'incomplete'


class LightpandaCleanupState(StrEnum):
    INSPECTING = 'inspecting'
    TERMINATING = 'terminating'
    VERIFIED = 'verified'
    INCOMPLETE = 'incomplete'


_RUN_TRANSITIONS = {
    (ScreenshotRunState.PREPARING, 'running'): ScreenshotRunState.RUNNING,
    (ScreenshotRunState.PREPARING, 'incomplete'): ScreenshotRunState.INCOMPLETE,
    (ScreenshotRunState.RUNNING, 'cleaning'): ScreenshotRunState.CLEANING,
    (ScreenshotRunState.RUNNING, 'incomplete'): ScreenshotRunState.INCOMPLETE,
    (ScreenshotRunState.CLEANING, 'complete'): ScreenshotRunState.COMPLETE,
    (ScreenshotRunState.CLEANING, 'incomplete'): ScreenshotRunState.INCOMPLETE,
}
_ATTEMPT_TRANSITIONS = {
    (ScreenshotAttemptState.PENDING, 'running'): ScreenshotAttemptState.RUNNING,
    (ScreenshotAttemptState.RUNNING, 'passed'): ScreenshotAttemptState.PASSED,
    (ScreenshotAttemptState.RUNNING, 'pixel_mismatch'): ScreenshotAttemptState.PIXEL_MISMATCH,
    (ScreenshotAttemptState.RUNNING, 'failed'): ScreenshotAttemptState.FAILED,
    (ScreenshotAttemptState.RUNNING, 'incomplete'): ScreenshotAttemptState.INCOMPLETE,
}
_LIGHTPANDA_CLEANUP_TRANSITIONS = {
    (LightpandaCleanupState.INSPECTING, 'found_live_process'): LightpandaCleanupState.TERMINATING,
    (LightpandaCleanupState.INSPECTING, 'all_processes_retired'): LightpandaCleanupState.VERIFIED,
    (LightpandaCleanupState.INSPECTING, 'uncertain'): LightpandaCleanupState.INCOMPLETE,
    (LightpandaCleanupState.TERMINATING, 'retry_inventory'): LightpandaCleanupState.INSPECTING,
    (LightpandaCleanupState.TERMINATING, 'all_processes_retired'): LightpandaCleanupState.VERIFIED,
    (LightpandaCleanupState.TERMINATING, 'uncertain'): LightpandaCleanupState.INCOMPLETE,
}


def transition_run(state, event):
    try:
        return _RUN_TRANSITIONS[(ScreenshotRunState(state), event)]
    except (KeyError, ValueError) as error:
        raise ValueError(f'illegal screenshot run transition: {state!r} + {event!r}') from error


def transition_attempt(state, event):
    try:
        return _ATTEMPT_TRANSITIONS[(ScreenshotAttemptState(state), event)]
    except (KeyError, ValueError) as error:
        raise ValueError(f'illegal screenshot attempt transition: {state!r} + {event!r}') from error


def transition_lightpanda_cleanup(state, event):
    try:
        return _LIGHTPANDA_CLEANUP_TRANSITIONS[(LightpandaCleanupState(state), event)].value
    except (KeyError, ValueError) as error:
        raise ValueError(f'illegal Lightpanda cleanup transition: {state!r} + {event!r}') from error


def expected_rgb(index: int, quadrant: int) -> tuple[int, int, int]:
    if type(index) is not int or not 0 <= index < 6:
        raise ValueError('screenshot index must be an integer from 0 through 5')
    if type(quadrant) is not int or not 0 <= quadrant < 4:
        raise ValueError('screenshot quadrant must be an integer from 0 through 3')
    return (
        (17 + 37 * index + 71 * quadrant) % 256,
        (23 + 61 * index + 43 * quadrant) % 256,
        (41 + 29 * index + 97 * quadrant) % 256,
    )


def png_for_index(index: int) -> bytes:
    colors = [expected_rgb(index, quadrant) for quadrant in range(4)]
    image = Image.new('RGB', (64, 64))
    image.paste(colors[0], (0, 0, 32, 32))
    image.paste(colors[1], (32, 0, 64, 32))
    image.paste(colors[2], (0, 32, 32, 64))
    image.paste(colors[3], (32, 32, 64, 64))
    output = BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def inspect_png(content: bytes, index: int) -> dict:
    if not isinstance(content, bytes) or not content:
        return {'parseable': False, 'error': 'screenshot is empty or not bytes'}
    try:
        image = Image.open(BytesIO(content))
        if image.format != 'PNG':
            return {'parseable': False, 'error': 'screenshot format is not PNG'}
        image.load()
    except (OSError, ValueError) as error:
        return {'parseable': False, 'error': f'PNG decode failed: {type(error).__name__}: {error}'}

    width, height = image.size
    rgb = image.convert('RGB')
    points = ((16, 16), (48, 16), (16, 48), (48, 48))
    probes = []
    mismatches = []
    for quadrant, (x, y) in enumerate(points):
        expected = expected_rgb(index, quadrant)
        actual = rgb.getpixel((x, y)) if x < width and y < height else None
        matches = actual == expected
        probes.append({
            'quadrant': quadrant, 'x': x, 'y': y,
            'expected_rgb': list(expected),
            'actual_rgb': list(actual) if actual is not None else None,
            'matches': matches,
        })
        if not matches:
            mismatches.append(quadrant)
    failures = []
    if (width, height) != (128, 128):
        failures.append(f'dimensions={width}x{height}')
    if mismatches:
        failures.append('probes=' + ','.join(str(quadrant) for quadrant in mismatches))
    return {
        'parseable': True,
        'format': 'PNG',
        'width': width,
        'height': height,
        'dimension_ok': (width, height) == (128, 128),
        'probes': probes,
        'probe_count': 4,
        'mismatch_positions': mismatches,
        'mismatch_count': len(mismatches),
        'mismatch_signature': ';'.join(failures) if failures else None,
    }


def screenshot_work_counts(attempts):
    if not isinstance(attempts, list):
        attempts = []
    records = [attempt for attempt in attempts if isinstance(attempt, dict)]
    attempted = [attempt for attempt in records if attempt.get('state') != 'pending']
    completed = [
        attempt for attempt in attempted
        if isinstance(attempt.get('pixel'), dict)
        and attempt['pixel'].get('parseable') is True
        and attempt['pixel'].get('probe_count') == 4
    ]
    return {
        'attempted_work_count': len(attempted),
        'completed_work_count': len(completed),
        'successful_work_count': sum(attempt.get('state') == 'passed' for attempt in records),
        'pixel_mismatch_attempt_count': sum(
            attempt.get('state') == 'pixel_mismatch' for attempt in records
        ),
        'pixel_probe_count': sum(
            attempt['pixel'].get('probe_count', 0) for attempt in completed
        ),
        'pixel_probe_mismatch_count': sum(
            attempt['pixel'].get('mismatch_count', 0) for attempt in completed
        ),
        'error_count': sum(
            len(attempt['errors']) for attempt in records if isinstance(attempt.get('errors'), list)
        ),
    }


class ScreenshotRequestHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        return None

    def do_GET(self):
        parsed = urlsplit(self.path)
        values = parse_qs(parsed.query, keep_blank_values=True)
        nonce = values.get('nonce', [''])[0]
        status = 200
        content_type = 'text/html; charset=utf-8'
        if parsed.path == '/screenshot' and self.server.valid_nonce(nonce):
            body = self.server.page(nonce).encode()
        elif parsed.path == '/pixels.png' and self.server.valid_nonce(nonce):
            content_type = 'image/png'
            body = png_for_index(self.server.nonce_index(nonce))
        else:
            status, body = 404, b'not found'
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            status = 499
        finally:
            self.server.record_event({
                'method': 'GET', 'path': parsed.path, 'nonce': nonce, 'status': status,
            })


class ScreenshotFixture(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, run_id):
        self.run_id = run_id
        self.events = []
        self.lock = threading.Lock()
        super().__init__(('127.0.0.1', 0), ScreenshotRequestHandler)

    def nonce_index(self, nonce):
        prefix = f'{self.run_id}-'
        if not isinstance(nonce, str) or not nonce.startswith(prefix):
            return None
        tail = nonce[len(prefix):]
        for arm in SCREENSHOT_ARMS:
            arm_prefix = f'{arm}-'
            if tail.startswith(arm_prefix):
                raw_index = tail[len(arm_prefix):]
                if raw_index and all(character in '0123456789' for character in raw_index):
                    index = int(raw_index)
                    if 0 <= index < 6 and str(index) == raw_index:
                        return index
        return None

    def valid_nonce(self, nonce):
        return self.nonce_index(nonce) is not None

    def page(self, nonce):
        image_url = f'/pixels.png?nonce={quote(nonce, safe="")}'
        nonce_json = json.dumps(nonce)
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Screenshot fixture</title>
<link rel="icon" href="data:,">
<style>html,body{{margin:0;padding:0;width:128px;height:128px;overflow:hidden}}
#proof{{position:absolute;left:0;top:0;width:128px;height:128px;overflow:hidden}}
#proof img{{display:block;width:64px;height:64px}}</style></head>
<body><div id="proof"><img id="proof-image" alt="proof image" src="{html.escape(image_url, quote=True)}"></div>
<script>const nonce={nonce_json};const image=document.getElementById('proof-image');
const markReady=()=>{{if(image.complete&&image.naturalWidth===64&&image.naturalHeight===64)
document.documentElement.dataset.screenshotReady=nonce;}};
if(image.complete)markReady();else image.addEventListener('load',markReady,{{once:true}});</script>
</body></html>'''

    def record_event(self, event):
        with self.lock:
            self.events.append(dict(event))

    def snapshot(self, nonce):
        with self.lock:
            events = [dict(event) for event in self.events if event['nonce'] == nonce]
        return {
            'events': events,
            'image_requests': [event for event in events if event['path'] == '/pixels.png'],
        }


def arms_for_mode(mode: str) -> tuple[str, ...]:
    if mode == 'smoke':
        return SMOKE_ARMS
    if mode == 'run':
        return SCREENSHOT_ARMS
    raise ValueError(f'unsupported screenshot mode: {mode!r}')


def classify_arm_status(arm, attempts, controls, *, fixture_stopped, expected_attempts=6):
    if len(attempts) != expected_attempts or not fixture_stopped \
            or any(item.get('state') in ('pending', 'running', 'incomplete')
                   or item.get('cleanup_ok') is not True for item in attempts):
        return 'incomplete'
    states = [item.get('state') for item in attempts]
    if any(state == 'failed' for state in states):
        return 'failed'
    if arm in ('openbrowser', 'raw-cdp'):
        if any(state == 'pixel_mismatch' for state in states):
            return 'failed'
        return 'passed' if all(state == 'passed' for state in states) else 'failed'
    if arm != 'lightpanda':
        raise ValueError(f'unsupported screenshot arm: {arm!r}')
    if controls.get('openbrowser') != 'passed' or controls.get('raw-cdp') != 'passed':
        return 'inconclusive'
    if all(state == 'passed' for state in states):
        return 'passed'
    if all(state == 'pixel_mismatch' for state in states):
        signatures = [item.get('mismatch_signature') for item in attempts]
        if all(item.get('parseable_png') is True and item.get('image_request_ok') is True
               and item.get('cleanup_ok') is True for item in attempts) \
                and signatures[0] is not None and all(signature == signatures[0] for signature in signatures):
            return 'unsupported'
    if any(state in ('passed', 'pixel_mismatch') for state in states):
        return 'inconclusive'
    return 'failed'


def screenshot_run_exit_code(manifest) -> int:
    if not isinstance(manifest, dict):
        return 1
    mode = manifest.get('mode')
    expected_schema = SMOKE_SCHEMA if mode == 'smoke' else RUN_SCHEMA if mode == 'run' else None
    if expected_schema is None or manifest.get('schema') != expected_schema:
        return 1
    cleanup = manifest.get('cleanup')
    fixture = manifest.get('fixture')
    if manifest.get('state') != 'complete' or manifest.get('errors') != [] \
            or not isinstance(cleanup, dict) \
            or set(cleanup) != {'ok', 'attempts_ok', 'runtime_roots_removed'} \
            or any(cleanup.get(key) is not True for key in cleanup) \
            or not isinstance(fixture, dict) \
            or set(fixture) != {'started', 'stopped', 'socket_closed', 'thread_stopped'} \
            or any(fixture.get(key) is not True for key in fixture):
        return 1
    attempts = manifest.get('attempts')
    if not isinstance(attempts, list):
        return 1
    expected_arms = SMOKE_ARMS if mode == 'smoke' else SCREENSHOT_ARMS if mode == 'run' else ()
    expected_per_arm = 1 if mode == 'smoke' else 6 if mode == 'run' else 0
    if not expected_arms or len(attempts) != len(expected_arms) * expected_per_arm:
        return 1
    if any(not isinstance(attempt, dict) for attempt in attempts):
        return 1
    expected_keys = [(arm, index) for arm in expected_arms for index in range(expected_per_arm)]
    actual_keys = [(attempt.get('arm'), attempt.get('index')) for attempt in attempts]
    if actual_keys != expected_keys:
        return 1
    if any(attempt.get('arm') not in expected_arms or type(attempt.get('index')) is not int
           or attempt.get('state') not in ('passed', 'pixel_mismatch', 'failed')
           or attempt.get('cleanup_ok') is not True
           or attempt.get('runtime_root_removed') is not True for attempt in attempts):
        return 1
    if any(sum(attempt.get('arm') == arm for attempt in attempts) != expected_per_arm
           for arm in expected_arms):
        return 1
    if mode == 'smoke' and any(attempt.get('state') != 'passed' for attempt in attempts):
        return 1
    return 0


def result_root_for_output(value, mode):
    physical_root, output = navigation.physical_profile_path(value)
    if mode == 'smoke':
        if output.parent.name != 'smoke' or output.name != 'screenshot':
            raise ValueError('smoke output must be RESULT_ROOT/smoke/screenshot')
        result_root = output.parent.parent
    elif mode == 'run':
        if output.name not in ('screenshot', 'screenshot-final'):
            raise ValueError('run output must be RESULT_ROOT/screenshot or screenshot-final')
        result_root = output.parent
    else:
        raise ValueError(f'unsupported screenshot mode: {mode!r}')
    if result_root.parent != physical_root:
        raise ValueError('RESULT_ROOT must be a direct child of physical /tmp')
    result_root = navigation.validate_profile_result_root(result_root)
    if output.exists() or output.is_symlink():
        raise ValueError('screenshot output must be new')
    return result_root, output


def result_root_for_run(value, mode):
    physical_root, run_root = navigation.physical_profile_path(value)
    if mode == 'verify-smoke':
        if run_root.parent.name != 'smoke' or run_root.name != 'screenshot':
            raise ValueError('smoke run root must be RESULT_ROOT/smoke/screenshot')
        result_root = run_root.parent.parent
    elif mode == 'verify':
        if run_root.name not in ('screenshot', 'screenshot-final'):
            raise ValueError('run root must be RESULT_ROOT/screenshot or screenshot-final')
        result_root = run_root.parent
    else:
        raise ValueError(f'unsupported screenshot mode: {mode!r}')
    if result_root.parent != physical_root:
        raise ValueError('RESULT_ROOT must be a direct child of physical /tmp')
    result_root = navigation.validate_profile_result_root(result_root)
    try:
        info = run_root.lstat()
    except OSError as error:
        raise ValueError(f'screenshot run root does not exist: {run_root}') from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 \
            or info.st_uid != os.geteuid():
        raise ValueError('screenshot run root must be a private directory')
    return result_root, run_root


def read_json_artifact(result_root, value, name):
    physical_root, path = navigation.physical_profile_path(value)
    if physical_root != result_root.parent or not path.is_relative_to(result_root):
        raise ValueError(f'{name} must be below RESULT_ROOT')
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f'{name} is unavailable: {path}') from error
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 \
            or info.st_uid != os.geteuid():
        raise ValueError(f'{name} must be a private regular file with mode 0600')
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'{name} is invalid JSON: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a JSON object')
    return path, value


def validate_screenshot_inputs(args, result_root):
    benchmark_preflight._require_no_aws()
    identity_path, identity = read_json_artifact(
        result_root, args.candidate_identity, 'candidate identity',
    )
    version_path, versions = read_json_artifact(
        result_root, args.version_manifest, 'version manifest',
    )
    preflight_path, preflight = read_json_artifact(
        result_root, args.preflight, 'screenshot preflight',
    )
    expected_phase = (
        'screenshot-smoke' if args.command == 'smoke'
        else 'screenshot-final' if Path(args.output).name == 'screenshot-final'
        else 'screenshot-run'
    )
    if preflight.get('schema') != 'openbrowser-latest-preflight/1' \
            or preflight.get('phase') != expected_phase \
            or preflight.get('outcome') != 'passed' or preflight.get('errors') != []:
        raise ValueError(f'{expected_phase} preflight did not pass')
    identity_hash = navigation.sha256(identity_path)
    version_hash = navigation.sha256(version_path)
    if versions.get('schema') != 'openbrowser-latest-version-manifest/1' \
            or versions.get('candidate_identity_sha256') != identity_hash:
        raise ValueError('version manifest does not match the candidate identity')
    if preflight.get('candidate_identity_sha256') != identity_hash \
            or preflight.get('version_manifest_sha256') != version_hash:
        raise ValueError('screenshot preflight does not match the frozen identities')
    frozen_hash = None
    if args.command == 'run':
        frozen_path, frozen = read_json_artifact(
            result_root, args.frozen_harness, 'frozen harness',
        )
        tool_manifest_path = benchmark_preflight._artifact_file(
            result_root / 'tool-manifest.json', result_root, 'tool manifest',
        )
        candidate_freeze_path = benchmark_preflight._artifact_file(
            result_root / 'candidate-freeze.txt', result_root, 'candidate dependency freeze',
        )
        browser_use_freeze_path = benchmark_preflight._artifact_file(
            result_root / 'browser-use-freeze.txt', result_root, 'browser-use dependency freeze',
        )
        tool_sha = benchmark_preflight.sha256_file(tool_manifest_path)
        if versions.get('tool_manifest_sha256') != tool_sha:
            raise ValueError('tool manifest does not match the sealed version manifest')
        frozen = benchmark_preflight._validate_frozen_harness(
            frozen,
            identity_sha=identity_hash,
            candidate_freeze_sha=benchmark_preflight.sha256_file(candidate_freeze_path),
            browser_use_freeze_sha=benchmark_preflight.sha256_file(browser_use_freeze_path),
            tool_manifest_sha=tool_sha,
        )
        if frozen.get('version_manifest_sha256') != version_hash \
                or frozen.get('executed_file_sha256') != identity.get('executed_files'):
            raise ValueError('frozen harness does not match the sealed version and runner files')
        benchmark_preflight.validate_frozen_smoke_preflight_hashes(result_root, frozen)
        frozen_hash = navigation.sha256(frozen_path)

    identity = benchmark_preflight._validate_identity(identity)
    versions = benchmark_preflight._validate_version_manifest(versions)
    preflight = benchmark_preflight._validate_preflight_result(preflight)
    if any(path.parent != result_root for path in (identity_path, version_path, preflight_path)) \
            or preflight_path.name != f'preflight-{expected_phase}.json':
        raise ValueError('screenshot manifests must use their documented direct-child paths')
    if not preflight['checks'] or any(check['state'] != 'passed' for check in preflight['checks']):
        raise ValueError('screenshot preflight must contain only passed checks')
    source = benchmark_preflight._source_root(args.source, result_root=result_root)
    snapshot = benchmark_preflight._git_snapshot(source, result_root=result_root)
    if any(identity.get(key) != value for key, value in snapshot.items()):
        raise ValueError('screenshot source differs from the sealed candidate identity')
    if benchmark_preflight._hash_executed_files(source) != identity['executed_files']:
        raise ValueError('screenshot runner or verifier files changed after identity creation')
    python = benchmark_preflight._inspect_python(args.python, root=result_root, require_current=True)
    if python != identity['candidate_python']:
        raise ValueError('screenshot Python differs from the sealed candidate identity')
    tools_path = benchmark_preflight._artifact_file(
        result_root / 'tool-manifest.json', result_root, 'tool manifest',
    )
    tools = benchmark_preflight.load_tool_manifest(tools_path, result_root=result_root)
    if navigation.sha256(tools_path) != versions['tool_manifest_sha256']:
        raise ValueError('screenshot tools differ from the sealed version manifest')
    for name, tool in (('chrome', 'chrome_for_testing'), ('lightpanda', 'lightpanda')):
        supplied = getattr(args, name, None)
        if supplied is not None and Path(supplied).expanduser().absolute() != Path(tools['tools'][tool]['path']):
            raise ValueError(f'screenshot {name} path differs from the sealed tool manifest')
        if supplied is None and (name == 'chrome' or args.command == 'run'):
            raise ValueError(f'screenshot {name} executable is required')
    return {
        'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'preflight_sha256': navigation.sha256(preflight_path),
        'frozen_harness_sha256': frozen_hash,
        'tool_manifest_sha256': versions.get('tool_manifest_sha256'),
        'candidate_commit': identity.get('candidate_commit'),
    }


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            descriptor = None
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def create_private_child(parent, name):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('private directory name must be a basename')
    path = Path(parent) / name
    if path.exists() or path.is_symlink():
        raise ValueError(f'private directory already exists: {path.name}')
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    navigation.validate_private_directory(path)
    return path


def screenshot_request(run_id, arm, index, port):
    nonce = f'{run_id}-{arm}-{index}'
    return {
        'op': 'capture_screenshot',
        'schema': 'openbrowser-screenshot-request/1',
        'attempt_id': f'{run_id}:{arm}:{index}',
        'url': f'http://127.0.0.1:{port}/screenshot?nonce={nonce}',
        'nonce': nonce,
        'selector': '#proof',
        'output': 'result.png',
    }


@dataclass(frozen=True)
class LightpandaScreenshotRequest:
    attempt_id: str
    url: str
    nonce: str


def parse_lightpanda_screenshot_request(value):
    if not isinstance(value, dict):
        raise ValueError('Lightpanda screenshot request must be an object')
    expected_keys = {'op', 'schema', 'attempt_id', 'url', 'nonce', 'selector', 'output'}
    if set(value) != expected_keys:
        raise ValueError('Lightpanda screenshot request has unexpected or missing fields')
    if value['op'] != 'capture_screenshot' or value['schema'] != 'openbrowser-screenshot-request/1':
        raise ValueError('Lightpanda screenshot request operation or schema is invalid')
    attempt_id = value['attempt_id']
    nonce = value['nonce']
    url = value['url']
    if not all(isinstance(item, str) for item in (attempt_id, nonce, url)):
        raise ValueError('Lightpanda screenshot attempt, nonce, and URL must be strings')
    parts = attempt_id.split(':')
    if len(parts) != 3 or len(parts[0]) != 32 \
            or any(character not in '0123456789abcdef' for character in parts[0]) \
            or parts[1] != 'lightpanda' or not parts[2] \
            or any(character not in '0123456789' for character in parts[2]):
        raise ValueError('Lightpanda screenshot attempt_id is invalid')
    index = int(parts[2])
    if str(index) != parts[2] or not 0 <= index < 6:
        raise ValueError('Lightpanda screenshot attempt index must be from 0 through 5')
    expected_nonce = f'{parts[0]}-lightpanda-{index}'
    if nonce != expected_nonce:
        raise ValueError('Lightpanda screenshot nonce does not match its attempt_id')
    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        port = parsed_url.port
        loopback = hostname is not None and ipaddress.ip_address(hostname).is_loopback
    except ValueError as error:
        raise ValueError('Lightpanda screenshot URL is invalid') from error
    if parsed_url.scheme != 'http' or not loopback or port is None \
            or parsed_url.path != '/screenshot' or parsed_url.username is not None \
            or parsed_url.password is not None or parsed_url.fragment \
            or parse_qs(parsed_url.query, keep_blank_values=True) != {'nonce': [nonce]}:
        raise ValueError('Lightpanda screenshot URL must be the nonce-bound loopback fixture')
    if value['selector'] != '#proof' or value['output'] != 'result.png':
        raise ValueError('Lightpanda screenshot selector or output is invalid')
    return LightpandaScreenshotRequest(attempt_id, url, nonce)


def image_request_proof(fixture, nonce):
    events = fixture.snapshot(nonce)['image_requests']
    successful = [event for event in events if event['method'] == 'GET' and event['status'] == 200]
    return {
        'nonce': nonce,
        'image_requests': events,
        'count': len(events),
        'successful_count': len(successful),
        'ok': len(events) == 1 and len(successful) == 1,
    }


def verify_png_file(path, index, reply=None, *, normalize_mode=False):
    path = Path(path)
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f'screenshot PNG is missing: {error}') from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('screenshot output is not a regular file')
    if normalize_mode:
        os.chmod(path, 0o600, follow_symlinks=False)
        info = path.lstat()
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.geteuid():
        raise ValueError('screenshot output must have mode 0600 and belong to the current user')
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if reply is not None and (len(content) != reply.byte_count or digest != reply.sha256):
        raise ValueError('parent screenshot bytes do not match the worker reply')
    pixel = inspect_png(content, index)
    return {
        'file': 'result.png',
        'byte_count': len(content),
        'sha256': digest,
        'pixel': pixel,
    }


def new_runtime_root():
    physical_root = Path('/tmp').resolve(strict=True)
    runtime_root = Path(tempfile.mkdtemp(prefix='nb-screenshot-', dir=str(physical_root)))
    runtime_root.chmod(0o700)
    token = uuid.uuid4().hex
    navigation.atomic_profile_bytes(runtime_root / '.benchmark-owner', token.encode())
    return runtime_root, token


async def execute_attempt(args, fixture, attempt_dir, attempt):
    if attempt['arm'] == 'lightpanda':
        return await execute_lightpanda_attempt(args, fixture, attempt_dir, attempt)
    return await execute_browser_attempt(args, fixture, attempt_dir, attempt)


async def execute_browser_attempt(args, fixture, attempt_dir, attempt):
    arm = attempt['arm']
    index = attempt['index']
    request_value = attempt['request']
    request = navigation.parse_screenshot_request(arm, request_value)
    runtime_root, token = new_runtime_root()
    attempt['runtime_root'] = str(runtime_root)
    handle = None
    errors = []
    png = None
    pixel = {'parseable': False, 'probe_count': 0, 'mismatch_count': 0,
             'mismatch_signature': None}
    reply = None
    request_proof = {'nonce': request.nonce, 'image_requests': [], 'count': 0,
                     'successful_count': 0, 'ok': False}
    cleanup = {'ok': False, 'worker_identity': {'state': 'unknown'},
               'post_worker_exit_identities': [], 'fallback_used': False}
    runtime_root_removed = False
    try:
        log_directory = create_private_child(attempt_dir, 'logs')
        launch = navigation.make_worker_launch(
            arm=arm, mode=navigation.WorkerMode.SCREENSHOT,
            source_root=Path(args.source).expanduser().resolve(),
            candidate_python=Path(args.python).expanduser().absolute(),
            browser_use_python=None,
            runtime_root=runtime_root, worker_root=runtime_root / arm,
            working_directory=attempt_dir, log_directory=log_directory,
            log_prefix='worker', owner_token=token,
            chrome=Path(args.chrome).expanduser().absolute(),
        )
        handle = await navigation.launch_worker(launch)
        attempt['worker_identity'] = handle.worker_identity
        response, start_errors = await navigation.start_worker(handle)
        if start_errors or not isinstance(response, dict) or response.get('ok') is not True:
            raise RuntimeError(response.get('error', repr(start_errors)) if isinstance(response, dict)
                               else repr(start_errors))
        response = await navigation.send_operation(handle, request_value, 60)
        reply = navigation.parse_screenshot_reply(request, response)
        png = verify_png_file(attempt_dir / 'result.png', index, reply)
        pixel = png['pixel']
        if not pixel['parseable']:
            errors.append(pixel.get('error', 'screenshot PNG could not be parsed'))
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        try:
            request_proof = image_request_proof(fixture, request.nonce)
            if not request_proof['ok']:
                errors.append('screenshot fixture image request was not exactly one successful request')
        except Exception as error:
            errors.append(f'screenshot fixture proof failed: {type(error).__name__}: {error}')
        if handle is not None:
            try:
                cleanup = await navigation.cleanup_worker(handle)
            except Exception as error:
                errors.append(f'worker cleanup failed: {type(error).__name__}: {error}')
                cleanup = {'ok': False, 'error': errors[-1],
                           'worker_identity': {**handle.worker_identity, 'state': 'unknown'},
                           'post_worker_exit_identities': [], 'fallback_used': False}
            finally:
                handle.stdout_log.close()
                handle.stderr_log.close()
        states = [cleanup.get('worker_identity', {}).get('state', 'unknown')]
        states.extend(identity.get('state', 'unknown') for identity in cleanup.get('post_worker_exit_identities', []))
        runtime_root_removed = cleanup.get('ok') is True \
            and navigation.removal_allowed(runtime_root, token, states)
        if runtime_root_removed:
            shutil.rmtree(runtime_root)
        else:
            cleanup['ok'] = False
            errors.append('screenshot worker cleanup or runtime-root removal is unverified')

    attempt_state = 'incomplete' if cleanup.get('ok') is not True else 'failed'
    if cleanup.get('ok') is True and not errors:
        attempt_state = 'pixel_mismatch' if pixel.get('mismatch_signature') else 'passed'
    return {
        'state': attempt_state,
        'cleanup_ok': cleanup.get('ok') is True,
        'runtime_root_removed': runtime_root_removed,
        'worker_reply': asdict(reply) if reply is not None else None,
        'worker_cleanup': cleanup,
        'request_proof': request_proof,
        'png': png,
        'pixel': pixel,
        'parseable_png': pixel.get('parseable') is True,
        'image_request_ok': request_proof['ok'],
        'mismatch_signature': pixel.get('mismatch_signature'),
        'errors': errors,
    }


def open_private_log(path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.chmod(path, 0o600)
        return os.fdopen(descriptor, 'w')
    except BaseException:
        os.close(descriptor)
        raise


def lightpanda_process_group_members(process_group_id, runtime_root, token):
    if type(process_group_id) is not int or process_group_id <= 0:
        return [], ['Lightpanda process group ID is invalid']
    expected_owner_root = str(Path(runtime_root) / 'lightpanda')
    identities = {}
    errors = []
    try:
        for process in navigation.psutil.process_iter():
            try:
                pid = process.pid
                if os.getpgid(pid) != process_group_id:
                    continue
            except (ProcessLookupError, navigation.psutil.NoSuchProcess):
                continue
            except (OSError, navigation.psutil.Error) as error:
                if isinstance(error, PermissionError):
                    try:
                        if process.uids().real != os.geteuid():
                            continue
                    except navigation.psutil.Error:
                        continue
                errors.append(
                    f'cannot inspect Lightpanda process group member: {type(error).__name__}'
                )
                continue
            try:
                identity = navigation.record(process, token)
            except (OSError, navigation.psutil.Error) as error:
                errors.append(
                    f'cannot record Lightpanda process group member: {type(error).__name__}'
                )
                continue
            identity_errors = navigation.profile_identity_errors(identity, state_required=False)
            if identity_errors or identity.get('owner_token') != token \
                    or identity.get('owner_root') != expected_owner_root:
                errors.append('Lightpanda process group contains a member without matching ownership proof')
                continue
            identities[(identity['pid'], identity['created'])] = identity
    except (OSError, navigation.psutil.Error) as error:
        errors.append(f'Lightpanda process group inventory failed: {type(error).__name__}')
    return list(identities.values()), errors


async def cleanup_lightpanda_process_group(
    process_group_id, runtime_root, token, worker_identity, known_children, *, fallback_used,
):
    expected_owner_root = str(Path(runtime_root) / 'lightpanda')
    errors = []
    identities = {}

    def remember(identity, where):
        if not isinstance(identity, dict):
            errors.append(f'{where} identity is not an object')
            return None
        identity_errors = navigation.profile_identity_errors(identity, state_required=False)
        if identity_errors or identity.get('owner_token') != token \
                or identity.get('owner_root') != expected_owner_root:
            errors.append(f'{where} ownership proof is invalid')
            return None
        key = (identity['pid'], identity['created'])
        identities[key] = identity
        return key

    parent_key = remember(worker_identity, 'Lightpanda parent')
    if parent_key is None or not isinstance(worker_identity, dict) \
            or worker_identity.get('pid') != process_group_id:
        errors.append('Lightpanda process group is not bound to the recorded parent')
    for child in known_children:
        remember(child, 'Lightpanda child')

    cleanup_state = LightpandaCleanupState.INSPECTING
    for _attempt in range(3):
        members, inventory_errors = lightpanda_process_group_members(
            process_group_id, runtime_root, token,
        )
        errors.extend(inventory_errors)
        for member in members:
            remember(member, 'Lightpanda process group member')
        alive = []
        for identity in identities.values():
            try:
                identity_state = navigation.identity_state(identity)
            except Exception as error:
                identity_state = 'unknown'
                errors.append(f'Lightpanda identity check failed: {type(error).__name__}')
            if identity_state == 'alive':
                alive.append(identity)
            elif identity_state not in ('absent', 'reused'):
                errors.append('Lightpanda process identity is not proven retired')
        if not alive:
            break
        cleanup_state = LightpandaCleanupState(
            transition_lightpanda_cleanup(cleanup_state, 'found_live_process'),
        )
        try:
            await navigation.terminate_owned(alive, token, grace=0.25)
        except Exception as error:
            errors.append(f'Lightpanda process termination failed: {type(error).__name__}')
            break
        if _attempt < 2:
            cleanup_state = LightpandaCleanupState(
                transition_lightpanda_cleanup(cleanup_state, 'retry_inventory'),
            )
            await asyncio.sleep(0.05)

    final_members, final_inventory_errors = lightpanda_process_group_members(
        process_group_id, runtime_root, token,
    )
    errors.extend(final_inventory_errors)
    for member in final_members:
        remember(member, 'Lightpanda process group member')

    states = {}
    for key, identity in identities.items():
        try:
            states[key] = navigation.identity_state(identity)
        except Exception as error:
            states[key] = 'unknown'
            errors.append(f'Lightpanda identity check failed: {type(error).__name__}')
    worker_state = states.get(parent_key, 'unknown')
    child_records = [
        {**identity, 'state': states.get(key, 'unknown')}
        for key, identity in identities.items() if key != parent_key
    ]
    if worker_state not in ('absent', 'reused'):
        errors.append('Lightpanda parent is not proven retired')
    if any(record['state'] not in ('absent', 'reused') for record in child_records):
        errors.append('Lightpanda child is not proven retired')
    unique_errors = list(dict.fromkeys(errors))
    verified = not unique_errors and worker_state in ('absent', 'reused') \
        and all(record['state'] in ('absent', 'reused') for record in child_records)
    if verified:
        cleanup_state = LightpandaCleanupState(
            transition_lightpanda_cleanup(cleanup_state, 'all_processes_retired'),
        )
    else:
        cleanup_state = LightpandaCleanupState(
            transition_lightpanda_cleanup(cleanup_state, 'uncertain'),
        )
    return {
        'ok': cleanup_state is LightpandaCleanupState.VERIFIED,
        'fallback_used': fallback_used,
        'process_group_id': process_group_id,
        'process_state': cleanup_state.value,
        'worker_identity': ({**worker_identity, 'state': worker_state}
                            if isinstance(worker_identity, dict) else {'state': 'unknown'}),
        'post_worker_exit_identities': child_records,
        'errors': unique_errors,
    }


async def execute_lightpanda_attempt(args, fixture, attempt_dir, attempt):
    index = attempt['index']
    request_value = attempt['request']
    request = parse_lightpanda_screenshot_request(request_value)
    runtime_root, token = new_runtime_root()
    attempt['runtime_root'] = str(runtime_root)
    log_directory = None
    stdout_log = None
    stderr_log = None
    process = None
    identity = None
    identity_errors = []
    children = {}
    errors = []
    returncode = None
    timed_out = False
    process_group_id = None

    try:
        log_directory = create_private_child(attempt_dir, 'logs')
        launch = navigation.make_worker_launch(
            arm='lightpanda', mode=navigation.WorkerMode.NAVIGATION,
            source_root=Path(args.source).expanduser().resolve(),
            candidate_python=Path(args.python).expanduser().absolute(),
            browser_use_python=None,
            runtime_root=runtime_root, worker_root=runtime_root / 'lightpanda',
            working_directory=attempt_dir, log_directory=log_directory,
            log_prefix='lightpanda', owner_token=token,
            lightpanda=Path(args.lightpanda).expanduser().absolute(),
        )
        environment = navigation.worker_environment(launch)
        navigation.prepare_worker_directories(environment)
        environment['NAV_BENCH_SCREENSHOT_URL'] = request.url
        environment['NAV_BENCH_SCREENSHOT_NONCE'] = request.nonce
        stdout_log = open_private_log(log_directory / 'lightpanda.stdout.log')
        stderr_log = open_private_log(log_directory / 'lightpanda.stderr.log')
        script = Path(__file__).with_name('screenshot_lightpanda.js').resolve()
        process = await asyncio.create_subprocess_exec(
            str(launch.lightpanda), 'run', '--load-resources', 'image', str(script),
            cwd=attempt_dir,
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=stdout_log,
            stderr=stderr_log,
            start_new_session=True,
        )
        process_group_id = process.pid
        attempt['worker_identity'] = navigation.record(
            navigation.psutil.Process(process.pid), token,
        )
        identity = attempt['worker_identity']
        identity_errors = navigation.profile_identity_errors(identity, state_required=False)
        if identity_errors:
            errors.append(f'Lightpanda process identity is incomplete: {identity_errors!r}')
        waiter = asyncio.create_task(process.wait())
        deadline = asyncio.get_running_loop().time() + 90
        while not waiter.done() and asyncio.get_running_loop().time() < deadline:
            if not identity_errors:
                for child in navigation.descendant_identities(identity, token):
                    child_errors = navigation.profile_identity_errors(child, state_required=False)
                    if child_errors:
                        errors.append(f'Lightpanda child identity is incomplete: {child_errors!r}')
                    else:
                        children[(child['pid'], child['created'])] = child
            await asyncio.wait((waiter,), timeout=.05)
        if not waiter.done():
            timed_out = True
            errors.append('Lightpanda screenshot timed out')
            if not identity_errors:
                for child in navigation.descendant_identities(identity, token):
                    if not navigation.profile_identity_errors(child, state_required=False):
                        children[(child['pid'], child['created'])] = child
                await navigation.terminate_owned([identity, *children.values()], token)
                try:
                    returncode = await asyncio.wait_for(process.wait(), 5)
                except Exception as error:
                    errors.append(f'Lightpanda did not exit after verified termination: {type(error).__name__}: {error}')
        else:
            returncode = waiter.result()
        if returncode not in (None, 0):
            errors.append(f'Lightpanda exited with status {returncode}')
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        if process is not None and identity is not None and not identity_errors:
            for child in navigation.descendant_identities(identity, token):
                if not navigation.profile_identity_errors(child, state_required=False):
                    children[(child['pid'], child['created'])] = child
        if stdout_log is not None:
            stdout_log.close()
        if stderr_log is not None:
            stderr_log.close()

    try:
        cleanup = await cleanup_lightpanda_process_group(
            process_group_id, runtime_root, token, identity, list(children.values()),
            fallback_used=timed_out,
        )
    except Exception as error:
        cleanup = {
            'ok': False,
            'fallback_used': timed_out,
            'process_group_id': process_group_id,
            'worker_identity': ({**identity, 'state': 'unknown'} if identity is not None
                                else {'state': 'unknown'}),
            'post_worker_exit_identities': [],
            'errors': [f'Lightpanda cleanup failed: {type(error).__name__}: {error}'],
        }
    if identity_errors:
        cleanup['ok'] = False
        cleanup['errors'] = list(dict.fromkeys([*cleanup['errors'], *identity_errors]))
    cleanup_ok = cleanup.get('ok') is True and returncode is not None
    states = [cleanup.get('worker_identity', {}).get('state', 'unknown')]
    states.extend(
        child.get('state', 'unknown')
        for child in cleanup.get('post_worker_exit_identities', [])
    )
    runtime_root_removed = cleanup_ok and navigation.removal_allowed(runtime_root, token, states)
    if runtime_root_removed:
        try:
            shutil.rmtree(runtime_root)
        except OSError as error:
            cleanup['ok'] = False
            runtime_root_removed = False
            errors.append(f'Lightpanda runtime-root removal failed: {type(error).__name__}: {error}')
    else:
        cleanup['ok'] = False
        errors.append('Lightpanda process cleanup or runtime-root removal is unverified')

    try:
        request_proof = image_request_proof(fixture, request.nonce)
    except Exception as error:
        request_proof = {
            'nonce': request.nonce, 'image_requests': [], 'count': 0,
            'successful_count': 0, 'ok': False,
        }
        errors.append(f'screenshot fixture proof failed: {type(error).__name__}: {error}')
    if not request_proof['ok']:
        errors.append('screenshot fixture image request was not exactly one successful request')
    png = None
    pixel = {'parseable': False, 'probe_count': 0, 'mismatch_count': 0,
             'mismatch_signature': None}
    try:
        png = verify_png_file(attempt_dir / 'result.png', index, normalize_mode=True)
        pixel = png['pixel']
        if not pixel['parseable']:
            errors.append(pixel.get('error', 'Lightpanda output PNG could not be parsed'))
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')

    if cleanup['ok'] is not True:
        state = 'incomplete'
    elif errors:
        state = 'failed'
    else:
        state = 'pixel_mismatch' if pixel.get('mismatch_signature') else 'passed'
    return {
        'state': state,
        'cleanup_ok': cleanup['ok'],
        'runtime_root_removed': runtime_root_removed,
        'worker_identity': identity,
        'worker_cleanup': cleanup,
        'request_proof': request_proof,
        'png': png,
        'pixel': pixel,
        'parseable_png': pixel.get('parseable') is True,
        'image_request_ok': request_proof['ok'],
        'mismatch_signature': pixel.get('mismatch_signature'),
        'errors': errors,
    }


def parser():
    root = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    commands = root.add_subparsers(dest='command', required=True)
    for name in ('smoke', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--candidate-identity', required=True)
        command.add_argument('--preflight', required=True)
        command.add_argument('--version-manifest', required=True)
        command.add_argument('--source', required=True)
        command.add_argument('--python', required=True)
        command.add_argument('--chrome', required=True)
        command.add_argument('--lightpanda', required=name == 'run')
        command.add_argument('--frozen-harness', required=name == 'run')
        command.add_argument('--output', required=True)
        command.add_argument('--attempts-per-arm', type=int)
    for name in ('verify-smoke', 'verify'):
        command = commands.add_parser(name)
        command.add_argument('--run-root', required=True)
        command.add_argument('--candidate-identity', required=True)
        command.add_argument('--preflight', required=True)
        command.add_argument('--frozen-harness')
    return root


def dispatch(argv=None):
    try:
        args = parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        if args.command in ('smoke', 'run'):
            result_root, output = result_root_for_output(args.output, args.command)
            if args.command == 'run' and args.attempts_per_arm not in (None, 6):
                raise ValueError('full screenshot runs require six attempts per arm')
            if args.command == 'smoke' and args.attempts_per_arm not in (None, 1):
                raise ValueError('screenshot smoke uses one attempt per control arm')
            args.attempts_per_arm = 6 if args.command == 'run' else 1
            return run_screenshot(args, result_root, output)
        result_root, run_root = result_root_for_run(args.run_root, args.command)
        return verify_run_directory(run_root, result_root, args)
    except (ValueError, OSError, KeyError, TypeError) as error:
        LOG.error('%s', error)
        return 2


def create_attempt_directory(run_root, arm, attempt_number):
    arms_root = Path(run_root) / 'arms'
    if not arms_root.exists():
        create_private_child(run_root, 'arms')
    else:
        navigation.validate_private_directory(arms_root)
    arm_root = arms_root / arm
    if not arm_root.exists():
        create_private_child(arms_root, arm)
    else:
        navigation.validate_private_directory(arm_root)
    return create_private_child(arm_root, f'attempt-{attempt_number}')


def update_work_counts(manifest):
    manifest['work_counts'] = screenshot_work_counts(manifest['attempts'])


def save_run_manifest(run_root, manifest):
    update_work_counts(manifest)
    atomic_json(Path(run_root) / 'run-manifest.json', manifest)


def write_screenshot_summary(run_root, manifest):
    summary = {
        'schema': 'openbrowser-screenshot-capability-summary/1',
        'run_id': manifest['run_id'],
        'mode': manifest['mode'],
        'state': manifest['state'],
        'arm_status': manifest['arm_status'],
        'work_counts': manifest['work_counts'],
        'verification_errors': manifest['errors'],
    }
    atomic_json(Path(run_root) / 'summary.json', summary)


async def run_screenshot(args, result_root, output):
    inputs = validate_screenshot_inputs(args, result_root)
    if output.parent != result_root:
        if output.parent.exists():
            navigation.validate_private_directory(output.parent)
        else:
            create_private_child(result_root, output.parent.name)
    output.mkdir(mode=0o700, exist_ok=False)
    output.chmod(0o700)
    navigation.validate_private_directory(output)

    mode = args.command
    arms = arms_for_mode(mode)
    attempts_per_arm = 1 if mode == 'smoke' else args.attempts_per_arm
    run_id = uuid.uuid4().hex
    lightpanda_script = Path(__file__).with_name('screenshot_lightpanda.js')
    manifest = {
        'schema': SMOKE_SCHEMA if mode == 'smoke' else RUN_SCHEMA,
        'mode': mode,
        'run_id': run_id,
        'state': ScreenshotRunState.PREPARING.value,
        'candidate_identity_sha256': inputs['candidate_identity_sha256'],
        'version_manifest_sha256': inputs['version_manifest_sha256'],
        'preflight_sha256': inputs['preflight_sha256'],
        'frozen_harness_sha256': inputs['frozen_harness_sha256'],
        'tool_manifest_sha256': inputs.get('tool_manifest_sha256'),
        'harness_sha256': navigation.sha256(__file__),
        'lightpanda_script_sha256': navigation.sha256(lightpanda_script)
        if lightpanda_script.is_file() else None,
        'arms': list(arms),
        'attempts': [],
        'arm_status': {},
        'work_counts': screenshot_work_counts([]),
        'fixture': {
            'started': False, 'stopped': False, 'socket_closed': False,
            'thread_stopped': False,
        },
        'cleanup': {'ok': False, 'attempts_ok': False, 'runtime_roots_removed': True},
        'errors': [],
    }
    save_run_manifest(output, manifest)

    fixture = None
    fixture_thread = None
    stop_attempts = False
    expected_total = len(arms) * attempts_per_arm
    try:
        fixture = ScreenshotFixture(run_id)
        fixture_thread = threading.Thread(target=fixture.serve_forever, daemon=True)
        fixture_thread.start()
        manifest['fixture']['started'] = True
        manifest['state'] = transition_run(manifest['state'], 'running').value
        save_run_manifest(output, manifest)

        for arm in arms:
            for index in range(attempts_per_arm):
                attempt_number = index + 1
                attempt_dir = create_attempt_directory(output, arm, attempt_number)
                nonce = f'{run_id}-{arm}-{index}'
                attempt = {
                    'attempt_id': f'{run_id}:{arm}:{index}',
                    'arm': arm,
                    'index': index,
                    'nonce': nonce,
                    'request': screenshot_request(run_id, arm, index, fixture.server_address[1]),
                    'directory': attempt_dir.relative_to(output).as_posix(),
                    'state': ScreenshotAttemptState.PENDING.value,
                    'parseable_png': False,
                    'image_request_ok': False,
                    'cleanup_ok': False,
                    'errors': [],
                }
                manifest['attempts'].append(attempt)
                save_run_manifest(output, manifest)
                attempt['state'] = transition_attempt(attempt['state'], 'running').value
                save_run_manifest(output, manifest)
                try:
                    evidence = await execute_attempt(args, fixture, attempt_dir, attempt)
                    if not isinstance(evidence, dict):
                        raise ValueError('screenshot attempt returned no evidence object')
                    terminal_state = evidence.get('state')
                    attempt.update(evidence)
                    if terminal_state not in ('passed', 'pixel_mismatch', 'failed', 'incomplete'):
                        raise ValueError(f'screenshot attempt returned an invalid state: {terminal_state!r}')
                    if attempt.get('cleanup_ok') is not True:
                        terminal_state = 'incomplete'
                    attempt['state'] = transition_attempt(
                        ScreenshotAttemptState.RUNNING, terminal_state,
                    ).value
                except Exception as error:
                    attempt['errors'].append(f'{type(error).__name__}: {error}')
                    attempt['cleanup_ok'] = False
                    attempt['state'] = transition_attempt(
                        ScreenshotAttemptState.RUNNING, 'incomplete',
                    ).value
                save_run_manifest(output, manifest)
                if attempt['state'] == ScreenshotAttemptState.INCOMPLETE.value:
                    stop_attempts = True
                    break
            if stop_attempts:
                break
    except Exception as error:
        manifest['errors'].append(f'{type(error).__name__}: {error}')
        stop_attempts = True
    finally:
        if manifest['state'] == ScreenshotRunState.RUNNING.value:
            manifest['state'] = transition_run(manifest['state'], 'cleaning').value
            save_run_manifest(output, manifest)
        fixture_stopped = False
        if fixture is not None:
            try:
                fixture.shutdown()
            except Exception as error:
                manifest['errors'].append(f'fixture shutdown failed: {type(error).__name__}: {error}')
            try:
                fixture.server_close()
                manifest['fixture']['socket_closed'] = True
            except Exception as error:
                manifest['errors'].append(f'fixture socket close failed: {type(error).__name__}: {error}')
            if fixture_thread is not None:
                fixture_thread.join(timeout=5)
                manifest['fixture']['thread_stopped'] = not fixture_thread.is_alive()
            fixture_stopped = manifest['fixture']['socket_closed'] and manifest['fixture']['thread_stopped']
        manifest['fixture']['stopped'] = fixture_stopped

        if mode == 'run':
            controls = {}
            for arm in ('openbrowser', 'raw-cdp'):
                attempts = [item for item in manifest['attempts'] if item['arm'] == arm]
                controls[arm] = classify_arm_status(
                    arm, attempts, controls, fixture_stopped=fixture_stopped,
                    expected_attempts=attempts_per_arm,
                )
            attempts = [item for item in manifest['attempts'] if item['arm'] == 'lightpanda']
            controls['lightpanda'] = classify_arm_status(
                'lightpanda', attempts, controls, fixture_stopped=fixture_stopped,
                expected_attempts=attempts_per_arm,
            )
            manifest['arm_status'] = controls

        attempts_ok = len(manifest['attempts']) == expected_total and all(
            item.get('cleanup_ok') is True and item.get('runtime_root_removed') is True
            for item in manifest['attempts']
        )
        manifest['cleanup'] = {
            'ok': fixture_stopped and attempts_ok,
            'attempts_ok': attempts_ok,
            'runtime_roots_removed': all(
                item.get('runtime_root_removed') is True for item in manifest['attempts']
            ),
        }
        if manifest['state'] == ScreenshotRunState.PREPARING.value:
            final_state = 'incomplete'
        else:
            complete = len(manifest['attempts']) == expected_total \
                and all(item.get('state') in ('passed', 'pixel_mismatch', 'failed')
                        for item in manifest['attempts']) \
                and manifest['cleanup']['ok'] and not manifest['errors']
            final_state = 'complete' if complete else 'incomplete'
        manifest['state'] = transition_run(manifest['state'], final_state).value
        save_run_manifest(output, manifest)
        write_screenshot_summary(output, manifest)
    return screenshot_run_exit_code(manifest)
def contains_latency_fields(value):
    forbidden = {'elapsed_ns', 'latency_ns', 'navigation_ns', 'duration_ns', 'start_ns', 'end_ns'}
    if isinstance(value, dict):
        return any(key in forbidden or contains_latency_fields(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_latency_fields(item) for item in value)
    return False


def verify_attempt_artifacts(run_root, attempt):
    errors = []
    if not isinstance(attempt, dict):
        return ['screenshot attempt record is not an object']
    arm = attempt.get('arm')
    index = attempt.get('index')
    if arm not in SCREENSHOT_ARMS or type(index) is not int or not 0 <= index < 6:
        return ['attempt arm or index is invalid']
    expected_directory = Path('arms') / arm / f'attempt-{index + 1}'
    directory_value = attempt.get('directory')
    if not isinstance(directory_value, str):
        return ['attempt directory is not a string']
    relative = Path(directory_value)
    if relative.is_absolute() or '..' in relative.parts or relative != expected_directory:
        return ['attempt directory does not match its arm and index']
    attempt_dir = Path(run_root) / relative
    try:
        navigation.validate_private_directory(attempt_dir)
    except (OSError, ValueError) as error:
        return [f'attempt directory is invalid: {error}']

    attempt_id = attempt.get('attempt_id')
    run_id = attempt_id.split(':', 1)[0] if isinstance(attempt_id, str) else ''
    expected_attempt_id = f'{run_id}:{arm}:{index}'
    expected_nonce = f'{run_id}-{arm}-{index}'
    if attempt.get('attempt_id') != expected_attempt_id or attempt.get('nonce') != expected_nonce:
        errors.append('attempt identity or nonce does not match the manifest run')
    try:
        request = (
            parse_lightpanda_screenshot_request(attempt.get('request'))
            if arm == 'lightpanda'
            else navigation.parse_screenshot_request(arm, attempt.get('request'))
        )
        if request.attempt_id != expected_attempt_id or request.nonce != expected_nonce:
            errors.append('screenshot request does not match its attempt identity')
    except (TypeError, ValueError) as error:
        request = None
        errors.append(f'screenshot request is invalid: {error}')

    proof = attempt.get('request_proof')
    if not isinstance(proof, dict):
        errors.append('screenshot request proof is missing')
        proof_ok = False
    else:
        image_requests = proof.get('image_requests')
        valid_requests = isinstance(image_requests, list) and len(image_requests) == 1 \
            and isinstance(image_requests[0], dict) \
            and image_requests[0].get('method') == 'GET' \
            and image_requests[0].get('path') == '/pixels.png' \
            and image_requests[0].get('nonce') == expected_nonce \
            and image_requests[0].get('status') == 200
        proof_ok = proof.get('ok') is True and proof.get('nonce') == expected_nonce \
            and type(proof.get('count')) is int and proof.get('count') == 1 \
            and type(proof.get('successful_count')) is int \
            and proof.get('successful_count') == 1 and valid_requests
        if attempt.get('image_request_ok') is not proof_ok:
            errors.append('screenshot image-request proof disagrees with its status')
    if attempt.get('state') in ('passed', 'pixel_mismatch') and not proof_ok:
        errors.append('completed screenshot attempt has no successful image request proof')

    png_record = attempt.get('png')
    pixel = attempt.get('pixel')
    if png_record is not None:
        output = attempt_dir / 'result.png'
        try:
            info = output.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 \
                    or info.st_uid != os.geteuid():
                raise ValueError('result.png is not a private regular file with mode 0600')
            content = output.read_bytes()
            digest = navigation.sha256(output)
            if png_record != {'file': 'result.png', 'byte_count': len(content), 'sha256': digest,
                              'pixel': pixel}:
                errors.append('saved PNG bytes or pixel evidence do not match the manifest')
            actual_pixel = inspect_png(content, index)
            if actual_pixel != pixel:
                errors.append('saved PNG pixels do not match the manifest evidence')
            if attempt.get('parseable_png') is not actual_pixel.get('parseable'):
                errors.append('saved PNG parse status does not match the manifest')
            if attempt.get('mismatch_signature') != actual_pixel.get('mismatch_signature'):
                errors.append('saved PNG mismatch signature does not match the manifest')
        except (OSError, ValueError) as error:
            errors.append(f'saved PNG verification failed: {error}')
    elif attempt.get('state') in ('passed', 'pixel_mismatch'):
        errors.append('completed screenshot attempt has no PNG record')

    if arm in ('openbrowser', 'raw-cdp') and attempt.get('worker_reply') is not None and request is not None:
        try:
            reply = navigation.parse_screenshot_reply(
                request, {'ok': True, 'result': attempt['worker_reply']},
            )
            if png_record and (reply.sha256 != png_record['sha256']
                               or reply.byte_count != png_record['byte_count']):
                errors.append('worker reply does not match the saved PNG')
        except (TypeError, ValueError) as error:
            errors.append(f'worker screenshot reply is invalid: {error}')
    elif arm in ('openbrowser', 'raw-cdp') and attempt.get('state') in ('passed', 'pixel_mismatch'):
        errors.append('completed Chrome screenshot attempt has no worker reply')

    if attempt.get('state') in ('passed', 'pixel_mismatch', 'failed'):
        cleanup = attempt.get('worker_cleanup')
        if attempt.get('cleanup_ok') is not True or attempt.get('runtime_root_removed') is not True \
                or not isinstance(cleanup, dict) or cleanup.get('ok') is not True:
            errors.append('terminal screenshot attempt has incomplete cleanup proof')
        else:
            top_identity = attempt.get('worker_identity')
            cleanup_identity = cleanup.get('worker_identity')
            if not isinstance(top_identity, dict) or navigation.profile_identity_errors(
                top_identity, state_required=False,
            ):
                errors.append('screenshot worker identity is missing or invalid')
            if not isinstance(cleanup_identity, dict) or navigation.profile_identity_errors(
                cleanup_identity, state_required=True,
            ):
                errors.append('screenshot cleanup worker identity is missing or invalid')
            elif cleanup_identity.get('state') not in ('absent', 'reused'):
                errors.append('screenshot worker is not proven retired')
            elif isinstance(top_identity, dict) and any(
                top_identity.get(key) != cleanup_identity.get(key)
                for key in ('pid', 'created', 'cmdline', 'owner_token', 'owner_root')
            ):
                errors.append('screenshot cleanup worker identity differs from the launched worker')

            runtime_root_value = attempt.get('runtime_root')
            runtime_root = Path(runtime_root_value) if isinstance(runtime_root_value, str) else None
            if runtime_root is None or not runtime_root.is_absolute() or '..' in runtime_root.parts:
                errors.append('screenshot runtime root path is invalid')
            else:
                physical_tmp = Path('/tmp').resolve(strict=True)
                result_root = Path(run_root).parent.parent \
                    if Path(run_root).parent.name == 'smoke' and Path(run_root).name == 'screenshot' \
                    else Path(run_root).parent
                if runtime_root.parent != physical_tmp or not runtime_root.name.startswith('nb-screenshot-') \
                        or runtime_root == result_root or runtime_root.is_relative_to(result_root):
                    errors.append('screenshot runtime root is not separate from RESULT_ROOT')
                expected_owner_root = str(runtime_root / arm)
                if isinstance(top_identity, dict) and top_identity.get('owner_root') != expected_owner_root:
                    errors.append('screenshot worker identity does not belong to its runtime root')

            child_identities = cleanup.get('post_worker_exit_identities')
            if not isinstance(child_identities, list):
                errors.append('screenshot worker child cleanup is not a list')
            else:
                owner_token = top_identity.get('owner_token') if isinstance(top_identity, dict) else None
                for child in child_identities:
                    if not isinstance(child, dict) or navigation.profile_identity_errors(
                        child, state_required=True,
                    ):
                        errors.append('screenshot worker child identity is missing or invalid')
                        continue
                    if child.get('state') not in ('absent', 'reused'):
                        errors.append('screenshot worker child cleanup is not proven')
                    if child.get('owner_token') != owner_token \
                            or (runtime_root is not None and child.get('owner_root') != str(runtime_root / arm)):
                        errors.append('screenshot worker child ownership differs from the launched worker')
            if isinstance(cleanup.get('errors'), list) and cleanup['errors']:
                errors.append('screenshot worker cleanup contains errors')
            if arm == 'lightpanda' and cleanup.get('process_state') != 'verified':
                errors.append('Lightpanda cleanup lifecycle is not verified')
            if arm == 'lightpanda' and isinstance(top_identity, dict) \
                    and cleanup.get('process_group_id') != top_identity.get('pid'):
                errors.append('Lightpanda process group is not bound to the launched process')
    if attempt.get('state') == 'passed' and (
        not isinstance(pixel, dict) or pixel.get('mismatch_signature') is not None
        or pixel.get('dimension_ok') is not True or pixel.get('probe_count') != 4
    ):
        errors.append('passed screenshot attempt does not satisfy the pixel contract')
    if attempt.get('state') == 'pixel_mismatch' and (
        not isinstance(pixel, dict) or pixel.get('parseable') is not True
        or pixel.get('mismatch_signature') is None
    ):
        errors.append('pixel_mismatch attempt has no verified pixel-contract failure')
    if attempt.get('state') == 'failed' and not attempt.get('errors'):
        errors.append('failed screenshot attempt has no execution error')
    if attempt.get('state') not in ('passed', 'pixel_mismatch', 'failed', 'incomplete'):
        errors.append('screenshot attempt has a non-terminal state')
    if contains_latency_fields(attempt):
        errors.append('screenshot attempt contains a latency field')

    logs = attempt_dir / 'logs'
    if logs.exists() or logs.is_symlink():
        try:
            navigation.validate_private_directory(logs)
            for path in logs.iterdir():
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 \
                        or path.name not in {
                            'worker.stdout.log', 'worker.stderr.log',
                            'lightpanda.stdout.log', 'lightpanda.stderr.log',
                        }:
                    errors.append(f'unexpected or non-private screenshot log: {path.name}')
        except (OSError, ValueError) as error:
            errors.append(f'screenshot log directory is invalid: {error}')
    return errors


def verify_screenshot_manifest(run_root, manifest, mode):
    errors = []
    expected_schema = SMOKE_SCHEMA if mode == 'smoke' else RUN_SCHEMA
    expected_root_keys = {
        'schema', 'mode', 'run_id', 'state', 'candidate_identity_sha256',
        'version_manifest_sha256', 'preflight_sha256', 'frozen_harness_sha256',
        'tool_manifest_sha256', 'harness_sha256', 'lightpanda_script_sha256',
        'arms', 'attempts', 'arm_status', 'work_counts', 'fixture', 'cleanup', 'errors',
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_root_keys \
            or manifest.get('schema') != expected_schema or manifest.get('mode') != mode:
        return ['screenshot manifest schema or mode is invalid']
    expected_arms = arms_for_mode(mode)
    if manifest.get('arms') != list(expected_arms):
        errors.append('screenshot manifest arm list is invalid')
    fixture = manifest.get('fixture')
    if not isinstance(fixture, dict) \
            or set(fixture) != {'started', 'stopped', 'socket_closed', 'thread_stopped'} \
            or any(type(value) is not bool for value in fixture.values()):
        errors.append('screenshot fixture lifecycle record is invalid')
    elif manifest.get('state') == 'complete' and any(value is not True for value in fixture.values()):
        errors.append('complete screenshot manifest has incomplete fixture cleanup')
    cleanup = manifest.get('cleanup')
    if not isinstance(cleanup, dict) \
            or set(cleanup) != {'ok', 'attempts_ok', 'runtime_roots_removed'} \
            or any(type(value) is not bool for value in cleanup.values()):
        errors.append('screenshot cleanup record is invalid')
    elif manifest.get('state') == 'complete' and any(value is not True for value in cleanup.values()):
        errors.append('complete screenshot manifest has incomplete cleanup')
    if not isinstance(manifest.get('errors'), list) \
            or any(not isinstance(error, str) for error in manifest['errors']):
        errors.append('screenshot run errors must be a list of strings')
    elif manifest.get('state') == 'complete' and manifest['errors']:
        errors.append('complete screenshot manifest contains run errors')
    run_id = manifest.get('run_id')
    if not isinstance(run_id, str) or len(run_id) != 32 \
            or any(character not in '0123456789abcdef' for character in run_id):
        errors.append('screenshot manifest run_id is invalid')
    attempts = manifest.get('attempts')
    if not isinstance(attempts, list):
        return [*errors, 'screenshot manifest attempts are not a list']
    if any(not isinstance(attempt, dict) for attempt in attempts):
        errors.append('screenshot manifest contains a non-object attempt record')
        return errors
    expected_per_arm = 1 if mode == 'smoke' else 6
    expected_keys = [(arm, index) for arm in expected_arms for index in range(expected_per_arm)]
    actual_keys = [(attempt.get('arm'), attempt.get('index')) for attempt in attempts if isinstance(attempt, dict)]
    if manifest.get('state') == 'complete' and actual_keys != expected_keys:
        errors.append('complete screenshot manifest does not contain every ordered attempt')
    if len(set(actual_keys)) != len(actual_keys):
        errors.append('screenshot manifest contains duplicate attempts')
    for attempt in attempts:
        errors.extend(verify_attempt_artifacts(run_root, attempt))
    if mode == 'run':
        controls = {}
        fixture_stopped = isinstance(fixture, dict) and fixture.get('stopped') is True
        for arm in ('openbrowser', 'raw-cdp'):
            arm_attempts = [attempt for attempt in attempts if attempt.get('arm') == arm]
            controls[arm] = classify_arm_status(
                arm, arm_attempts, controls,
                fixture_stopped=fixture_stopped,
                expected_attempts=expected_per_arm,
            )
        controls['lightpanda'] = classify_arm_status(
            'lightpanda', [attempt for attempt in attempts if attempt.get('arm') == 'lightpanda'],
            controls, fixture_stopped=fixture_stopped,
            expected_attempts=expected_per_arm,
        )
        if manifest.get('arm_status') != controls:
            errors.append('screenshot arm status does not match the six attempt records')
    elif manifest.get('arm_status') != {}:
        errors.append('screenshot smoke must not classify arm capability')
    expected_counts = screenshot_work_counts(attempts)
    if manifest.get('work_counts') != expected_counts:
        errors.append('screenshot work counts do not match attempt records')
    if contains_latency_fields(manifest):
        errors.append('screenshot manifest contains a latency field')
    if screenshot_run_exit_code(manifest) != 0:
        errors.append('screenshot manifest does not describe a complete verified run')

    try:
        summary_path = Path(run_root) / 'summary.json'
        info = summary_path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 \
                or info.st_uid != os.geteuid():
            raise ValueError('screenshot summary must be a private regular file with mode 0600')
        summary = json.loads(summary_path.read_text())
        expected_summary = {
            'schema': 'openbrowser-screenshot-capability-summary/1',
            'run_id': manifest.get('run_id'), 'mode': mode,
            'state': manifest.get('state'), 'arm_status': manifest.get('arm_status'),
            'work_counts': expected_counts, 'verification_errors': manifest.get('errors'),
        }
        if summary != expected_summary or contains_latency_fields(summary):
            errors.append('screenshot summary does not match the verified manifest')
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f'screenshot summary is invalid: {error}')
    return errors


def verify_run_directory(run_root, result_root, args):
    mode = 'smoke' if args.command == 'verify-smoke' else 'run'
    try:
        root = benchmark_preflight._private_result_root(result_root)
        run_root = benchmark_preflight._private_artifact_directory(run_root, root, 'screenshot run')
        manifest_path = benchmark_preflight._artifact_file(
            run_root / 'run-manifest.json', root, 'screenshot run manifest',
        )
        info = manifest_path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 \
                or info.st_uid != os.geteuid():
            raise ValueError('screenshot run manifest must be a private regular file with mode 0600')
        manifest = json.loads(manifest_path.read_text())
        identity_path, identity = read_json_artifact(
            result_root, args.candidate_identity, 'candidate identity',
        )
        preflight_path, preflight = read_json_artifact(
            result_root, args.preflight, 'screenshot preflight',
        )
        identity = benchmark_preflight._validate_identity(identity)
        preflight = benchmark_preflight._validate_preflight_result(preflight)
        if any(path.parent != root for path in (identity_path, preflight_path)):
            raise ValueError('candidate and preflight manifests must be direct children of RESULT_ROOT')
        identity_hash = navigation.sha256(identity_path)
        if identity.get('schema') != 'openbrowser-candidate-identity/1' \
                or manifest.get('candidate_identity_sha256') != identity_hash:
            raise ValueError('screenshot run candidate identity does not match')
        version_hash = manifest.get('version_manifest_sha256')
        if mode == 'run':
            version_path, version = read_json_artifact(
                result_root, root / 'version-manifest.json', 'version manifest',
            )
            version = benchmark_preflight._validate_version_manifest(version)
            if version_path.parent != root:
                raise ValueError('version manifest must be a direct child of RESULT_ROOT')
            version_hash = navigation.sha256(version_path)
            if version.get('candidate_identity_sha256') != identity_hash:
                raise ValueError('version manifest does not match the candidate identity')
        expected_phase = 'screenshot-smoke' if mode == 'smoke' \
            else 'screenshot-final' if Path(run_root).name == 'screenshot-final' \
            else 'screenshot-run'
        if preflight.get('schema') != 'openbrowser-latest-preflight/1' \
                or preflight.get('phase') != expected_phase or preflight.get('outcome') != 'passed' \
                or preflight.get('errors') != []:
            raise ValueError(f'{expected_phase} preflight did not pass')
        if manifest.get('preflight_sha256') != navigation.sha256(preflight_path) \
                or preflight.get('candidate_identity_sha256') != identity_hash \
                or preflight.get('version_manifest_sha256') != version_hash \
                or manifest.get('version_manifest_sha256') != version_hash:
            raise ValueError('screenshot run preflight does not match the candidate identity')
        if manifest.get('harness_sha256') != navigation.sha256(__file__):
            raise ValueError('screenshot runner changed after this run')
        lightpanda_script = Path(__file__).with_name('screenshot_lightpanda.js')
        if not lightpanda_script.is_file() \
                or manifest.get('lightpanda_script_sha256') != navigation.sha256(lightpanda_script):
            raise ValueError('Lightpanda screenshot script changed after this run')
        if mode == 'run':
            frozen_path, frozen = read_json_artifact(
                result_root, args.frozen_harness, 'frozen harness',
            )
            tool_manifest_path = benchmark_preflight._artifact_file(
                root / 'tool-manifest.json', root, 'tool manifest',
            )
            candidate_freeze_path = benchmark_preflight._artifact_file(
                root / 'candidate-freeze.txt', root, 'candidate dependency freeze',
            )
            browser_use_freeze_path = benchmark_preflight._artifact_file(
                root / 'browser-use-freeze.txt', root, 'browser-use dependency freeze',
            )
            tool_sha = benchmark_preflight.sha256_file(tool_manifest_path)
            frozen = benchmark_preflight._validate_frozen_harness(
                frozen,
                identity_sha=identity_hash,
                candidate_freeze_sha=benchmark_preflight.sha256_file(candidate_freeze_path),
                browser_use_freeze_sha=benchmark_preflight.sha256_file(browser_use_freeze_path),
                tool_manifest_sha=tool_sha,
            )
            if frozen_path != root / 'frozen-harness.json' \
                    or manifest.get('frozen_harness_sha256') != navigation.sha256(frozen_path) \
                    or frozen.get('version_manifest_sha256') != version_hash \
                    or frozen.get('tool_manifest_sha256') != tool_sha \
                    or frozen.get('executed_file_sha256') != identity.get('executed_files') \
                    or manifest.get('tool_manifest_sha256') != tool_sha:
                raise ValueError('screenshot run frozen harness does not match the candidate')
            benchmark_preflight.validate_frozen_smoke_preflight_hashes(root, frozen)
            benchmark_preflight.validate_frozen_smoke_preflight_hashes(root, frozen)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        LOG.error('%s', error)
        return 1
    errors = verify_screenshot_manifest(run_root, manifest, mode)
    for error in errors:
        LOG.error('%s', error)
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(dispatch())
