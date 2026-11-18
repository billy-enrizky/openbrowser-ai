from __future__ import annotations

CLI_DESCRIPTION = (
    'Local current-release operation diagnostic. No LLM, cloud, Docker, or user profiles.\n\n'
    'Each owned worker has one arm, a private profile, and an explicit command protocol.\n'
    'The measured workload is navigation through load, observation, then one DOM mutation\n'
    'and one POST. CLI transport is included only for the CLI-only agent-browser arm.'
)

import argparse
import asyncio
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import html
import importlib.metadata
import ipaddress
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal, TextIO
from urllib.parse import parse_qs, urlencode, urlsplit

import psutil

LOG = logging.getLogger('comparison')
PREFIX = 'BENCH_RESULT:'
SCRIPT = Path(__file__).resolve()
ARMS = ('raw-cdp', 'lightpanda', 'agent-browser', 'browser-use', 'openbrowser')
OBSERVE = """JSON.stringify({url:location.href,title:document.title,
readyState:document.readyState,marker:document.getElementById('marker')?.textContent,
loaded:document.documentElement.dataset.loaded,scriptDone:document.documentElement.dataset.scriptDone,
field:document.getElementById('value')?.value})"""
SCREENSHOT_OBSERVE = """JSON.stringify({readyState:document.readyState,
readyMarker:document.documentElement.dataset.screenshotReady||null,
imageReady:(()=>{const proof=document.querySelector('#proof');
const image=proof?.querySelector('img');const box=proof?.getBoundingClientRect();
return !!proof&&box?.x===0&&box?.y===0&&box?.width===128&&box?.height===128
&&!!image&&image.complete&&image.naturalWidth===64&&image.naturalHeight===64})()})"""


class WorkerMode(StrEnum):
    NAVIGATION = 'navigation'
    SCREENSHOT = 'screenshot'
    PROFILE = 'profile'


_MODE_OPERATION = {
    WorkerMode.NAVIGATION: 'trial',
    WorkerMode.SCREENSHOT: 'capture_screenshot',
    WorkerMode.PROFILE: 'profile_pair',
}


class WorkerState(StrEnum):
    CREATED = 'created'
    STARTING = 'starting'
    READY = 'ready'
    OPERATION_RUNNING = 'operation_running'
    OPERATION_COMPLETE = 'operation_complete'
    FAILED = 'failed'
    STOPPING = 'stopping'
    STOPPED = 'stopped'
    INCOMPLETE = 'incomplete'


class WorkerEvent(StrEnum):
    START = 'start'
    START_SUCCEEDED = 'start_succeeded'
    START_FAILED = 'start_failed'
    VALID_OPERATION = 'valid_operation'
    INVALID_OPERATION = 'invalid_operation'
    NAVIGATION_RESULT = 'navigation_result'
    ONE_SHOT_RESULT = 'one_shot_result'
    OPERATION_FAILED = 'operation_failed'
    CLEANUP = 'cleanup'
    CLEANUP_WITHOUT_CHILD = 'cleanup_without_child'
    CLEANUP_VERIFIED = 'cleanup_verified'
    CLEANUP_UNCERTAIN = 'cleanup_uncertain'
    LATE_OPERATION = 'late_operation'


_WORKER_TRANSITIONS = {
    (WorkerState.CREATED, WorkerEvent.START): WorkerState.STARTING,
    (WorkerState.CREATED, WorkerEvent.CLEANUP_WITHOUT_CHILD): WorkerState.STOPPED,
    (WorkerState.STARTING, WorkerEvent.START_SUCCEEDED): WorkerState.READY,
    (WorkerState.STARTING, WorkerEvent.START_FAILED): WorkerState.FAILED,
    (WorkerState.STARTING, WorkerEvent.INVALID_OPERATION): WorkerState.FAILED,
    (WorkerState.READY, WorkerEvent.VALID_OPERATION): WorkerState.OPERATION_RUNNING,
    (WorkerState.READY, WorkerEvent.INVALID_OPERATION): WorkerState.FAILED,
    (WorkerState.OPERATION_RUNNING, WorkerEvent.NAVIGATION_RESULT): WorkerState.READY,
    (WorkerState.OPERATION_RUNNING, WorkerEvent.ONE_SHOT_RESULT): WorkerState.OPERATION_COMPLETE,
    (WorkerState.OPERATION_RUNNING, WorkerEvent.OPERATION_FAILED): WorkerState.FAILED,
    (WorkerState.OPERATION_COMPLETE, WorkerEvent.INVALID_OPERATION): WorkerState.OPERATION_COMPLETE,
    (WorkerState.FAILED, WorkerEvent.INVALID_OPERATION): WorkerState.FAILED,
    (WorkerState.STOPPING, WorkerEvent.CLEANUP): WorkerState.STOPPING,
    (WorkerState.STOPPING, WorkerEvent.CLEANUP_VERIFIED): WorkerState.STOPPED,
    (WorkerState.STOPPING, WorkerEvent.CLEANUP_UNCERTAIN): WorkerState.INCOMPLETE,
    (WorkerState.STOPPED, WorkerEvent.CLEANUP): WorkerState.STOPPED,
    (WorkerState.STOPPED, WorkerEvent.LATE_OPERATION): WorkerState.STOPPED,
}
for _live_state in (
    WorkerState.STARTING,
    WorkerState.READY,
    WorkerState.OPERATION_RUNNING,
    WorkerState.OPERATION_COMPLETE,
    WorkerState.FAILED,
):
    _WORKER_TRANSITIONS[(_live_state, WorkerEvent.CLEANUP)] = WorkerState.STOPPING


def transition_worker_state(state: WorkerState | str, event: WorkerEvent | str) -> WorkerState:
    try:
        current = WorkerState(state)
        action = WorkerEvent(event)
    except ValueError as error:
        raise ValueError(f'unknown worker state or event: {state!r}, {event!r}') from error
    try:
        return _WORKER_TRANSITIONS[(current, action)]
    except KeyError as error:
        raise ValueError(f'illegal worker transition: {current.value} + {action.value}') from error


@dataclass(frozen=True)
class WorkerLaunch:
    arm: Literal['raw-cdp', 'lightpanda', 'agent-browser', 'browser-use', 'openbrowser']
    mode: WorkerMode
    source_root: Path
    interpreter: Path
    runtime_root: Path
    worker_root: Path
    working_directory: Path
    log_directory: Path
    log_prefix: str
    owner_token: str
    chrome: Path | None = None
    lightpanda: Path | None = None
    agent_browser: Path | None = None
    node: Path | None = None


@dataclass
class WorkerHandle:
    process: asyncio.subprocess.Process
    stdout_log: TextIO
    stderr_log: TextIO
    worker_identity: dict
    initial_owned: tuple[dict, ...] | None
    owner_token: str
    mode: WorkerMode
    state: WorkerState
    cleanup_result: dict | None = None


@dataclass(frozen=True)
class ScreenshotRequest:
    attempt_id: str
    arm: Literal['openbrowser', 'raw-cdp']
    url: str
    nonce: str
    selector: Literal['#proof']
    output: Literal['result.png']


@dataclass(frozen=True)
class ScreenshotReply:
    schema: Literal['openbrowser-screenshot-reply/1']
    attempt_id: str
    arm: Literal['openbrowser', 'raw-cdp']
    nonce: str
    ready_marker: str
    output: Literal['result.png']
    byte_count: int
    sha256: str


def parse_screenshot_request(arm, value) -> ScreenshotRequest:
    if arm not in ('openbrowser', 'raw-cdp'):
        raise ValueError('screenshot mode supports only OpenBrowser and raw CDP')
    if not isinstance(value, dict):
        raise ValueError('screenshot request must be an object')
    expected = {'op', 'schema', 'attempt_id', 'url', 'nonce', 'selector', 'output'}
    extra = value.keys() - expected
    missing = expected - value.keys()
    if extra:
        raise ValueError(f'screenshot request has unsupported fields: {sorted(extra)!r}')
    if missing:
        raise ValueError(f'screenshot request is missing fields: {sorted(missing)!r}')
    if value['op'] != 'capture_screenshot' or value['schema'] != 'openbrowser-screenshot-request/1':
        raise ValueError('screenshot request operation or schema is invalid')
    attempt_id, nonce, url = value['attempt_id'], value['nonce'], value['url']
    if not all(isinstance(item, str) for item in (attempt_id, nonce, url)):
        raise ValueError('screenshot attempt, nonce, and URL must be strings')
    parts = attempt_id.split(':')
    if len(parts) != 3 or len(parts[0]) != 32 \
            or any(character not in '0123456789abcdef' for character in parts[0]) \
            or parts[1] != arm or not parts[2] \
            or any(character not in '0123456789' for character in parts[2]):
        raise ValueError('screenshot attempt_id is invalid')
    index = int(parts[2])
    if str(index) != parts[2]:
        raise ValueError('screenshot attempt_id index is not canonical')
    if nonce != f'{parts[0]}-{arm}-{index}':
        raise ValueError('screenshot nonce does not match attempt_id')
    try:
        parsed_url = urlsplit(url)
        hostname = parsed_url.hostname
        port = parsed_url.port
        loopback = hostname is not None and ipaddress.ip_address(hostname).is_loopback
    except ValueError as error:
        raise ValueError('screenshot URL is invalid') from error
    if parsed_url.scheme != 'http' or not loopback or port is None \
            or parsed_url.path != '/screenshot' or parsed_url.username is not None \
            or parsed_url.password is not None or parsed_url.fragment:
        raise ValueError('screenshot URL must use a loopback /screenshot endpoint')
    if parse_qs(parsed_url.query, keep_blank_values=True) != {'nonce': [nonce]}:
        raise ValueError('screenshot URL nonce does not match the request')
    if value['selector'] != '#proof':
        raise ValueError('screenshot selector must be #proof')
    if value['output'] != 'result.png':
        raise ValueError('screenshot output must be result.png')
    return ScreenshotRequest(attempt_id, arm, url, nonce, '#proof', 'result.png')


def parse_screenshot_reply(request: ScreenshotRequest, envelope) -> ScreenshotReply:
    if not isinstance(envelope, dict):
        raise ValueError('screenshot reply envelope must be an object')
    if envelope.keys() != {'ok', 'result'}:
        raise ValueError('screenshot reply envelope has missing or extra fields')
    if envelope['ok'] is not True:
        raise ValueError('screenshot worker returned an error envelope')
    result = envelope['result']
    expected = {'schema', 'attempt_id', 'arm', 'nonce', 'ready_marker', 'output', 'byte_count', 'sha256'}
    if not isinstance(result, dict) or result.keys() != expected:
        raise ValueError('screenshot reply result has missing or extra fields')
    if result['schema'] != 'openbrowser-screenshot-reply/1':
        raise ValueError('screenshot reply schema is invalid')
    for field, expected_value in (
        ('attempt_id', request.attempt_id), ('arm', request.arm), ('nonce', request.nonce),
        ('ready_marker', request.nonce), ('output', request.output),
    ):
        if result[field] != expected_value:
            raise ValueError(f'screenshot reply {field} does not match the request')
    if type(result['byte_count']) is not int or result['byte_count'] <= 0:
        raise ValueError('screenshot reply byte_count must be a positive integer')
    digest = result['sha256']
    if not isinstance(digest, str) or len(digest) != 64 \
            or any(character not in '0123456789abcdef' for character in digest):
        raise ValueError('screenshot reply sha256 must be 64 lowercase hexadecimal characters')
    return ScreenshotReply(**result)


def make_worker_launch(
    *, arm, mode, source_root, candidate_python, browser_use_python,
    runtime_root, worker_root, working_directory, log_directory, log_prefix,
    owner_token, chrome=None, lightpanda=None, agent_browser=None, node=None,
):
    if arm not in ARMS:
        raise ValueError(f'unsupported worker arm: {arm!r}')
    try:
        mode = WorkerMode(mode)
    except ValueError as error:
        raise ValueError(f'unsupported worker mode: {mode!r}') from error
    if mode is WorkerMode.PROFILE and arm != 'openbrowser':
        raise ValueError('profile mode supports only OpenBrowser')
    if mode is WorkerMode.SCREENSHOT and arm not in ('openbrowser', 'raw-cdp'):
        raise ValueError('screenshot mode supports only OpenBrowser and raw CDP')
    if arm == 'browser-use' and browser_use_python is None:
        raise ValueError('browser-use requires BROWSER_USE_PYTHON')
    if node is not None and arm != 'agent-browser':
        raise ValueError('a Node executable is allowed only for the agent-browser worker')

    interpreter = browser_use_python if arm == 'browser-use' else candidate_python
    paths = {
        'source_root': source_root,
        'interpreter': interpreter,
        'runtime_root': runtime_root,
        'worker_root': worker_root,
        'working_directory': working_directory,
        'log_directory': log_directory,
    }
    normalized = {}
    for name, value in paths.items():
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f'worker launch {name} must be absolute')
        normalized[name] = path
    for name, value in (('chrome', chrome), ('lightpanda', lightpanda),
                        ('agent_browser', agent_browser), ('node', node)):
        if value is not None and not Path(value).is_absolute():
            raise ValueError(f'worker launch {name} must be absolute')
    if normalized['worker_root'].parent != normalized['runtime_root'] \
            or normalized['worker_root'].name != arm:
        raise ValueError('worker root must be exactly RUNTIME_ROOT/<arm>')
    if normalized['log_directory'].is_relative_to(normalized['runtime_root']):
        raise ValueError('worker logs must be outside RUNTIME_ROOT')
    if not owner_token:
        raise ValueError('worker owner token must not be empty')
    if not isinstance(log_prefix, str) or not log_prefix or log_prefix in ('.', '..') \
            or Path(log_prefix).name != log_prefix \
            or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for character in log_prefix):
        raise ValueError('worker log prefix must be a safe basename')

    launch = WorkerLaunch(
        arm=arm, mode=mode,
        source_root=normalized['source_root'],
        interpreter=normalized['interpreter'],
        runtime_root=normalized['runtime_root'],
        worker_root=normalized['worker_root'],
        working_directory=normalized['working_directory'],
        log_directory=normalized['log_directory'],
        log_prefix=log_prefix,
        owner_token=owner_token,
        chrome=Path(chrome) if chrome is not None else None,
        lightpanda=Path(lightpanda) if lightpanda is not None else None,
        agent_browser=Path(agent_browser) if agent_browser is not None else None,
        node=Path(node) if node is not None else None,
    )
    validate_worker_launch(launch)
    return launch


def validate_worker_launch(launch: WorkerLaunch) -> None:
    if launch.arm not in ARMS:
        raise ValueError(f'unsupported worker arm: {launch.arm!r}')
    if launch.mode is WorkerMode.PROFILE and launch.arm != 'openbrowser':
        raise ValueError('profile mode supports only OpenBrowser')
    if launch.mode is WorkerMode.SCREENSHOT and launch.arm not in ('openbrowser', 'raw-cdp'):
        raise ValueError('screenshot mode supports only OpenBrowser and raw CDP')
    if launch.node is not None and launch.arm != 'agent-browser':
        raise ValueError('a Node executable is allowed only for the agent-browser worker')
    for name in ('source_root', 'interpreter', 'runtime_root', 'worker_root',
                 'working_directory', 'log_directory'):
        path = Path(getattr(launch, name))
        if not path.is_absolute():
            raise ValueError(f'worker launch {name} must be absolute')
    for name in ('chrome', 'lightpanda', 'agent_browser', 'node'):
        path = getattr(launch, name)
        if path is not None and not Path(path).is_absolute():
            raise ValueError(f'worker launch {name} must be absolute')
    if launch.worker_root.parent != launch.runtime_root or launch.worker_root.name != launch.arm:
        raise ValueError('worker root must be exactly RUNTIME_ROOT/<arm>')
    if launch.log_directory.is_relative_to(launch.runtime_root):
        raise ValueError('worker logs must be outside RUNTIME_ROOT')
    if not launch.owner_token:
        raise ValueError('worker owner token must not be empty')
    if not isinstance(launch.log_prefix, str) or not launch.log_prefix \
            or launch.log_prefix in ('.', '..') or Path(launch.log_prefix).name != launch.log_prefix \
            or any(character not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for character in launch.log_prefix):
        raise ValueError('worker log prefix must be a safe basename')


def record(p, token):
    identity = {'pid': p.pid, 'created': p.create_time(), 'cmdline': p.cmdline(),
                'owner_token': None, 'owner_root': None}
    try:
        env = p.environ()
        runtime = Path(env.get('BH_RUNTIME_DIR', ''))
        root = runtime.parent
        if not token or env.get('NAVIGATION_BENCHMARK_OWNER') != token:
            raise ValueError('process ownership marker does not match')
        if not runtime.is_absolute() or runtime.name != 'runtime' or root.name not in ARMS:
            raise ValueError('process runtime is not a private benchmark runtime')
        validate_worker_root(root, root.name, token)
        if env.get('HOME') != str(root / 'home'):
            raise ValueError('process home does not match private runtime')
        for index, arg in enumerate(identity['cmdline']):
            if arg == '--user-data-dir':
                profile = identity['cmdline'][index + 1] if index + 1 < len(identity['cmdline']) else ''
            elif arg.startswith('--user-data-dir='):
                profile = arg.partition('=')[2]
            else:
                continue
            if not profile or Path(profile).resolve() != (root / 'profile').resolve():
                raise ValueError('process profile does not match private runtime')
        identity.update(owner_token=token, owner_root=str(root))
    except (psutil.Error, OSError, ValueError) as error:
        identity['ownership_error'] = str(error)
    return identity


def same_owned_identity(saved, current, token):
    keys = ('pid', 'created', 'cmdline')
    return (bool(token) and all(saved.get(key) for key in keys)
            and all(saved.get(key) == current.get(key) for key in keys)
            and saved.get('owner_root') == current.get('owner_root')
            and saved.get('owner_token') == current.get('owner_token') == token)


def removal_allowed(root, token, states):
    root = Path(root)
    try:
        return (not root.is_symlink() and root.is_dir() and bool(token)
                and (root / '.benchmark-owner').read_text() == token
                and all(value in ('absent', 'reused') for value in states))
    except OSError:
        return False


def validate_worker_root(root, name, token):
    root = Path(root)
    if (root.is_symlink() or not root.is_dir() or root.name != name
            or not removal_allowed(root.parent, token, [])):
        raise ValueError('worker root does not belong to this benchmark run')


def _finite_creation_time(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def profile_identity_errors(identity, *, state_required=True):
    if not isinstance(identity, dict):
        return ['must be an object']
    errors = []
    expected_keys = {'pid', 'created', 'cmdline', 'owner_token', 'owner_root'}
    if state_required:
        expected_keys.add('state')
    missing_keys = expected_keys - identity.keys()
    unexpected_keys = identity.keys() - expected_keys
    if 'ownership_error' in unexpected_keys:
        errors.append('ownership error key is present')
        unexpected_keys.remove('ownership_error')
    if missing_keys:
        errors.append(f'missing identity keys: {sorted(repr(key) for key in missing_keys)}')
    if unexpected_keys:
        errors.append(f'unexpected identity keys: {sorted(repr(key) for key in unexpected_keys)}')
    if type(identity.get('pid')) is not int or identity['pid'] <= 0:
        errors.append('PID must be a positive integer')
    if not _finite_creation_time(identity.get('created')):
        errors.append('creation time must be finite')
    command_line = identity.get('cmdline')
    if not isinstance(command_line, list) or not command_line \
            or any(not isinstance(argument, str) for argument in command_line):
        errors.append('command line must be a non-empty list of strings')
    owner_token = identity.get('owner_token')
    if not isinstance(owner_token, str) or not owner_token.strip():
        errors.append('owner token must be a non-empty string')
    owner_root = identity.get('owner_root')
    if not isinstance(owner_root, str) or not Path(owner_root).is_absolute():
        errors.append('owner root must be an absolute path')
    return errors


def state(r):
    try:
        p = psutil.Process(r['pid'])
        process_created = p.create_time()
        saved_created = r.get('created') if isinstance(r, dict) else None
        if not _finite_creation_time(saved_created) or not _finite_creation_time(process_created):
            return 'unknown'
        if process_created != saved_created:
            return 'reused'
        if p.status() == psutil.STATUS_ZOMBIE:
            return 'zombie'
        return 'alive' if r.get('owner_token') and not r.get('ownership_error') else 'unknown'
    except psutil.NoSuchProcess:
        return 'absent'
    except psutil.AccessDenied:
        return 'unknown'


async def terminate_owned(identities, token, grace=2):
    for operation in ('terminate', 'kill'):
        for identity in reversed(identities):
            try:
                if state(identity) != 'alive':
                    continue
                process = psutil.Process(identity['pid'])
                if same_owned_identity(identity, record(process, token), token):
                    getattr(process, operation)()
            except psutil.Error:
                pass
        if identities:
            await asyncio.sleep(grace)
    return [{**identity, 'state': state(identity)} for identity in identities]


class Fixture(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, delay):
        super().__init__(('127.0.0.1', 0), Handler)
        self.delay, self.events, self.effects = delay, [], {}
        self.lock = threading.Lock()

    def snapshot(self, nonce):
        with self.lock:
            return {'requests': [dict(x) for x in self.events if x['nonce'] == nonce],
                    'effects': self.effects.get(nonce, 0)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_args):
        pass

    def handle_fixture(self):
        u = urlsplit(self.path)
        q = parse_qs(u.query)
        nonce = q.get('nonce', [''])[0]
        if u.path == '/audit' and self.command == 'GET':
            body = json.dumps(self.server.snapshot(nonce)).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        r = {'path': u.path, 'nonce': nonce, 'method': self.command, 'received_ns': time.perf_counter_ns()}
        status, content_type = 200, 'text/html; charset=utf-8'
        if u.path == '/fixture':
            delayed = q.get('delayed') == ['1']
            script = f'<script defer src="/slow.js?{urlencode({"nonce": nonce})}"></script>' if delayed else ''
            body = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                    f'<title>fixture-{html.escape(nonce)}</title><link rel="icon" href="data:,">{script}</head>'
                    f'<body><h1 id="marker">{html.escape(nonce)}</h1><input id="value" aria-label="value">'
                    f'<script>document.documentElement.dataset.scriptDone={json.dumps("pending" if delayed else nonce)};'
                    f'addEventListener("load",()=>{{document.documentElement.dataset.loaded={json.dumps(nonce)}}})</script>'
                    '</body></html>').encode()
        elif u.path == '/slow.js':
            time.sleep(self.server.delay)
            body = f'document.documentElement.dataset.scriptDone={json.dumps(nonce)};'.encode()
            content_type = 'text/javascript'
        elif u.path == '/effect' and self.command == 'POST':
            payload = self.rfile.read(int(self.headers.get('Content-Length', '0'))).decode()
            with self.server.lock:
                self.server.effects[nonce] = self.server.effects.get(nonce, 0) + 1
                count = self.server.effects[nonce]
            body = json.dumps({'count': count, 'nonce': nonce, 'payload': payload}).encode()
            content_type = 'application/json'
        else:
            status, body = 404, b'not found'
        r.update(status=status, bytes=len(body))
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()
            r['completed_ns'] = time.perf_counter_ns()
        except (BrokenPipeError, ConnectionResetError) as error:
            r['error'] = repr(error)
        finally:
            with self.server.lock:
                self.server.events.append(r)

    do_GET = handle_fixture
    do_POST = handle_fixture


class Cdp:
    def __init__(self):
        self.sequence, self.pending = 0, {}
        self.events = asyncio.Queue()

    async def connect(self, url):
        import websockets
        self.ws = await websockets.connect(url, max_size=16 * 1024 * 1024, proxy=None)
        self.reader = asyncio.create_task(self.read())

    async def read(self):
        try:
            async for raw in self.ws:
                message = json.loads(raw)
                if 'id' in message:
                    future = self.pending.pop(message['id'], None)
                    if future and not future.done():
                        if 'error' in message:
                            future.set_exception(RuntimeError(message['error']))
                        else:
                            future.set_result(message.get('result', {}))
                else:
                    self.events.put_nowait(message)
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError('CDP connection closed'))

    async def send(self, method, params=None, session=None):
        self.sequence += 1
        message = {'id': self.sequence, 'method': method, 'params': params or {}}
        if session:
            message['sessionId'] = session
        future = asyncio.get_running_loop().create_future()
        self.pending[self.sequence] = future
        await self.ws.send(json.dumps(message))
        return await asyncio.wait_for(future, 20)

    async def close(self):
        await self.ws.close()
        await self.reader


class Arm:
    def __init__(self, args):
        self.name, self.root = args.worker, Path(args.run_root)
        self.mode = WorkerMode(args.worker_mode)
        self.source, self.chrome = Path(args.source), Path(args.chrome) if args.chrome else None
        self.lightpanda, self.agent_browser = args.lightpanda, args.agent_browser
        self.token = os.environ['NAVIGATION_BENCHMARK_OWNER']
        validate_worker_root(self.root, self.name, self.token)
        self.proc = self.session = self.cdp = self.browser = None
        self.owned = {}

    def ownership(self):
        roots = [psutil.Process(os.getpid())]
        for pidfile in self.root.rglob('*.pid'):
            try:
                candidate = psutil.Process(int(pidfile.read_text().strip()))
                expected = {Path(sys.executable).resolve()}
                if self.agent_browser:
                    expected.add(Path(self.agent_browser).resolve())
                env = candidate.environ()
                if (Path(candidate.exe()).resolve() in expected
                        and env.get('NAVIGATION_BENCHMARK_OWNER') == self.token
                        and env.get('BH_RUNTIME_DIR') == str(self.root / 'runtime')):
                    roots.append(candidate)
            except (ValueError, psutil.Error, OSError):
                continue
        for p in roots:
            for child in [p, *p.children(recursive=True)]:
                try:
                    if child.pid != os.getpid():
                        self.owned[child.pid] = record(child, self.token)
                except psutil.Error:
                    pass
        return list(self.owned.values())

    async def cli(self, *args):
        command = [str(self.agent_browser), '--config', str(self.root / 'agent.json'),
                   '--session', 'benchmark', '--json', *args]
        proc = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 30)
        except BaseException:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()
            raise
        reply = {'command': list(args), 'returncode': proc.returncode, 'stdout': stdout.decode(), 'stderr': stderr.decode()}
        with (self.root / 'cli.jsonl').open('a') as f:
            f.write(json.dumps(reply) + '\n')
        if proc.returncode:
            raise RuntimeError(reply)
        value = json.loads(stdout)
        if not value.get('success'):
            raise RuntimeError(value)
        return value.get('data')

    async def start(self):
        if self.name in ('raw-cdp', 'lightpanda'):
            if self.name == 'raw-cdp':
                command = [str(self.chrome), '--headless=new', '--remote-debugging-port=0',
                           f'--user-data-dir={self.root / "profile"}', '--no-first-run', '--no-default-browser-check',
                           '--disable-background-networking', '--disable-component-update', '--disable-extensions',
                           '--use-mock-keychain']
                if self.mode is WorkerMode.SCREENSHOT:
                    command.append('--force-color-profile=srgb')
                command.append('about:blank')
            else:
                with socket.socket() as s:
                    s.bind(('127.0.0.1', 0))
                    port = s.getsockname()[1]
                command = [str(self.lightpanda), 'serve', '--host', '127.0.0.1', '--port', str(port)]
            self.log = (self.root / 'browser.log').open('w')
            self.proc = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
            self.ownership()
            deadline = time.monotonic() + 30
            while True:
                try:
                    if self.proc.poll() is not None:
                        raise RuntimeError(f'engine exited {self.proc.returncode}')
                    if self.name == 'raw-cdp':
                        lines = (self.root / 'profile' / 'DevToolsActivePort').read_text().splitlines()
                        if not lines:
                            raise ValueError('CDP port is not ready')
                        port = int(lines[0])
                    with local_open(f'http://127.0.0.1:{port}/json/version', timeout=1) as r:
                        self.version = json.load(r)
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('CDP startup')
                    await asyncio.sleep(.05)
            self.cdp = Cdp()
            await self.cdp.connect(self.version['webSocketDebuggerUrl'])
            target = await self.cdp.send('Target.createTarget', {'url': 'about:blank'})
            attached = await self.cdp.send('Target.attachToTarget', {'targetId': target['targetId'], 'flatten': True})
            self.session = attached['sessionId']
            await self.cdp.send('Page.enable', session=self.session)
            self.version['cdp_get_version'] = await self.cdp.send('Browser.getVersion')
        elif self.name == 'agent-browser':
            (self.root / 'agent.json').write_text(json.dumps({'executablePath': str(self.chrome), 'profile': str(self.root / 'profile'),
                                                           'headed': False, 'noWebmcp': True}))
            await self.cli('open')
            self.version = await self.cli('eval', 'JSON.stringify({userAgent:navigator.userAgent})')
        elif self.name == 'browser-use':
            from browser_use import Browser
            self.browser = Browser('chromium', headless=True, executable_path=str(self.chrome), user_data_dir=self.root / 'profile')
            env = await self.browser.start()
            os.environ.update(env)
            from browser_harness import helpers
            helpers.NAME = env['BU_NAME']
            self.helpers = helpers
            self.version = await asyncio.to_thread(helpers.cdp, 'Browser.getVersion')
        elif self.name == 'openbrowser':
            import openbrowser
            from openbrowser.browser import BrowserProfile, BrowserSession
            module = Path(openbrowser.__file__).resolve()
            if not module.is_relative_to(self.source / 'src'):
                raise RuntimeError('OpenBrowser imported from wrong source')
            browser_args = ['--use-mock-keychain']
            if self.mode is WorkerMode.SCREENSHOT:
                browser_args.append('--force-color-profile=srgb')
            self.browser = BrowserSession(browser_profile=BrowserProfile(headless=True, executable_path=str(self.chrome), args=browser_args,
                                            user_data_dir=self.root / 'profile', downloads_path=self.root / 'downloads'))
            await self.browser.start()
            self.version = await self.browser.cdp_client.send.Browser.getVersion()
        self.ownership()
        return {'version': self.version, 'owned': list(self.owned.values()),
                'packages': {x: importlib.metadata.version(x) for x in ('psutil', 'websockets')},
                'loaded_packages': {x: importlib.metadata.version(x) for x in ('browser-use', 'browser-harness', 'openbrowser-ai')
                                    if x in {d.metadata['Name'] for d in importlib.metadata.distributions()}}}

    async def navigate(self, url):
        if self.name in ('raw-cdp', 'lightpanda'):
            while not self.cdp.events.empty():
                self.cdp.events.get_nowait()
            response = await self.cdp.send('Page.navigate', {'url': url}, self.session)
            if response.get('errorText'):
                raise RuntimeError(response)
            while True:
                event = await asyncio.wait_for(self.cdp.events.get(), 15)
                if event.get('method') == 'Page.loadEventFired' and event.get('sessionId') == self.session:
                    return response
        elif self.name == 'agent-browser':
            await self.cli('open', url)
            return await self.cli('wait', '--load', 'load')
        elif self.name == 'browser-use':
            def operation():
                result = self.helpers.goto_url(url)
                if not self.helpers.wait_for_load(timeout=15):
                    raise TimeoutError('browser-use prescribed wait_for_load')
                return result
            return await asyncio.to_thread(operation)
        else:
            return await self.browser.navigate_to(url, wait_until='load')

    async def evaluate(self, expression):
        if self.name in ('raw-cdp', 'lightpanda'):
            result = await self.cdp.send('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True}, self.session)
        elif self.name == 'agent-browser':
            data = await self.cli('eval', expression)
            return data['result']
        elif self.name == 'browser-use':
            result = await asyncio.to_thread(self.helpers.cdp, 'Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
        else:
            session = await self.browser.get_or_create_cdp_session()
            result = await session.cdp_client.send.Runtime.evaluate(params={'expression': expression, 'returnByValue': True, 'awaitPromise': True}, session_id=session.session_id)
        if result.get('exceptionDetails'):
            raise RuntimeError(result['exceptionDetails'])
        return result['result']['value']

    async def wait_for_screenshot_ready(self, nonce):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            observation = json.loads(await self.evaluate(SCREENSHOT_OBSERVE))
            if observation.get('readyState') == 'complete' \
                    and observation.get('readyMarker') == nonce \
                    and observation.get('imageReady') is True:
                return
            await asyncio.sleep(.05)
        raise TimeoutError('screenshot fixture did not reach its ready marker')

    async def capture_screenshot(self, request: ScreenshotRequest):
        if self.mode is not WorkerMode.SCREENSHOT:
            raise ValueError('capture_screenshot requires screenshot mode')
        if self.name == 'openbrowser':
            await self.browser.navigate_to(request.url, wait_until='load')
            await self.wait_for_screenshot_ready(request.nonce)
            return await self.browser.screenshot_element(
                request.selector, path=request.output, format='png',
            )
        if self.name == 'raw-cdp':
            await self.cdp.send('Emulation.setDeviceMetricsOverride', {
                'width': 128, 'height': 128, 'deviceScaleFactor': 1, 'mobile': False,
            }, self.session)
            await self.navigate(request.url)
            await self.wait_for_screenshot_ready(request.nonce)
            result = await self.cdp.send('Page.captureScreenshot', {
                'format': 'png', 'fromSurface': True, 'captureBeyondViewport': False,
                'clip': {'x': 0, 'y': 0, 'width': 128, 'height': 128, 'scale': 1},
            }, self.session)
            try:
                content = base64.b64decode(result['data'], validate=True)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('CDP screenshot reply has invalid PNG data') from error
            Path(request.output).write_bytes(content)
            return content
        raise ValueError('capture_screenshot supports only OpenBrowser and raw CDP')

    async def trial(self, request):
        start = time.perf_counter_ns()
        result = {**request, 'start_ns': start, 'outcome': 'failure'}
        try:
            async with asyncio.timeout(25):
                await self.navigate(request['url'])
                result['navigation_return_ns'] = time.perf_counter_ns()
                result['navigation_observation'] = json.loads(await self.evaluate(OBSERVE))
                navigation_errors = navigation_errors_for(result)
                if navigation_errors:
                    raise AssertionError(navigation_errors)
                n = json.dumps(request['nonce'])
                expression = f'''(async()=>{{const n={n};const f=document.getElementById('value');f.value=n;
const r=await fetch('/effect?nonce='+encodeURIComponent(n),{{method:'POST',body:n}});
return JSON.stringify({{field:f.value,response:await r.json()}})}})()'''
                result['effect_observation'] = json.loads(await self.evaluate(expression))
                result['final_observation'] = json.loads(await self.evaluate(OBSERVE))
                audit = request['url'].replace('/fixture?', '/audit?')
                def fetch_audit():
                    with local_open(audit, timeout=2) as response:
                        return json.load(response)
                for _ in range(100):
                    result['server'] = await asyncio.to_thread(fetch_audit)
                    if any(r['path'] == '/effect' for r in result['server']['requests']):
                        break
                    await asyncio.sleep(.001)
                result['checked_ns'] = time.perf_counter_ns()
                errors = workload_errors(result, result['server'], result['checked_ns'])
                if errors:
                    raise AssertionError(errors)
                result['outcome'] = 'success'
        except Exception as error:
            result['exception'] = f'{type(error).__name__}: {error}'
            result['timed_out'] = isinstance(error, TimeoutError)
            result['traceback'] = traceback.format_exc()
        result['end_ns'] = time.perf_counter_ns()
        result['elapsed_ns'] = result['end_ns'] - start
        if 'navigation_return_ns' in result:
            result['navigation_ns'] = result['navigation_return_ns'] - start
            result['post_navigation_ns'] = result['end_ns'] - result['navigation_return_ns']
        self.ownership()
        return result

    async def stop(self):
        self.ownership()
        failures = []
        try:
            async with asyncio.timeout(10):
                if self.name == 'agent-browser':
                    await self.cli('close')
                elif self.browser:
                    if self.name == 'browser-use':
                        await self.browser.stop()
                    else:
                        await self.browser.kill()
                elif self.cdp:
                    await self.cdp.send('Browser.close')
                    await self.cdp.close()
        except Exception as error:
            failures.append(repr(error))
        self.ownership()
        identities = await terminate_owned(list(self.owned.values()), self.token)
        if self.proc:
            try:
                await asyncio.to_thread(self.proc.wait, 2)
            except subprocess.TimeoutExpired:
                failures.append('owned engine did not exit after verified termination')
        if hasattr(self, 'log'):
            self.log.close()
        identities = [{**identity, 'state': state(identity)} for identity in identities]
        return {'errors': failures, 'identities': identities}


def validate_profile_pair_request(arm, request):
    if arm.name != 'openbrowser':
        raise ValueError('profile_pair is supported only by the OpenBrowser worker')
    if not isinstance(request, dict) or set(request) != {'op', 'workload', 'fixture_port', 'warmup', 'target'}:
        raise ValueError('profile_pair request has unsupported fields')
    if request.get('op') != 'profile_pair' or request.get('workload') not in ('fast', 'delayed'):
        raise ValueError('profile_pair workload is invalid')
    port = request.get('fixture_port')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('profile_pair fixture_port is invalid')
    expected_trial_fields = {'arm', 'kind', 'index', 'nonce'}
    for label in ('warmup', 'target'):
        trial = request.get(label)
        if not isinstance(trial, dict) or set(trial) != expected_trial_fields:
            raise ValueError(f'profile_pair {label} request has unsupported fields')
        if trial.get('arm') != 'openbrowser' or type(trial.get('index')) is not int or trial['index'] != 0:
            raise ValueError(f'profile_pair {label} must select OpenBrowser index zero')
        if not isinstance(trial.get('nonce'), str) or not trial['nonce']:
            raise ValueError(f'profile_pair {label} nonce is invalid')
    if request['warmup']['kind'] != 'fast':
        raise ValueError('profile_pair warmup must be fast')
    if request['target']['kind'] != request['workload']:
        raise ValueError('profile_pair target must match the selected workload')
    if request['warmup']['nonce'] == request['target']['nonce']:
        raise ValueError('profile_pair warmup and target nonces must differ')
    return port


def profile_trial_request(spec, port):
    query = urlencode({'nonce': spec['nonce'], 'delayed': int(spec['kind'] == 'delayed')})
    return {**spec, 'url': f'http://127.0.0.1:{port}/fixture?{query}'}


def profile_trial_record(sample, request, label):
    errors = []
    if not isinstance(sample, dict):
        return {'expected': 1, 'attempted': 1, 'outcome': 'failure', 'sample': None}, [f'{label}: trial returned no record']
    for key in ('arm', 'kind', 'nonce', 'url'):
        if sample.get(key) != request.get(key):
            errors.append(f'{label}: {key} does not match the requested trial')
    if type(sample.get('index')) is not int or sample.get('index') != request.get('index'):
        errors.append(f'{label}: index must be an exact integer matching the requested trial')
    if sample.get('outcome') not in ('success', 'failure'):
        errors.append(f'{label}: outcome is invalid')
    try:
        errors.extend(trial_proof_errors(sample, label))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f'{label}: invalid trial evidence: {error}')
    outcome = sample.get('outcome') if sample.get('outcome') in ('success', 'failure') else 'failure'
    return {'expected': 1, 'attempted': 1, 'outcome': outcome, 'sample': sample}, errors


async def profile_pair_operation(arm, args, request, started):
    port = validate_profile_pair_request(arm, request)
    if not started:
        raise ValueError('OpenBrowser worker must start before profile_pair')
    warmup_request = profile_trial_request(request['warmup'], port)
    target_request = profile_trial_request(request['target'], port)
    errors = []
    warmup = {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    target = {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    profile = {'enabled': False, 'disabled': False, 'dumped': False, 'file': 'trial.pstats'}
    failed_phase = None
    progress = {
        'schema': 'openbrowser-navigation-progress/1',
        'phase': 'ready',
        'warmup': {'attempted': 0, 'outcome': 'not_run'},
        'target': {'attempted': 0, 'outcome': 'not_run'},
    }

    def result():
        return {
            'warmup': warmup,
            'target': target,
            'profile': profile,
            'browser_runtime': started.get('version'),
            'packages': {
                'packages': started.get('packages', {}),
                'loaded_packages': started.get('loaded_packages', {}),
            },
            'failed_phase': failed_phase,
            'errors': errors,
        }

    def save_progress(phase):
        nonlocal failed_phase
        progress['phase'] = phase
        try:
            atomic_profile_json(Path(args.run_root) / 'profile-progress.json', progress)
        except Exception as error:
            errors.append(f'profile progress write failed: {type(error).__name__}: {error}')
            if failed_phase is None:
                failed_phase = 'profile'
            return False
        return True

    if not save_progress('ready'):
        return result()
    progress['warmup'] = {'attempted': 1, 'outcome': 'in_progress'}
    if not save_progress('warmup'):
        return result()

    try:
        warmup_sample = await arm.trial(warmup_request)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        warmup = {
            'expected': 1, 'attempted': 1, 'outcome': 'failure',
            'sample': {'exception': f'{type(error).__name__}: {error}', 'outcome': 'failure'},
        }
        errors.append(f'warmup: {type(error).__name__}: {error}')
        failed_phase = 'warmup'
    else:
        warmup, warmup_errors = profile_trial_record(warmup_sample, warmup_request, 'warmup')
        errors.extend(warmup_errors)
        if warmup['outcome'] != 'success' or warmup_errors:
            if warmup['outcome'] == 'success':
                warmup['outcome'] = 'failure'
            errors.append('warmup did not produce one verified successful fast sample')
            failed_phase = 'warmup'

    progress['warmup'] = {'attempted': 1, 'outcome': warmup['outcome']}
    if not save_progress('warmup'):
        return result()

    if failed_phase is None:
        if not save_progress('profile'):
            return result()
        progress['target'] = {'attempted': 1, 'outcome': 'in_progress'}
        if not save_progress('target'):
            return result()

        target_not_returned = object()
        target_sample = target_not_returned
        target_error = None
        disable_error = None
        profiler = None
        try:
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
        except Exception as error:
            errors.append(f'profile enable failed: {type(error).__name__}: {error}')
            if failed_phase is None:
                failed_phase = 'profile'
            if profiler is not None:
                try:
                    profiler.disable()
                    profile['disabled'] = True
                except Exception as disable_error:
                    errors.append(
                        f'profile disable after enable failure failed: '
                        f'{type(disable_error).__name__}: {disable_error}'
                    )
            progress['target'] = {'attempted': 0, 'outcome': 'not_run'}
            if not save_progress('profile'):
                return result()
        else:
            try:
                try:
                    target_sample = await arm.trial(target_request)
                finally:
                    try:
                        profiler.disable()
                        profile['disabled'] = True
                    except Exception as error:
                        disable_error = error
            except asyncio.CancelledError:
                raise
            except Exception as error:
                target_error = error

            profile['enabled'] = True
            target_returned = target_sample is not target_not_returned
            if target_error is not None:
                target = {
                    'expected': 1, 'attempted': 1, 'outcome': 'failure',
                    'sample': {'exception': f'{type(target_error).__name__}: {target_error}', 'outcome': 'failure'},
                }
                errors.append(f'target: {type(target_error).__name__}: {target_error}')
                if failed_phase is None:
                    failed_phase = 'target'
            else:
                try:
                    target, target_errors = profile_trial_record(target_sample, target_request, 'target')
                    errors.extend(target_errors)
                    if target['outcome'] != 'success' or target_errors:
                        if target['outcome'] == 'success':
                            target['outcome'] = 'failure'
                        errors.append('target did not produce one verified successful selected sample')
                        if failed_phase is None:
                            failed_phase = 'target'
                except Exception as error:
                    target = {
                        'expected': 1, 'attempted': 1, 'outcome': 'failure',
                        'sample': {'exception': f'{type(error).__name__}: {error}', 'outcome': 'failure'},
                    }
                    errors.append(f'target sample validation failed: {type(error).__name__}: {error}')
                    if failed_phase is None:
                        failed_phase = 'target'

            progress['target'] = {'attempted': 1, 'outcome': target['outcome']}
            progress_saved = save_progress('profile')

            if disable_error is not None:
                errors.append(f'profile disable failed: {type(disable_error).__name__}: {disable_error}')
                if failed_phase is None:
                    failed_phase = 'profile'

            if not progress_saved:
                return result()
            if target_returned and profile['disabled']:
                stats_path = Path(args.run_root) / 'trial.pstats'
                try:
                    if stats_path.exists() or stats_path.is_symlink():
                        raise FileExistsError('worker profile destination already exists')
                    profiler.dump_stats(str(stats_path))
                    if stats_path.is_symlink() or not stats_path.is_file():
                        raise OSError('worker profile stats are not a regular file')
                    stats_path.chmod(0o600)
                    profile['dumped'] = True
                except Exception as error:
                    errors.append(f'profile dump failed: {type(error).__name__}: {error}')
                    if failed_phase is None:
                        failed_phase = 'profile'

    save_progress('complete')

    return result()


async def capture_screenshot_operation(arm, value):
    request = parse_screenshot_request(arm.name, value)
    working_directory = Path.cwd()
    directory_info = working_directory.lstat()
    if not stat.S_ISDIR(directory_info.st_mode) or stat.S_IMODE(directory_info.st_mode) != 0o700 \
            or directory_info.st_uid != os.geteuid():
        raise ValueError('screenshot worker directory must be private')
    output = working_directory / request.output
    if output.exists() or output.is_symlink():
        raise ValueError('screenshot output already exists')
    try:
        await arm.capture_screenshot(request)
        info = output.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('screenshot output is not a regular file')
        os.chmod(output, 0o600, follow_symlinks=False)
        info = output.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 \
                or info.st_uid != os.geteuid():
            raise ValueError('screenshot output is not a private regular file')
        content = output.read_bytes()
        if not content.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('screenshot output is not a PNG')
        return asdict(ScreenshotReply(
            schema='openbrowser-screenshot-reply/1',
            attempt_id=request.attempt_id,
            arm=request.arm,
            nonce=request.nonce,
            ready_marker=request.nonce,
            output=request.output,
            byte_count=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        ))
    except BaseException:
        try:
            if output.is_symlink() or output.is_file():
                output.unlink()
        except OSError:
            pass
        raise


async def worker(args):
    mode = WorkerMode(args.worker_mode)
    arm = Arm(args)
    state = WorkerState.STARTING
    stopped = False
    started = None

    def error_reply(error):
        trace = ''.join(traceback.format_exception(error)) if error.__traceback__ else ''
        return {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': trace}

    def reject_operation(error):
        nonlocal state
        if state in (WorkerState.STARTING, WorkerState.READY):
            state = transition_worker_state(state, WorkerEvent.INVALID_OPERATION)
        elif state is WorkerState.OPERATION_RUNNING:
            state = transition_worker_state(state, WorkerEvent.OPERATION_FAILED)
        elif state in (WorkerState.OPERATION_COMPLETE, WorkerState.FAILED):
            state = transition_worker_state(state, WorkerEvent.INVALID_OPERATION)
        elif state is WorkerState.STOPPED:
            state = transition_worker_state(state, WorkerEvent.LATE_OPERATION)
        return error_reply(error)

    def write_reply(reply):
        sys.stdout.write(PREFIX + json.dumps(reply) + '\n')
        sys.stdout.flush()

    try:
        while raw := await asyncio.to_thread(sys.stdin.readline):
            try:
                request = json.loads(raw)
                if not isinstance(request, dict) or not isinstance(request.get('op'), str):
                    raise ValueError('worker request must be an object with a string op')
            except Exception as error:
                write_reply(reject_operation(error))
                continue

            if request['op'] == 'stop':
                try:
                    state = transition_worker_state(state, WorkerEvent.CLEANUP)
                    result = await arm.stop()
                    identities = result.get('identities') if isinstance(result, dict) else None
                    cleanup_verified = (
                        isinstance(identities, list)
                        and all(
                            isinstance(identity, dict)
                            and identity.get('state') in ('absent', 'reused')
                            for identity in identities
                        )
                    )
                    if cleanup_verified:
                        state = transition_worker_state(state, WorkerEvent.CLEANUP_VERIFIED)
                        reply = {'ok': True, 'result': result}
                    else:
                        state = transition_worker_state(state, WorkerEvent.CLEANUP_UNCERTAIN)
                        reply = {'ok': False, 'error': 'worker process cleanup is incomplete', 'result': result}
                except Exception as error:
                    if state is WorkerState.STOPPING:
                        state = transition_worker_state(state, WorkerEvent.CLEANUP_UNCERTAIN)
                    reply = error_reply(error)
                stopped = True
                write_reply(reply)
                break

            try:
                if request['op'] == 'start':
                    if state is not WorkerState.STARTING:
                        raise ValueError('worker start is allowed only once')
                    if mode is WorkerMode.PROFILE and arm.name != 'openbrowser':
                        raise ValueError('profile mode supports only OpenBrowser')
                    if mode is WorkerMode.SCREENSHOT and arm.name not in ('openbrowser', 'raw-cdp'):
                        raise ValueError('screenshot mode supports only OpenBrowser and raw CDP')
                    result = await arm.start()
                    if not isinstance(result, dict) or not isinstance(result.get('owned'), list):
                        raise ValueError('worker start reply has no owned-process inventory')
                    started = result
                    state = transition_worker_state(state, WorkerEvent.START_SUCCEEDED)
                    reply = {'ok': True, 'result': result}
                elif state is not WorkerState.READY:
                    raise ValueError(f'worker operation {request["op"]!r} is not allowed in state {state.value}')
                elif mode is WorkerMode.NAVIGATION and request['op'] == 'trial':
                    state = transition_worker_state(state, WorkerEvent.VALID_OPERATION)
                    result = await arm.trial(request)
                    required = ('arm', 'kind', 'index', 'nonce', 'url', 'outcome',
                                'start_ns', 'end_ns', 'elapsed_ns')
                    if (not isinstance(result, dict)
                            or any(key not in result for key in required)
                            or result.get('arm') != arm.name
                            or result.get('outcome') not in ('success', 'failure')
                            or type(result.get('index')) is not int
                            or any(type(result.get(key)) is not int or result[key] < 0
                                   for key in ('start_ns', 'end_ns', 'elapsed_ns'))):
                        raise ValueError('navigation trial did not return a complete sample')
                    state = transition_worker_state(state, WorkerEvent.NAVIGATION_RESULT)
                    reply = {'ok': True, 'result': result}
                elif mode is WorkerMode.PROFILE and request['op'] == 'profile_pair':
                    state = transition_worker_state(state, WorkerEvent.VALID_OPERATION)
                    result = await profile_pair_operation(arm, args, request, started)
                    if not isinstance(result, dict):
                        raise ValueError('profile operation did not return a complete result')
                    state = transition_worker_state(state, WorkerEvent.ONE_SHOT_RESULT)
                    reply = {'ok': True, 'result': result}
                elif mode is WorkerMode.SCREENSHOT and request['op'] == 'capture_screenshot':
                    state = transition_worker_state(state, WorkerEvent.VALID_OPERATION)
                    result = await capture_screenshot_operation(arm, request)
                    if not isinstance(result, dict):
                        raise ValueError('screenshot operation did not return a complete result')
                    state = transition_worker_state(state, WorkerEvent.ONE_SHOT_RESULT)
                    reply = {'ok': True, 'result': result}
                else:
                    raise ValueError(f'worker operation {request["op"]!r} is not allowed for mode {mode!r}')
            except Exception as error:
                if state is WorkerState.OPERATION_RUNNING:
                    state = transition_worker_state(state, WorkerEvent.OPERATION_FAILED)
                elif state in (WorkerState.STARTING, WorkerState.READY):
                    event = (WorkerEvent.START_FAILED if state is WorkerState.STARTING
                             and request['op'] == 'start' else WorkerEvent.INVALID_OPERATION)
                    state = transition_worker_state(state, event)
                reply = error_reply(error)
            write_reply(reply)
    finally:
        if not stopped:
            await arm.stop()


async def send(proc, request, log, timeout=90):
    proc.stdin.write((json.dumps(request) + '\n').encode())
    await proc.stdin.drain()
    async with asyncio.timeout(timeout):
        while raw := await proc.stdout.readline():
            log.write(raw.decode(errors='replace'))
            log.flush()
            if raw.startswith(PREFIX.encode()):
                return json.loads(raw[len(PREFIX):])
    raise RuntimeError('worker stdout closed without reply')


def descendant_identities(identity, token):
    try:
        process = psutil.Process(identity['pid'])
        if same_owned_identity(identity, record(process, token), token):
            return [record(child, token) for child in process.children(recursive=True)]
    except psutil.Error:
        pass
    return []


def identity_state(identity):
    if profile_identity_errors(identity, state_required=False):
        return 'unknown'
    return state(identity)


async def cleanup_worker(handle: WorkerHandle) -> dict:
    if handle.cleanup_result is not None:
        return handle.cleanup_result

    cleanup_errors = []
    captured = {}
    invalid_identities = []
    if handle.initial_owned is None:
        cleanup_errors.append('initial owned-process inventory is unknown')
    else:
        for identity in handle.initial_owned:
            errors = profile_identity_errors(identity, state_required=False)
            if errors:
                invalid_identities.append(identity)
                cleanup_errors.append(f'initial owned-process identity is invalid: {errors!r}')
            else:
                captured[(identity['pid'], identity['created'])] = identity

    handle.state = transition_worker_state(handle.state, WorkerEvent.CLEANUP)
    cleanup = {}
    try:
        response = await send(handle.process, {'op': 'stop'}, handle.stdout_log, 60)
        if not isinstance(response, dict):
            cleanup_errors.append('worker stop reply is not an object')
        else:
            cleanup.update(response)
            if response.get('ok') is not True:
                cleanup_errors.append(response.get('error', 'worker stop failed'))
            result = response.get('result')
            if not isinstance(result, dict):
                cleanup_errors.append('worker stop reply has no result object')
            else:
                identities = result.get('identities')
                if not isinstance(identities, list):
                    cleanup_errors.append('worker stop reply has no identity list')
                else:
                    for identity in identities:
                        errors = profile_identity_errors(identity, state_required=True)
                        if errors:
                            invalid_identities.append(identity)
                            cleanup_errors.append(f'worker stop identity is invalid: {errors!r}')
                        else:
                            captured[(identity['pid'], identity['created'])] = {
                                key: value for key, value in identity.items() if key != 'state'
                            }
                worker_errors = result.get('errors')
                if not isinstance(worker_errors, list):
                    cleanup_errors.append('worker stop result has no error list')
                elif worker_errors:
                    cleanup_errors.extend(f'worker stop: {error}' for error in worker_errors)
    except Exception as error:
        cleanup_errors.append(f'{type(error).__name__}: {error}')
    finally:
        try:
            handle.process.stdin.close()
        except Exception as error:
            cleanup_errors.append(f'cannot close worker input: {type(error).__name__}: {error}')

    try:
        await asyncio.wait_for(handle.process.wait(), 15)
    except Exception as error:
        cleanup_errors.append(f'worker did not exit: {type(error).__name__}: {error}')

    worker_identity_errors = profile_identity_errors(handle.worker_identity, state_required=False)
    worker_state = identity_state(handle.worker_identity)
    child_states = {key: identity_state(identity) for key, identity in captured.items()}
    fallback_used = worker_state == 'alive' or 'alive' in child_states.values()
    if fallback_used:
        if worker_identity_errors:
            cleanup_errors.append('worker identity is unverified; refusing to signal processes')
        else:
            for identity in descendant_identities(handle.worker_identity, handle.owner_token):
                errors = profile_identity_errors(identity, state_required=False)
                if errors:
                    invalid_identities.append(identity)
                    cleanup_errors.append(f'descendant identity is invalid: {errors!r}')
                else:
                    captured[(identity['pid'], identity['created'])] = identity
            await terminate_owned([handle.worker_identity, *captured.values()], handle.owner_token)
            try:
                await asyncio.wait_for(handle.process.wait(), 5)
            except Exception as error:
                cleanup_errors.append(f'worker did not exit after verified termination: {type(error).__name__}: {error}')

    worker_state = identity_state(handle.worker_identity)
    post_worker_exit_identities = [
        {**identity, 'state': identity_state(identity)}
        for identity in captured.values()
    ]
    post_worker_exit_identities.extend(
        {**identity, 'state': 'unknown'} if isinstance(identity, dict) else {'state': 'unknown'}
        for identity in invalid_identities
    )
    worker_identity = {**handle.worker_identity, 'state': worker_state}
    cleanup['fallback_used'] = fallback_used
    cleanup['post_worker_exit_identities'] = post_worker_exit_identities
    cleanup['worker_identity'] = worker_identity

    retired = worker_state in ('absent', 'reused') and all(
        identity['state'] in ('absent', 'reused') for identity in post_worker_exit_identities
    )
    cleanup['ok'] = not cleanup_errors and retired and handle.process.returncode is not None
    if cleanup['ok']:
        handle.state = transition_worker_state(handle.state, WorkerEvent.CLEANUP_VERIFIED)
    else:
        cleanup['error'] = '; '.join(str(error) for error in cleanup_errors) or 'worker cleanup could not be verified'
        handle.state = transition_worker_state(handle.state, WorkerEvent.CLEANUP_UNCERTAIN)
    handle.cleanup_result = cleanup
    return cleanup


def navigation_errors_for(sample):
    n = sample['nonce']
    expected = {'url': sample['url'], 'title': f'fixture-{n}', 'readyState': 'complete',
                'marker': n, 'loaded': n, 'scriptDone': n}
    actual = sample.get('navigation_observation', {})
    return [f'navigation {key}: expected {value!r}, got {actual.get(key)!r}'
            for key, value in expected.items() if actual.get(key) != value]


def workload_errors(sample, server, through_ns):
    errors = navigation_errors_for(sample)
    n = sample['nonce']
    final = sample.get('final_observation', {})
    errors.extend('final ' + error for error in navigation_errors_for({**sample, 'navigation_observation': final}))
    if final.get('field') != n:
        errors.append('final form value does not match nonce')
    effect = sample.get('effect_observation', {})
    if effect != {'field': n, 'response': {'count': 1, 'nonce': n, 'payload': n}}:
        errors.append(f'effect mismatch: {effect!r}')
    if server.get('effects') != 1:
        errors.append(f"server effect count {server.get('effects')} != 1")
    needed = {'/fixture': 'GET', '/effect': 'POST'}
    if sample['kind'] == 'delayed':
        needed['/slow.js'] = 'GET'
    for path, method in needed.items():
        found = [r for r in server.get('requests', []) if r['path'] == path and r['method'] == method]
        if (len(found) != 1 or found[0].get('status') != 200 or found[0].get('nonce') != n
                or not sample['start_ns'] <= found[0].get('received_ns', 0)
                <= found[0].get('completed_ns', 0) <= through_ns):
            errors.append(f'required response not completed once inside timer: {path}: {found!r}')
    return errors


def summarize(samples, states, count):
    result = {}
    for name, status in states.items():
        for kind in ('fast', 'delayed'):
            matching = [s for s in samples if s['arm'] == name and s['kind'] == kind]
            good = [s['elapsed_ns'] for s in matching if s['outcome'] == 'success']
            unavailable = count if status['state'] == 'unavailable' else 0
            result[f'{name}/{kind}'] = {
                'expected': count, 'attempted': len(matching), 'successful': len(good),
                'failures': len(matching) - len(good), 'timeouts': sum(bool(s.get('timed_out')) for s in matching),
                'unavailable': unavailable, 'not_run': max(0, count - len(matching) - unavailable),
                'median_ns': statistics.median(good) if good else None,
                'min_ns': min(good) if good else None, 'max_ns': max(good) if good else None,
            }
    return result


def trial_proof_errors(sample, label):
    errors = []
    if sample.get('elapsed_ns') != sample.get('end_ns', 0) - sample.get('start_ns', 0):
        errors.append(f'{label}: invalid elapsed_ns')
    if sample['outcome'] == 'success':
        errors.extend(f'{label}: {e}' for e in workload_errors(sample, sample.get('server', {}), sample['end_ns']))
        if not sample['start_ns'] <= sample.get('checked_ns', 0) <= sample['end_ns']:
            errors.append(f'{label}: assertions were not checked inside timer')
    return errors


def result_errors(result):
    errors = []
    count = result['metadata']['samples_per_workload_arm']
    expected = summarize(result['samples'], result['availability'], count)
    if result.get('summary') != expected:
        errors.append('summary does not match raw samples')
    for key, summary in expected.items():
        for field in ('failures', 'unavailable', 'not_run'):
            if summary[field]:
                errors.append(f'{key}: {field}={summary[field]}')
        if summary['attempted'] != count:
            errors.append(f'{key}: expected {count} attempts, got {summary["attempted"]}')
    for name, status in result['availability'].items():
        if status['state'] != 'ready':
            errors.append(f'{name}: not ready, {status["state"]}')
            continue
        cleanup = result.get('cleanup', {}).get(name, {})
        identities = [cleanup.get('worker_identity', {}), *cleanup.get('post_worker_exit_identities', [])]
        if not cleanup.get('ok') or any(r.get('state') not in ('absent', 'reused') for r in identities):
            errors.append(f'{name}: cleanup identity proof is incomplete')
        warmups = [s for s in result.get('warmup', []) if s['arm'] == name]
        if len(warmups) != 1:
            errors.append(f'{name}: expected exactly one warmup, got {len(warmups)}')
    for warmup in result.get('warmup', []):
        name = warmup['arm']
        label = f'warmup {name}'
        if result['availability'].get(name, {}).get('state') != 'ready':
            errors.append(f'{label}: arm is not ready')
        if warmup['kind'] != 'fast' or warmup['index'] != 0:
            errors.append(f'{label}: expected fast workload at index zero')
        if warmup['outcome'] != 'success':
            errors.append(f'{label}: outcome={warmup["outcome"]}')
        errors.extend(trial_proof_errors(warmup, label))
    seen = set()
    for sample in result['samples']:
        key = (sample['arm'], sample['kind'], sample['index'])
        if key in seen or not 0 <= sample['index'] < count or sample['arm'] not in result['availability'] or sample['kind'] not in ('fast', 'delayed'):
            errors.append(f'invalid or duplicate sample key: {key}')
        seen.add(key)
        errors.extend(trial_proof_errors(sample, key))
    if not result['metadata'].get('source_unchanged'):
        errors.append('source changed during run')
    metadata = result['metadata']
    if metadata.get('source_manifest') != metadata.get('source_manifest_end'):
        errors.append('source manifests do not match')
    if not result.get('cleanup', {}).get('root_removed'):
        errors.append('owned runtime cleanup is incomplete')
    if result.get('error'):
        errors.append(result['error'])
    return errors


def local_open(url, timeout):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=timeout)


def worker_environment(launch: WorkerLaunch) -> dict[str, str]:
    validate_worker_launch(launch)

    worker_root = Path(launch.worker_root)
    interpreter = Path(launch.interpreter)
    path_parts = [str(interpreter.parent)]
    if launch.node is not None:
        node = Path(launch.node)
        if not node.is_absolute():
            raise ValueError('worker Node executable must be absolute')
        path_parts.append(str(node.parent))
    path_parts.extend(('/usr/bin', '/bin', '/usr/sbin', '/sbin'))

    env = {
        'PATH': os.pathsep.join(path_parts),
        'VIRTUAL_ENV': str(interpreter.parent.parent),
        'HOME': str(worker_root / 'home'),
        'XDG_CONFIG_HOME': str(worker_root / 'config'),
        'XDG_CACHE_HOME': str(worker_root / 'cache'),
        'TMPDIR': str(worker_root / 'tmp'),
        'LANG': 'C',
        'PYTHON_DOTENV_DISABLED': '1',
        'NAVIGATION_BENCHMARK_OWNER': launch.owner_token,
        'BH_HOME': str(worker_root / 'bh'),
        'BH_RUNTIME_DIR': str(worker_root / 'runtime'),
        'BH_TMP_DIR': str(worker_root / 'tmp'),
        'BH_TAB_MARKER': '0',
        'BH_RECORD': '0',
        'ANONYMIZED_TELEMETRY': 'false',
        'LIGHTPANDA_DISABLE_TELEMETRY': 'true',
        'BH_TELEMETRY': 'false',
        'BROWSER_HARNESS_TELEMETRY': 'false',
        'BROWSER_USE_SETUP_LOGGING': 'false',
        'BROWSER_USE_NO_UPDATE_CHECK': '1',
        'AGENT_BROWSER_SOCKET_DIR': str(worker_root / 'agent'),
        'AGENT_BROWSER_NO_VERSION_CHECK': '1',
        'AGENT_BROWSER_NO_WEBMCP': '1',
        'DO_NOT_TRACK': '1',
        'OPENBROWSER_HEADLESS': 'true',
        'OPENBROWSER_DISABLE_EXTENSIONS': 'true',
        'AWS_EC2_METADATA_DISABLED': 'true',
        'AWS_CONFIG_FILE': '/dev/null',
        'AWS_SHARED_CREDENTIALS_FILE': '/dev/null',
    }
    if launch.arm == 'openbrowser':
        env['PYTHONPATH'] = str(Path(launch.source_root) / 'src')
    return env


def prepare_worker_directories(env: dict[str, str]) -> None:
    for name in ('HOME', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'TMPDIR', 'BH_HOME',
                 'BH_RUNTIME_DIR', 'BH_TMP_DIR', 'AGENT_BROWSER_SOCKET_DIR'):
        path = Path(env[name])
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f'worker environment path is not a private directory: {name}')
        path.chmod(0o700)


def validate_private_directory(path):
    physical_root, directory = physical_profile_path(path)
    for part in directory.relative_to(physical_root).parts:
        physical_root /= part
        info = physical_root.lstat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 \
                or info.st_uid != os.geteuid():
            raise ValueError(f'benchmark path is not a private directory: {physical_root}')
    return directory


def worker_command(launch: WorkerLaunch) -> list[str]:
    command = [
        str(launch.interpreter), '-u', str(SCRIPT), '--worker', launch.arm,
        '--worker-mode', launch.mode.value, '--run-root', str(launch.worker_root),
        '--source', str(launch.source_root),
    ]
    for option in ('chrome', 'lightpanda', 'agent_browser'):
        value = getattr(launch, option)
        if value is not None:
            command.extend(['--' + option.replace('_', '-'), str(value)])
    return command


async def launch_worker(launch: WorkerLaunch) -> WorkerHandle:
    validate_worker_launch(launch)
    if not launch.interpreter.is_file() or not os.access(launch.interpreter, os.X_OK):
        raise ValueError('worker interpreter is unavailable or not executable')
    for name in ('chrome', 'lightpanda', 'agent_browser', 'node'):
        path = getattr(launch, name)
        if path is not None and (not path.is_file() or not os.access(path, os.X_OK)):
            raise ValueError(f'worker executable {name} is unavailable or not executable')

    physical_root = Path('/tmp').resolve(strict=True)
    runtime_root = validate_private_directory(launch.runtime_root)
    if runtime_root.parent != physical_root:
        raise ValueError('RUNTIME_ROOT must be a direct child of physical /tmp')
    marker = runtime_root / '.benchmark-owner'
    try:
        marker_info = marker.lstat()
    except OSError as error:
        raise ValueError('RUNTIME_ROOT has no private owner marker') from error
    if not stat.S_ISREG(marker_info.st_mode) or stat.S_IMODE(marker_info.st_mode) != 0o600 \
            or marker_info.st_uid != os.geteuid() or marker.read_text() != launch.owner_token:
        raise ValueError('RUNTIME_ROOT owner marker does not match this worker')

    log_directory = validate_private_directory(launch.log_directory)
    worker_root = Path(launch.worker_root)
    if worker_root.exists() or worker_root.is_symlink():
        raise ValueError('worker root already exists')
    worker_root.mkdir(mode=0o700)
    worker_root.chmod(0o700)
    working_directory = Path(launch.working_directory)
    if working_directory == worker_root:
        validate_private_directory(working_directory)
    else:
        validate_private_directory(working_directory)
    if working_directory == launch.source_root or working_directory.is_relative_to(launch.source_root):
        raise ValueError('worker working directory must not be inside the source checkout')

    env = worker_environment(launch)
    prepare_worker_directories(env)
    log_paths = (
        log_directory / f'{launch.log_prefix}.stdout.log',
        log_directory / f'{launch.log_prefix}.stderr.log',
    )
    descriptors = []
    streams = []
    try:
        for path in log_paths:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            descriptors.append(descriptor)
            os.chmod(path, 0o600)
            streams.append(os.fdopen(descriptor, 'w'))
            descriptors.pop()
        process = await asyncio.create_subprocess_exec(
            *worker_command(launch),
            cwd=working_directory,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=streams[1],
        )
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        for stream in streams:
            stream.close()
        raise

    try:
        identity = record(psutil.Process(process.pid), launch.owner_token)
    except Exception as error:
        identity = {
            'pid': process.pid, 'created': None, 'cmdline': [],
            'owner_token': launch.owner_token, 'owner_root': str(worker_root),
            'ownership_error': f'{type(error).__name__}: {error}',
        }
    return WorkerHandle(
        process=process, stdout_log=streams[0], stderr_log=streams[1],
        worker_identity=identity, initial_owned=None,
        owner_token=launch.owner_token, mode=launch.mode,
        state=transition_worker_state(WorkerState.CREATED, WorkerEvent.START),
    )


async def start_worker(handle: WorkerHandle, timeout=90):
    if handle.state is not WorkerState.STARTING:
        raise ValueError(f'worker cannot start from state {handle.state.value}')
    identity_errors = profile_identity_errors(handle.worker_identity, state_required=False)
    if identity_errors:
        handle.state = transition_worker_state(handle.state, WorkerEvent.START_FAILED)
        return {'ok': False, 'error': f'worker identity is incomplete: {identity_errors!r}'}, identity_errors
    try:
        response = await send(handle.process, {'op': 'start'}, handle.stdout_log, timeout)
    except Exception:
        handle.state = transition_worker_state(handle.state, WorkerEvent.START_FAILED)
        raise
    errors = []
    if not isinstance(response, dict) or response.get('ok') is not True:
        errors.append('worker start did not return an ok reply')
    result = response.get('result') if isinstance(response, dict) else None
    owned = result.get('owned') if isinstance(result, dict) else None
    if not isinstance(owned, list):
        errors.append('worker start reply has no owned-process inventory')
        owned = None
    if owned is not None:
        for index, identity in enumerate(owned):
            identity_errors = profile_identity_errors(identity, state_required=False)
            if identity_errors:
                errors.append(f'worker start identity {index} is invalid: {identity_errors!r}')
    if errors:
        handle.initial_owned = None
        handle.state = transition_worker_state(handle.state, WorkerEvent.START_FAILED)
    else:
        handle.initial_owned = tuple(owned)
        handle.state = transition_worker_state(handle.state, WorkerEvent.START_SUCCEEDED)
    return response, errors


async def send_operation(handle: WorkerHandle, request, timeout=90):
    if handle.state is not WorkerState.READY:
        raise ValueError(f'worker operation is not allowed in state {handle.state.value}')
    if not isinstance(request, dict) or request.get('op') != _MODE_OPERATION[handle.mode]:
        handle.state = transition_worker_state(handle.state, WorkerEvent.INVALID_OPERATION)
        raise ValueError(f'worker operation is not allowed in mode {handle.mode.value}')

    handle.state = transition_worker_state(handle.state, WorkerEvent.VALID_OPERATION)
    try:
        response = await send(handle.process, request, handle.stdout_log, timeout)
    except Exception:
        handle.state = transition_worker_state(handle.state, WorkerEvent.OPERATION_FAILED)
        raise
    if not isinstance(response, dict) or response.get('ok') is not True:
        handle.state = transition_worker_state(handle.state, WorkerEvent.OPERATION_FAILED)
        return response
    result = response.get('result')
    if not isinstance(result, dict):
        handle.state = transition_worker_state(handle.state, WorkerEvent.OPERATION_FAILED)
        raise ValueError('worker operation reply has no result object')
    if handle.mode is WorkerMode.NAVIGATION:
        required = ('arm', 'kind', 'index', 'nonce', 'url', 'outcome',
                    'start_ns', 'end_ns', 'elapsed_ns')
        if any(key not in result for key in required) \
                or result.get('outcome') not in ('success', 'failure') \
                or type(result.get('index')) is not int \
                or any(type(result.get(key)) is not int or result[key] < 0
                       for key in ('start_ns', 'end_ns', 'elapsed_ns')):
            handle.state = transition_worker_state(handle.state, WorkerEvent.OPERATION_FAILED)
            raise ValueError('navigation operation reply is not a complete sample')
        handle.state = transition_worker_state(handle.state, WorkerEvent.NAVIGATION_RESULT)
    else:
        handle.state = transition_worker_state(handle.state, WorkerEvent.ONE_SHOT_RESULT)
    return response


def parser():
    p = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    p.add_argument('--source', default=os.getenv('NAV_BENCH_SOURCE'))
    p.add_argument('--output')
    p.add_argument('--chrome', default=os.getenv('NAV_BENCH_CHROME'))
    p.add_argument('--lightpanda', default=os.getenv('NAV_BENCH_LIGHTPANDA'))
    p.add_argument('--agent-browser', default=os.getenv('NAV_BENCH_AGENT_BROWSER'))
    p.add_argument('--browser-use-python', default=os.getenv('NAV_BENCH_BROWSER_USE_PYTHON'))
    p.add_argument('--openbrowser-python', default=os.getenv('NAV_BENCH_OPENBROWSER_PYTHON', sys.executable))
    p.add_argument('--version-manifest', help='Optional JSON release-discovery evidence, copied into results')
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--delay', type=float, default=.15)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--verify', metavar='RESULT_DIRECTORY', help='Verify saved hashes and result contracts without browsers')
    p.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    p.add_argument('--worker-mode', choices=tuple(mode.value for mode in WorkerMode), help=argparse.SUPPRESS)
    p.add_argument('--run-root', help=argparse.SUPPRESS)
    return p


def validate_config(args):
    if args.samples < 1 or not math.isfinite(args.delay) or args.delay < 0:
        raise ValueError('samples must be positive and delay must be finite and non-negative')
    if len(args.arms) != len(set(args.arms)):
        raise ValueError('duplicate arms are not allowed')
    if not args.source or not (Path(args.source) / 'src/openbrowser/__init__.py').is_file():
        raise ValueError('--source must identify the source repository')
    if not args.output:
        raise ValueError('--output must identify a new result root')
    physical_root, output = physical_profile_path(args.output)
    if output.parent != physical_root:
        raise ValueError('--output must be a direct child of physical /tmp')
    if output.exists() or output.is_symlink():
        raise ValueError('--output must be a new directory')
    args.source = str(Path(args.source).resolve())
    args.output = str(output)
    for key in ('chrome', 'lightpanda', 'agent_browser', 'browser_use_python', 'openbrowser_python'):
        value = getattr(args, key)
        if value:
            path = Path(value).expanduser().absolute()
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f'{key} is not an executable file: {path}')
            setattr(args, key, str(path))
    if args.version_manifest:
        json.loads(Path(args.version_manifest).read_text())


def availability(args):
    required = {'raw-cdp': args.chrome, 'lightpanda': args.lightpanda,
                'agent-browser': args.chrome and args.agent_browser,
                'browser-use': args.chrome and args.browser_use_python,
                'openbrowser': args.chrome and args.openbrowser_python}
    return {name: {'state': 'configured' if required[name] else 'unavailable',
                   'reason': None if required[name] else 'required executable or Python environment was not supplied'}
            for name in args.arms}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(source):
    source = Path(source)
    return {str(p.relative_to(source)): sha256(p) for p in sorted((source / 'src/openbrowser').rglob('*.py'))}


def executable_metadata(path):
    if not path:
        return None
    try:
        version = subprocess.check_output([path, '--version'], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        version = f'unavailable: {error}'
    return {'path': path, 'sha256': sha256(path), 'version': version}


def git_value(source, *args):
    try:
        return subprocess.check_output(['git', '-C', str(source), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return None


def require_profile_source_clean(source):
    status = git_value(source, 'status', '--porcelain', '--untracked-files=all', '--', 'src/openbrowser')
    if status is None:
        raise ValueError('cannot inspect profile source Git status')
    if status:
        raise ValueError('profile source has changes under src/openbrowser')


def load_average():
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


@dataclass(frozen=True)
class ProfileRequest:
    source: Path
    openbrowser_python: Path
    chrome: Path
    version_manifest: Path
    result_root: Path
    workload: Literal['fast', 'delayed']
    output: Path


def profile_parser():
    p = argparse.ArgumentParser(
        prog=f'{Path(sys.argv[0]).name} profile',
        description='Collect one workload-specific OpenBrowser cProfile report.',
    )
    p.add_argument('--source')
    p.add_argument('--openbrowser-python')
    p.add_argument('--chrome')
    p.add_argument('--version-manifest')
    p.add_argument('--workload', choices=('fast', 'delayed'))
    p.add_argument('--result-root')
    p.add_argument('--output')
    p.add_argument('--verify', metavar='PROFILE_DIRECTORY')
    return p


def physical_profile_path(value):
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if '..' in raw.parts:
        raise ValueError('profile path must not contain a .. segment')
    raw = Path(os.path.abspath(raw))
    alias_root = Path('/tmp')
    physical_root = alias_root.resolve(strict=True)
    if raw == alias_root or raw.is_relative_to(alias_root):
        physical = physical_root / raw.relative_to(alias_root)
    elif raw == physical_root or raw.is_relative_to(physical_root):
        physical = raw
    else:
        raise ValueError('profile path must be under physical /tmp')
    if physical == physical_root:
        raise ValueError('profile path must be below physical /tmp')
    cursor = physical_root
    for part in physical.relative_to(physical_root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f'profile path contains a symlink component: {cursor}')
    return physical_root, physical


def validate_profile_result_root(value):
    physical_root, result_root = physical_profile_path(value)
    if result_root.parent != physical_root:
        raise ValueError('--result-root must be a direct child of physical /tmp')
    try:
        info = result_root.lstat()
    except OSError as error:
        raise ValueError(f'--result-root must be an existing directory: {result_root}') from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('--result-root must be a directory with mode 0700')
    if info.st_uid != os.geteuid():
        raise ValueError('--result-root must be owned by the current user')
    return result_root


def profile_result_path(value, result_root, workload=None):
    root = validate_profile_result_root(result_root)
    try:
        physical_root, directory = physical_profile_path(value)
    except ValueError as error:
        if 'must be under physical /tmp' in str(error):
            raise ValueError('profile path is outside RESULT_ROOT') from error
        raise
    if physical_root != root.parent or not directory.is_relative_to(root):
        raise ValueError('profile path is outside RESULT_ROOT')
    parts = directory.relative_to(root).parts
    if len(parts) != 3 or parts[0] != 'profile' or parts[1] not in ('fast', 'delayed') \
            or (workload is not None and parts[1] != workload):
        raise ValueError('profile path must be RESULT_ROOT/profile/<workload>/<run-id>')
    for parent in (root / 'profile', root / 'profile' / parts[1]):
        try:
            info = parent.lstat()
        except OSError as error:
            raise ValueError(f'profile output parent must exist: {parent}') from error
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 \
                or info.st_uid != os.geteuid():
            raise ValueError(f'profile output parent must be a private directory: {parent}')
    return root, directory


def validate_profile_directory(value, result_root, workload=None, verify=False):
    _root, directory = profile_result_path(value, result_root, workload)
    if verify:
        try:
            info = directory.lstat()
        except OSError as error:
            raise ValueError(f'profile directory does not exist: {directory}') from error
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700 \
                or info.st_uid != os.geteuid():
            raise ValueError('profile directory must be a private directory with mode 0700')
    elif directory.exists() or directory.is_symlink():
        raise ValueError('profile output must be a new directory')
    return directory


def profile_manifest_errors(manifest):
    errors = []
    if not isinstance(manifest, dict):
        return ['version manifest must be a JSON object'], None, None, None, None
    timestamp = manifest.get('fetched_at_utc')
    try:
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            errors.append('fetched_at_utc must be an ISO-8601 UTC timestamp')
    except (AttributeError, TypeError, ValueError):
        errors.append('fetched_at_utc must be an ISO-8601 UTC timestamp')

    def string_map(value, name):
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not isinstance(item, str) or not item
            for key, item in value.items()
        ):
            errors.append(f'{name} must map string keys to non-empty string values')

    def version_record(parent, key, name):
        record_value = parent.get(key) if isinstance(parent, dict) else None
        if not isinstance(record_value, dict) or not isinstance(record_value.get('version'), str) \
                or not record_value['version'] or not isinstance(record_value.get('sha256'), str) \
                or len(record_value['sha256']) != 64 or any(char not in '0123456789abcdef' for char in record_value['sha256']):
            errors.append(f'{name} must contain a version and lowercase SHA-256')
            return None
        return record_value

    string_map(manifest.get('latest_versions'), 'latest_versions')
    string_map(manifest.get('installed_versions'), 'installed_versions')
    installed = manifest.get('installed_executables')
    python_interpreters = manifest.get('python_interpreters')
    chrome = version_record(installed, 'chrome_for_testing', 'installed_executables.chrome_for_testing')
    python = version_record(python_interpreters, 'candidate_openbrowser', 'python_interpreters.candidate_openbrowser')
    source = manifest.get('source')
    commit = source.get('candidate_commit') if isinstance(source, dict) else None
    if not isinstance(commit, str) or len(commit) != 40 or any(char not in '0123456789abcdefABCDEF' for char in commit):
        errors.append('source.candidate_commit must be a 40-character hexadecimal Git ID')
    harness = manifest.get('harness')
    harness_hash = harness.get('sha256') if isinstance(harness, dict) else None
    if not isinstance(harness_hash, str) or len(harness_hash) != 64 \
            or any(char not in '0123456789abcdef' for char in harness_hash):
        errors.append('harness.sha256 must be a lowercase SHA-256 digest')
    return errors, chrome, python, commit, harness_hash


def validate_profile_request(args):
    if args.workload not in ('fast', 'delayed'):
        raise ValueError('--workload must be fast or delayed')
    result_root = validate_profile_result_root(args.result_root)
    output = validate_profile_directory(args.output, result_root, workload=args.workload)

    if not isinstance(args.source, str) or not args.source.strip():
        raise ValueError('--source must identify the candidate source repository')
    source_value = Path(args.source).expanduser()
    if not source_value.is_dir():
        raise ValueError('--source must identify the candidate source repository')
    source = source_value.resolve()
    if not (source / 'src/openbrowser/__init__.py').is_file():
        raise ValueError('--source must identify a repository with src/openbrowser')
    require_profile_source_clean(source)

    executable_paths = {}
    for key in ('openbrowser_python', 'chrome'):
        supplied = Path(getattr(args, key)).expanduser().absolute()
        if not supplied.is_file() or not os.access(supplied, os.X_OK):
            raise ValueError(f'--{key.replace("_", "-")} must identify an executable file')
        executable_paths[key] = supplied

    manifest_path = Path(args.version_manifest).expanduser().absolute()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('--version-manifest must identify a regular JSON file')
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'cannot read version manifest: {error}') from error
    manifest_errors, chrome_record, python_record, candidate_commit, harness_hash = profile_manifest_errors(manifest)
    if manifest_errors:
        raise ValueError('invalid version manifest: ' + '; '.join(manifest_errors))
    if sha256(executable_paths['chrome']) != chrome_record['sha256']:
        raise ValueError('--chrome digest does not match installed_executables.chrome_for_testing')
    if sha256(executable_paths['openbrowser_python']) != python_record['sha256']:
        raise ValueError('--openbrowser-python digest does not match python_interpreters.candidate_openbrowser')
    actual_commit = git_value(source, 'rev-parse', 'HEAD')
    if actual_commit != candidate_commit:
        raise ValueError('--source commit does not match source.candidate_commit in the version manifest')
    if sha256(SCRIPT) != harness_hash:
        raise ValueError('--version-manifest harness.sha256 does not match navigation_compare.py')

    return ProfileRequest(
        source=source,
        openbrowser_python=executable_paths['openbrowser_python'],
        chrome=executable_paths['chrome'],
        version_manifest=manifest_path,
        result_root=result_root,
        workload=args.workload,
        output=output,
    )


def profile_cli(argv):
    p = profile_parser()
    args = p.parse_args(argv)
    run_fields = ('source', 'openbrowser_python', 'chrome', 'version_manifest', 'workload', 'output')
    if args.verify is not None:
        if any(getattr(args, name) is not None for name in run_fields):
            p.error('--verify cannot be combined with run options')
        if args.result_root is None:
            p.error('--verify requires --result-root')
        try:
            profile_result_path(args.verify, args.result_root)
        except (ValueError, OSError) as error:
            LOG.error('invalid profile directory: %s', error)
            return 2
        return verify_profile(args.verify, args.result_root)
    missing = [f'--{name.replace("_", "-")}' for name in run_fields if getattr(args, name) is None]
    if args.result_root is None:
        missing.append('--result-root')
    if missing:
        p.error('run form requires ' + ', '.join(missing))
    try:
        request = validate_profile_request(args)
    except (ValueError, OSError, KeyError) as error:
        LOG.error('%s', error)
        return 2
    return asyncio.run(run_profile(request))


def atomic_profile_bytes(path, content):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_profile_json(path, value):
    content = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    atomic_profile_bytes(path, content)


def copy_profile_manifest(source, destination):
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    try:
        shutil.copyfile(source, temporary)
        if temporary.is_symlink() or not temporary.is_file():
            raise OSError('copied version manifest is not a regular file')
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        destination.chmod(0o600)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def copy_profile_stats(source, destination):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_file():
        raise OSError('worker profile stats are not a regular file')
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    try:
        shutil.copyfile(source, temporary)
        if temporary.is_symlink() or not temporary.is_file() or temporary.stat().st_size <= 0:
            raise OSError('copied profile stats are not a non-empty regular file')
        temporary.chmod(0o600)
        os.replace(temporary, destination)
        destination.chmod(0o600)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def profile_hash_entries(directory):
    directory = Path(directory)
    hashes = {}
    for path in sorted(directory.iterdir()):
        if path.name == 'sha256.json':
            continue
        if path.is_symlink() or not path.is_file():
            raise OSError(f'profile artifact is not a regular file: {path.name}')
        path.chmod(0o600)
        hashes[path.name] = sha256(path)
    return hashes


def _profile_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in '0123456789abcdef' for char in value)


def _profile_version_map(value):
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) and item for key, item in value.items()
    )


def profile_pair_errors(warmup, target, workload):
    errors = []
    records = []
    for label, record_value, expected_kind in (('warmup', warmup, 'fast'), ('target', target, workload)):
        if not isinstance(record_value, dict) or type(record_value.get('expected')) is not int \
                or record_value.get('expected') != 1 or type(record_value.get('attempted')) is not int \
                or record_value.get('attempted') != 1 or record_value.get('outcome') != 'success' \
                or not isinstance(record_value.get('sample'), dict):
            errors.append(f'{label}: expected one successful sample')
            continue
        sample = record_value['sample']
        if sample.get('arm') != 'openbrowser' or sample.get('kind') != expected_kind \
                or type(sample.get('index')) is not int or sample.get('index') != 0:
            errors.append(f'{label}: sample does not match the selected OpenBrowser workload')
        try:
            errors.extend(f'{label}: {error}' for error in trial_proof_errors(sample, label))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f'{label}: invalid sample proof: {error}')
        records.append(record_value)
    if len(records) == 2 and records[0]['sample'].get('nonce') == records[1]['sample'].get('nonce'):
        errors.append('warmup and target nonces must differ')
    return errors


def recover_profile_progress(run_root):
    untouched = {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    progress_path = Path(run_root) / 'profile-progress.json'
    try:
        if progress_path.is_symlink() or not progress_path.is_file():
            raise ValueError('profile progress file is unavailable or not regular')
        progress = json.loads(progress_path.read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return untouched, dict(untouched), 'profile', f'profile progress could not be recovered: {error}'

    phases = {'ready', 'warmup', 'target', 'profile', 'complete'}
    expected_state_keys = {'attempted', 'outcome'}
    if not isinstance(progress, dict) or set(progress) != {'schema', 'phase', 'warmup', 'target'} \
            or progress.get('schema') != 'openbrowser-navigation-progress/1' \
            or not isinstance(progress.get('phase'), str) or progress['phase'] not in phases:
        return untouched, dict(untouched), 'profile', 'profile progress object does not match its schema'

    states = {}
    for label in ('warmup', 'target'):
        state_value = progress.get(label)
        if not isinstance(state_value, dict) or set(state_value) != expected_state_keys \
                or type(state_value.get('attempted')) is not int or state_value['attempted'] not in (0, 1) \
                or state_value.get('outcome') not in ('not_run', 'in_progress', 'success', 'failure'):
            return untouched, dict(untouched), 'profile', f'profile progress {label} state is invalid'
        if state_value['attempted'] == 0 and state_value['outcome'] != 'not_run' \
                or state_value['attempted'] == 1 and state_value['outcome'] == 'not_run':
            return untouched, dict(untouched), 'profile', f'profile progress {label} attempt state is inconsistent'
        states[label] = state_value

    warmup_state, target_state = states['warmup'], states['target']
    phase = progress['phase']
    valid_phase = (
        phase == 'ready' and warmup_state['attempted'] == 0 and target_state['attempted'] == 0
        or phase == 'warmup' and warmup_state['attempted'] == 1 and target_state['attempted'] == 0
        or phase == 'target' and warmup_state['outcome'] == 'success'
        and target_state == {'attempted': 1, 'outcome': 'in_progress'}
        or phase == 'profile' and warmup_state['outcome'] == 'success'
        and target_state['outcome'] != 'in_progress'
        or phase == 'complete' and warmup_state['attempted'] == 1
        and warmup_state['outcome'] != 'in_progress' and target_state['outcome'] != 'in_progress'
    )
    if not valid_phase or target_state['attempted'] == 1 and warmup_state['outcome'] != 'success':
        return untouched, dict(untouched), 'profile', 'profile progress phase and trial states are inconsistent'

    def report_record(state_value):
        outcome = 'failure' if state_value['outcome'] == 'in_progress' else state_value['outcome']
        return {
            'expected': 1,
            'attempted': state_value['attempted'],
            'outcome': outcome,
            'sample': None,
        }

    warmup = report_record(warmup_state)
    target = report_record(target_state)
    if warmup_state['outcome'] in ('in_progress', 'failure'):
        failed_phase = 'warmup'
    elif target_state['outcome'] in ('in_progress', 'failure'):
        failed_phase = 'target'
    else:
        failed_phase = 'profile'
    return warmup, target, failed_phase, 'profile_pair response was lost; recovered the last atomic progress state'


def profile_report_errors(report):
    if not isinstance(report, dict):
        return ['profile report must be a JSON object']
    errors = []
    phases = {
        'startup', 'warmup', 'target', 'profile', 'source', 'worker_stop', 'cleanup', 'copy',
        'artifact_verify',
    }

    def exact_keys(value, expected, label):
        if isinstance(value, dict) and set(value) != expected:
            errors.append(f'{label} keys do not match the documented profile schema')

    def valid_source_path(name):
        if not isinstance(name, str) or not name or '\\' in name or '\x00' in name:
            return False
        path = PurePosixPath(name)
        parts = path.parts
        return (path.as_posix() == name and not path.is_absolute()
                and len(parts) > 2 and parts[:2] == ('src', 'openbrowser')
                and all(part not in ('.', '..') for part in parts)
                and path.suffix == '.py')

    exact_keys(report, {
        'schema', 'status', 'failed_phase', 'arm', 'workload', 'delay_seconds', 'source', 'harness',
        'version_manifest_file', 'version_manifest_sha256', 'tools', 'warmup', 'target', 'profile',
        'cleanup', 'errors', 'verification_errors',
    }, 'profile report')
    if report.get('schema') != 'openbrowser-navigation-profile/1':
        errors.append('profile report schema is invalid')
    if report.get('status') not in ('complete', 'failed'):
        errors.append('profile report status is invalid')
    if report.get('arm') != 'openbrowser' or report.get('workload') not in ('fast', 'delayed'):
        errors.append('profile report arm or workload is invalid')
    if report.get('delay_seconds') != .15:
        errors.append('profile report delay is not 0.15 seconds')
    if report.get('version_manifest_file') != 'version-manifest.json' or not _profile_digest(
        report.get('version_manifest_sha256')
    ):
        errors.append('profile report version manifest identity is invalid')
    if any(key in report for key in ('summary', 'median_ns', 'p50', 'speedup', 'paired_saving_ns')):
        errors.append('profile report must not contain benchmark results')
    source = report.get('source')
    if not isinstance(source, dict):
        errors.append('profile report source identity is missing')
        source = {}
    exact_keys(source, {'commit', 'manifest_start', 'manifest_end', 'unchanged'}, 'profile report source')
    commit = source.get('commit')
    if not isinstance(commit, str) or len(commit) != 40 or any(
        char not in '0123456789abcdefABCDEF' for char in commit
    ):
        errors.append('profile report source commit is invalid')
    for field in ('manifest_start', 'manifest_end'):
        value = source.get(field)
        if not isinstance(value, dict) or any(
            not valid_source_path(name) or not _profile_digest(digest)
            for name, digest in value.items()
        ):
            errors.append(f'profile report source.{field} is invalid')
        if report.get('status') == 'complete' and not value:
            errors.append(f'complete profile report source.{field} cannot be empty')
    if source.get('unchanged') is not True or source.get('manifest_start') != source.get('manifest_end'):
        errors.append('profile report source changed during the run')
    harness = report.get('harness')
    if not isinstance(harness, dict) or not _profile_digest(harness.get('sha256')):
        errors.append('profile report harness digest is invalid')
    exact_keys(harness, {'sha256'}, 'profile report harness')

    tools = report.get('tools')
    if not isinstance(tools, dict):
        errors.append('profile report tool identities are missing')
        tools = {}
    exact_keys(tools, {'openbrowser_python', 'chrome', 'browser_runtime', 'packages'}, 'profile report tools')
    for key in ('openbrowser_python', 'chrome'):
        tool = tools.get(key)
        exact_keys(tool, {'path', 'version', 'sha256'}, f'profile report tools.{key}')
        if not isinstance(tool, dict) or not isinstance(tool.get('path'), str) or not Path(tool['path']).is_absolute() \
                or not isinstance(tool.get('version'), str) or not tool['version'] or not _profile_digest(tool.get('sha256')):
            errors.append(f'profile report tools.{key} is invalid')
    runtime = tools.get('browser_runtime')
    if not isinstance(runtime, dict) or not isinstance(runtime.get('product'), str) or not runtime['product']:
        errors.append('profile report browser runtime product is missing')
    packages = tools.get('packages')
    exact_keys(packages, {'packages', 'loaded_packages'}, 'profile report tools.packages')
    if not isinstance(packages, dict) or not _profile_version_map(packages.get('packages')) \
            or not _profile_version_map(packages.get('loaded_packages')):
        errors.append('profile report package versions are invalid')

    workload = report.get('workload')
    trial_records = []
    for label, expected_kind in (('warmup', 'fast'), ('target', workload)):
        record_value = report.get(label)
        exact_keys(record_value, {'expected', 'attempted', 'outcome', 'sample'}, f'profile report {label}')
        if not isinstance(record_value, dict) or type(record_value.get('expected')) is not int \
                or record_value.get('expected') != 1 or type(record_value.get('attempted')) is not int \
                or record_value.get('attempted') != 1 or record_value.get('outcome') != 'success' \
                or not isinstance(record_value.get('sample'), dict):
            errors.append(f'profile report {label} must contain exactly one successful sample')
            continue
        sample = record_value['sample']
        if sample.get('outcome') != 'success':
            errors.append(f'profile report {label} sample outcome must be success')
        if sample.get('arm') != 'openbrowser' or sample.get('kind') != expected_kind \
                or type(sample.get('index')) is not int or sample.get('index') != 0:
            errors.append(f'profile report {label} sample does not match its requested workload')
        nonce = sample.get('nonce')
        url = sample.get('url')
        try:
            parsed_url = urlsplit(url) if isinstance(url, str) else None
        except ValueError:
            parsed_url = None
        query = parse_qs(parsed_url.query) if parsed_url is not None else {}
        expected_delayed = '1' if expected_kind == 'delayed' else '0'
        if not isinstance(nonce, str) or not nonce or parsed_url is None \
                or parsed_url.scheme != 'http' or parsed_url.hostname != '127.0.0.1' \
                or parsed_url.path != '/fixture' or query.get('nonce') != [nonce] \
                or query.get('delayed') != [expected_delayed]:
            errors.append(f'profile report {label} URL is not the selected local fixture task')
        try:
            errors.extend(trial_proof_errors(sample, label))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f'profile report {label} sample evidence is invalid: {error}')
        trial_records.append(record_value)
    if len(trial_records) == 2:
        if trial_records[0]['sample'].get('nonce') == trial_records[1]['sample'].get('nonce'):
            errors.append('profile report warmup and target nonces must differ')

    profile = report.get('profile')
    exact_keys(profile, {'file', 'bytes', 'sha256'}, 'profile report profile')
    if report.get('status') == 'complete':
        if not isinstance(profile, dict) or profile.get('file') != 'trial.pstats' \
                or type(profile.get('bytes')) is not int or profile['bytes'] <= 0 \
                or not _profile_digest(profile.get('sha256')):
            errors.append('complete profile stats metadata is invalid')
        if report.get('failed_phase') is not None:
            errors.append('complete profile report cannot have a failed phase')
        if report.get('errors') != [] or report.get('verification_errors') != []:
            errors.append('complete profile report cannot contain errors')
    else:
        failed_phase = report.get('failed_phase')
        if not isinstance(failed_phase, str) or failed_phase not in phases:
            errors.append('failed profile report must name its first failing phase')
        if profile is not None:
            errors.append('failed profile report cannot contain stats metadata')
        if not isinstance(report.get('errors'), list) or not report['errors']:
            errors.append('failed profile report must contain operational errors')
        if not isinstance(report.get('verification_errors'), list) or not report['verification_errors']:
            errors.append('failed profile report must contain verification errors')

    cleanup = report.get('cleanup')
    if not isinstance(cleanup, dict):
        errors.append('profile report cleanup evidence is missing')
        cleanup = {}
    exact_keys(cleanup, {
        'worker_identity', 'post_worker_exit_identities', 'ok', 'root_removed', 'fixture_thread_stopped',
    }, 'profile report cleanup')
    worker = cleanup.get('worker_identity')
    post_identities = cleanup.get('post_worker_exit_identities')
    if not isinstance(post_identities, list):
        errors.append('profile report cleanup post-worker-exit identities must be a list')
        post_identities = []
    valid_identity_states = ('absent', 'reused', 'alive', 'unknown', 'zombie')
    for label, identity in [('worker', worker), *[('post-worker-exit', item) for item in post_identities]]:
        state_value = identity.get('state') if isinstance(identity, dict) else None
        if not isinstance(state_value, str) or state_value not in valid_identity_states:
            errors.append(f'profile report cleanup {label} identity is invalid')
        elif report.get('status') == 'complete' and state_value not in ('absent', 'reused'):
            errors.append('complete profile cleanup identities are not retired')
        if report.get('status') == 'complete':
            errors.extend(
                f'profile report cleanup {label} identity {error}'
                for error in profile_identity_errors(identity)
            )
    if report.get('status') == 'complete':
        if cleanup.get('ok') is not True or cleanup.get('root_removed') is not True \
                or cleanup.get('fixture_thread_stopped') is not True:
            errors.append('complete profile cleanup proof is incomplete')
    return errors


def verify_profile(directory, result_root):
    try:
        result_root = validate_profile_result_root(result_root)
        output = validate_profile_directory(directory, result_root, verify=True)
    except (OSError, ValueError) as error:
        LOG.error('invalid profile directory: %s', error)
        return 1

    errors = []
    required_files = {'profile.json', 'version-manifest.json', 'worker.stdout.log', 'worker.stderr.log', 'trial.pstats'}
    expected_names = required_files | {'sha256.json'}
    try:
        names = {path.name for path in output.iterdir()}
        if names != expected_names:
            errors.append('profile directory contains missing or unexpected files')
        for path in output.iterdir():
            if path.is_symlink() or not path.is_file():
                errors.append(f'profile artifact is not a regular file: {path.name}')
            elif stat.S_IMODE(path.stat().st_mode) != 0o600:
                errors.append(f'profile artifact mode is not 0600: {path.name}')
        if errors:
            raise ValueError('invalid profile artifact directory')
        report = json.loads((output / 'profile.json').read_text())
        manifest = json.loads((output / 'version-manifest.json').read_text())
        hashes = json.loads((output / 'sha256.json').read_text())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(f'cannot read profile artifacts: {error}')
        for error in errors:
            LOG.error('%s', error)
        return 1

    if not isinstance(hashes, dict):
        errors.append('sha256.json must be a JSON object')
        hashes = {}
    if 'sha256.json' in hashes:
        errors.append('sha256.json must not hash itself')
    if set(hashes) != required_files:
        errors.append('sha256.json must list every required profile artifact and no others')
    for name, expected in hashes.items():
        if not isinstance(name, str) or not name or '\x00' in name or Path(name).name != name \
                or name in ('.', '..') or '/' in name or '\\' in name or Path(name).is_absolute():
            errors.append(f'hash entry is not a basename: {name!r}')
            continue
        if not _profile_digest(expected):
            errors.append(f'hash entry is not a lowercase SHA-256 digest: {name}')
            continue
        artifact = output / name
        try:
            info = artifact.lstat()
            if not stat.S_ISREG(info.st_mode) or artifact.is_symlink():
                errors.append(f'hash entry is not a regular artifact: {name}')
            elif sha256(artifact) != expected:
                errors.append(f'file hash mismatch: {name}')
        except OSError:
            errors.append(f'missing profile artifact: {name}')

    if isinstance(report, dict) and report.get('status') != 'complete':
        errors.append('saved profile report is not complete')
    errors.extend(profile_report_errors(report))
    manifest_errors, chrome_record, python_record, candidate_commit, harness_hash = profile_manifest_errors(manifest)
    errors.extend(f'version manifest: {error}' for error in manifest_errors)
    if isinstance(report, dict):
        tools = report.get('tools', {})
        if isinstance(tools, dict):
            chrome = tools.get('chrome', {})
            python = tools.get('openbrowser_python', {})
            source = report.get('source', {})
            harness = report.get('harness', {})
            if isinstance(chrome_record, dict) and isinstance(chrome, dict):
                if chrome.get('version') != chrome_record.get('version') \
                        or chrome.get('sha256') != chrome_record.get('sha256'):
                    errors.append('profile Chrome identity does not match the copied version manifest')
            if isinstance(python_record, dict) and isinstance(python, dict):
                if python.get('version') != python_record.get('version') \
                        or python.get('sha256') != python_record.get('sha256'):
                    errors.append('profile Python identity does not match the copied version manifest')
            if isinstance(source, dict) and source.get('commit') != candidate_commit:
                errors.append('profile candidate commit does not match the copied version manifest')
            if isinstance(harness, dict) and harness.get('sha256') != harness_hash:
                errors.append('profile harness digest does not match the copied version manifest')
            manifest_sha = hashes.get('version-manifest.json')
            if report.get('version_manifest_sha256') != manifest_sha:
                errors.append('profile copied manifest digest does not match sha256.json')
            profile = report.get('profile')
            if isinstance(profile, dict):
                stats = output / 'trial.pstats'
                try:
                    if profile.get('bytes') != stats.stat().st_size:
                        errors.append('profile stats byte count does not match trial.pstats')
                    if profile.get('sha256') != hashes.get('trial.pstats'):
                        errors.append('profile stats digest does not match sha256.json')
                    import pstats

                    parsed_stats = pstats.Stats(str(stats))
                    if not parsed_stats.stats:
                        errors.append('trial.pstats contains no call-path data')
                except (OSError, TypeError, ValueError, EOFError) as error:
                    errors.append(f'trial.pstats is invalid: {error}')
    for error in errors:
        LOG.error('%s', error)
    if not errors:
        LOG.info('verified profile report, copied manifest, task proof, stats, hashes, source stability, and cleanup')
    return 0 if not errors else 1


def remove_new_profile_output(directory, identity):
    directory = Path(directory)
    try:
        current = directory.lstat()
        return (
            stat.S_ISDIR(current.st_mode)
            and not directory.is_symlink()
            and (current.st_dev, current.st_ino) == identity
            and stat.S_IMODE(current.st_mode) == 0o700
            and (shutil.rmtree(directory) is None)
        )
    except OSError:
        return False


async def run_profile(request):
    try:
        destination = validate_profile_directory(
            request.output, request.result_root, workload=request.workload,
        )
        source = Path(request.source).expanduser().resolve()
        python = Path(request.openbrowser_python).expanduser().absolute()
        chrome = Path(request.chrome).expanduser().absolute()
        manifest_path = Path(request.version_manifest).expanduser().absolute()
        if not source.is_dir() or not (source / 'src/openbrowser/__init__.py').is_file():
            raise ValueError('profile source no longer identifies the candidate source repository')
        require_profile_source_clean(source)
        if not python.is_file() or not os.access(python, os.X_OK):
            raise ValueError('profile OpenBrowser Python interpreter is unavailable')
        if not chrome.is_file() or not os.access(chrome, os.X_OK):
            raise ValueError('profile Chrome executable is unavailable')
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError('profile version manifest is unavailable')
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_errors, chrome_record, python_record, candidate_commit, harness_hash = profile_manifest_errors(manifest)
        if manifest_errors:
            raise ValueError('invalid version manifest: ' + '; '.join(manifest_errors))
        if sha256(chrome) != chrome_record['sha256'] or sha256(python) != python_record['sha256']:
            raise ValueError('profile executable digest does not match the version manifest')
        source_commit = git_value(source, 'rev-parse', 'HEAD')
        if source_commit != candidate_commit:
            raise ValueError('profile source commit does not match the version manifest')
        harness_start = sha256(SCRIPT)
        if harness_start != harness_hash:
            raise ValueError('profile harness digest does not match the version manifest')
        if request.workload not in ('fast', 'delayed'):
            raise ValueError('profile workload is invalid')
        source_start = source_manifest(source)
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        LOG.error('%s', error)
        return 2

    try:
        destination.mkdir(mode=0o700, exist_ok=False)
        destination.chmod(0o700)
        output_stat = destination.lstat()
        output_identity = (output_stat.st_dev, output_stat.st_ino)
    except OSError as error:
        LOG.error('cannot create profile output: %s', error)
        return 2

    try:
        copied_manifest = destination / 'version-manifest.json'
        copy_profile_manifest(manifest_path, copied_manifest)
        if copied_manifest.read_bytes() != manifest_bytes:
            raise OSError('copied version manifest differs from the validated input bytes')
    except (OSError, ValueError) as error:
        LOG.error('cannot copy version manifest before worker startup: %s', error)
        remove_new_profile_output(destination, output_identity)
        return 1

    failed_phase = None
    errors = []
    runtime_root = None
    run_root = None
    token = None
    fixture = None
    fixture_thread = None
    fixture_thread_started = False
    fixture_thread_stopped = False
    handle = None
    proc = None
    stdout_log = None
    stderr_log = None
    worker_identity = None
    worker_identity_complete = False
    cleanup_response = None
    cleanup_ok = False
    root_removed = False
    started = None
    pair_result = None
    browser_runtime = None
    package_data = {'packages': {}, 'loaded_packages': {}}
    warmup = {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    target = {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    profile_state = {'enabled': False, 'disabled': False, 'dumped': False, 'file': 'trial.pstats'}
    profile_metadata = None
    source_end = {}
    source_unchanged = False
    harness_unchanged = False
    source_commit_before = source_commit
    initial_identities: tuple[dict, ...] | None = None
    fixture_created = False

    def fail(phase, message):
        nonlocal failed_phase
        if failed_phase is None:
            failed_phase = phase
        if message:
            errors.append(str(message))

    def discard_profile_stats():
        nonlocal profile_metadata
        try:
            (destination / 'trial.pstats').unlink(missing_ok=True)
        except OSError as error:
            errors.append(f'cannot remove copied profile stats: {error}')
        profile_metadata = None

    def fail_lost_pair(message):
        nonlocal warmup, target
        warmup, target, phase, progress_message = recover_profile_progress(run_root)
        fail(phase, f'{message}; {progress_message}')

    try:
        physical_root = Path('/tmp').resolve(strict=True)
        runtime_root = Path(tempfile.mkdtemp(prefix='nb-profile-', dir=str(physical_root)))
        runtime_root.chmod(0o700)
        token = uuid.uuid4().hex
        atomic_profile_bytes(runtime_root / '.benchmark-owner', token.encode())
        run_root = runtime_root / 'openbrowser'

        fixture = Fixture(.15)
        fixture_created = True
        fixture_thread = threading.Thread(target=fixture.serve_forever, daemon=True)
        fixture_thread.start()
        fixture_thread_started = True

        launch = make_worker_launch(
            arm='openbrowser', mode=WorkerMode.PROFILE, source_root=source,
            candidate_python=python, browser_use_python=python,
            runtime_root=runtime_root, worker_root=run_root,
            working_directory=run_root, log_directory=destination,
            log_prefix='worker', owner_token=token, chrome=chrome,
        )
        handle = await launch_worker(launch)
        proc = handle.process
        stdout_log = handle.stdout_log
        stderr_log = handle.stderr_log
        worker_identity = handle.worker_identity
        identity_errors = profile_identity_errors(worker_identity, state_required=False)
        worker_identity_complete = not identity_errors
        if identity_errors:
            fail('startup', f'profile worker identity capture is incomplete: {identity_errors!r}')

        try:
            response, start_errors = await start_worker(handle)
            if start_errors:
                fail('startup', f'OpenBrowser worker startup contract is incomplete: {start_errors!r}')
            if not response.get('ok'):
                fail('startup', response.get('error', 'OpenBrowser worker startup failed'))
            elif not start_errors:
                started = response.get('result')
                if not isinstance(started, dict):
                    fail('startup', 'OpenBrowser worker returned an invalid startup result')
                else:
                    initial_identities = handle.initial_owned
                    browser_runtime = started.get('version')
                    package_data = {
                        'packages': started.get('packages', {}),
                        'loaded_packages': started.get('loaded_packages', {}),
                    }
        except Exception as error:
            fail('startup', f'OpenBrowser worker startup failed: {type(error).__name__}: {error}')

        if failed_phase is None:
            warmup_spec = {'arm': 'openbrowser', 'kind': 'fast', 'index': 0, 'nonce': f'warmup-{uuid.uuid4().hex}'}
            target_spec = {
                'arm': 'openbrowser', 'kind': request.workload, 'index': 0,
                'nonce': f'target-{uuid.uuid4().hex}',
            }
            pair_request = {
                'op': 'profile_pair',
                'workload': request.workload,
                'fixture_port': fixture.server_address[1],
                'warmup': warmup_spec,
                'target': target_spec,
            }
            try:
                response = await send_operation(handle, pair_request)
                if not response.get('ok'):
                    fail('profile', response.get('error', 'OpenBrowser worker rejected profile_pair'))
                else:
                    pair_result = response.get('result')
                    if not isinstance(pair_result, dict):
                        fail('profile', 'OpenBrowser worker returned an invalid profile_pair result')
                    else:
                        warmup = pair_result.get('warmup', warmup)
                        target = pair_result.get('target', target)
                        profile_state = pair_result.get('profile', profile_state)
                        browser_runtime = pair_result.get('browser_runtime', browser_runtime)
                        package_data = pair_result.get('packages', package_data)
                        worker_errors = pair_result.get('errors', [])
                        if isinstance(worker_errors, list):
                            errors.extend(str(error) for error in worker_errors if error)
                        worker_phase = pair_result.get('failed_phase')
                        if worker_phase:
                            fail(worker_phase, None)
                        if not isinstance(profile_state, dict):
                            profile_state = {'enabled': False, 'disabled': False, 'dumped': False, 'file': 'trial.pstats'}
                            fail('profile', 'worker profile state is invalid')
            except Exception as error:
                message = f'profile_pair failed: {type(error).__name__}: {error}'
                if isinstance(error, TimeoutError) or (
                    isinstance(error, RuntimeError) and str(error) == 'worker stdout closed without reply'
                ):
                    fail_lost_pair(message)
                else:
                    fail('profile', message)
    except Exception as error:
        fail('startup', f'profile startup failed: {type(error).__name__}: {error}')
    finally:
        if handle is not None:
            try:
                cleanup_response = await cleanup_worker(handle)
            except Exception as error:
                cleanup_response = {
                    'ok': False,
                    'error': f'{type(error).__name__}: {error}',
                    'worker_identity': {**(worker_identity or {}), 'state': 'unknown'},
                    'post_worker_exit_identities': [],
                }
        if fixture_created:
            try:
                if fixture_thread_started:
                    fixture.shutdown()
                fixture.server_close()
            except Exception as error:
                fail('cleanup', f'fixture shutdown failed: {type(error).__name__}: {error}')
            if fixture_thread is not None and fixture_thread_started:
                fixture_thread.join(timeout=5)
                fixture_thread_stopped = not fixture_thread.is_alive()
            else:
                fixture_thread_stopped = True
            if not fixture_thread_stopped:
                fail('cleanup', 'fixture thread did not stop')
        if stdout_log is not None:
            stdout_log.close()
        if stderr_log is not None:
            stderr_log.close()

    cleanup_response = cleanup_response or {}
    cleanup_worker_identity = cleanup_response.get('worker_identity')
    if not isinstance(cleanup_worker_identity, dict):
        cleanup_worker_identity = {**(worker_identity or {}), 'state': 'unknown'}
        fail('cleanup', 'worker cleanup returned invalid worker identity')
    valid_identity_states = ('absent', 'reused', 'alive', 'unknown', 'zombie')
    worker_identity_errors = profile_identity_errors(cleanup_worker_identity)
    if cleanup_worker_identity.get('state') not in valid_identity_states:
        worker_identity_errors.append('process state is invalid')
    if worker_identity_errors:
        cleanup_worker_identity = {**cleanup_worker_identity, 'state': 'unknown'}
        fail('cleanup', f'worker cleanup identity is incomplete: {worker_identity_errors!r}')

    post_identities = cleanup_response.get('post_worker_exit_identities')
    post_identities_valid = isinstance(post_identities, list)
    if not post_identities_valid:
        post_identities = [{'state': 'unknown'}]
        fail('cleanup', 'worker cleanup returned invalid post-exit identities')
    else:
        validated_identities = []
        for index, identity in enumerate(post_identities):
            identity_errors = profile_identity_errors(identity)
            state_value = identity.get('state') if isinstance(identity, dict) else None
            if state_value not in valid_identity_states:
                identity_errors.append('process state is invalid')
            if identity_errors:
                identity = {**identity, 'state': 'unknown'} if isinstance(identity, dict) else {'state': 'unknown'}
                fail('cleanup', f'post-worker-exit identity {index} is incomplete: {identity_errors!r}')
            validated_identities.append(identity)
        post_identities = validated_identities

    identity_states = [cleanup_worker_identity.get('state', 'unknown')]
    identity_states.extend(
        identity.get('state', 'unknown') if isinstance(identity, dict) else 'unknown'
        for identity in post_identities
    )
    identities_safe = all(state_value in ('absent', 'reused') for state_value in identity_states)
    if proc is not None and initial_identities is None:
        fail('cleanup', 'profile worker startup ownership inventory was not received or validated')
    cleanup_ok = cleanup_response.get('ok') is True and identities_safe \
        and post_identities_valid and (proc is None or worker_identity_complete and initial_identities is not None)
    cleanup_result = cleanup_response.get('result')
    stop_errors = cleanup_result.get('errors', []) if isinstance(cleanup_result, dict) else []
    if proc is not None and stop_errors:
        fail('worker_stop', f'profile worker stop reported errors: {stop_errors!r}')
    if proc is not None and cleanup_response.get('ok') is not True:
        fail('worker_stop', cleanup_response.get('error', 'profile worker did not stop cleanly'))
    if proc is not None and not identities_safe:
        fail('cleanup', 'profile worker or owned browser cleanup identity is live or unknown')
    if proc is not None and not worker_identity_complete:
        fail('cleanup', 'profile worker identity capture is incomplete')

    if source_end == {}:
        try:
            source_end = source_manifest(source)
            source_unchanged = source_start == source_end
            harness_unchanged = sha256(SCRIPT) == harness_start
            if not source_unchanged or not harness_unchanged:
                fail('source', 'source or harness changed during the profile run')
        except Exception as error:
            fail('source', f'cannot verify source stability: {type(error).__name__}: {error}')

    for label, record_value in (('warmup', warmup), ('target', target)):
        if not isinstance(record_value, dict):
            fail('warmup' if label == 'warmup' else 'target', f'{label} trial record is invalid')
            continue
        sample = record_value.get('sample')
        if not isinstance(sample, dict):
            continue
        if fixture is None:
            fail('cleanup', 'fixture ledger is unavailable for post-run validation')
            continue
        try:
            sample['server'] = fixture.snapshot(sample.get('nonce', ''))
            proof_errors = trial_proof_errors(sample, label)
            if proof_errors:
                fail('warmup' if label == 'warmup' else 'target', '; '.join(proof_errors))
                if record_value.get('outcome') == 'success':
                    record_value['outcome'] = 'failure'
        except Exception as error:
            fail('warmup' if label == 'warmup' else 'target', f'{label} proof failed: {error}')

    pair_errors = profile_pair_errors(warmup, target, request.workload)
    for error in pair_errors:
        phase = 'warmup' if error.startswith('warmup:') else 'target'
        fail(phase, error)

    if not fixture_thread_stopped:
        cleanup_ok = False
    if failed_phase is None:
        if warmup.get('outcome') != 'success' or target.get('outcome') != 'success':
            fail('target', 'profile run did not complete one successful warmup and target')
        elif not isinstance(profile_state, dict) or not all(
            profile_state.get(name) is True for name in ('enabled', 'disabled', 'dumped')
        ):
            fail('profile', 'profile enable, disable, or dump did not complete')
        elif not cleanup_ok or not fixture_thread_stopped:
            fail('cleanup', 'worker or fixture cleanup is incomplete')
        elif source_unchanged is not True or harness_unchanged is not True:
            fail('source', 'source stability was not proven')

    should_copy_stats = (
        failed_phase is None
        and warmup.get('outcome') == 'success'
        and target.get('outcome') == 'success'
        and cleanup_ok
        and fixture_thread_stopped
        and source_unchanged
        and harness_unchanged
        and not pair_errors
        and isinstance(profile_state, dict)
        and all(profile_state.get(name) is True for name in ('enabled', 'disabled', 'dumped'))
    )
    if should_copy_stats:
        try:
            stats_path = destination / 'trial.pstats'
            copy_profile_stats(run_root / 'trial.pstats', stats_path)
            size = stats_path.stat().st_size
            if size <= 0:
                raise OSError('copied profile stats are empty')
            profile_metadata = {'file': 'trial.pstats', 'bytes': size, 'sha256': sha256(stats_path)}
        except Exception as error:
            fail('copy', f'cannot copy profile stats: {type(error).__name__}: {error}')

    root_states = identity_states if proc is not None else []
    if runtime_root is not None and token:
        if (proc is None or worker_identity_complete and initial_identities is not None) \
                and removal_allowed(runtime_root, token, root_states):
            try:
                shutil.rmtree(runtime_root)
                root_removed = not runtime_root.exists()
            except OSError as error:
                fail('cleanup', f'cannot remove owned runtime root: {error}')
        if not root_removed:
            fail('cleanup', 'owned runtime root was retained because cleanup proof is incomplete')
    if not root_removed and profile_metadata is not None:
        discard_profile_stats()

    cleanup_report = {
        'worker_identity': cleanup_worker_identity,
        'post_worker_exit_identities': post_identities,
        'ok': cleanup_ok and fixture_thread_stopped,
        'root_removed': root_removed,
        'fixture_thread_stopped': fixture_thread_stopped,
    }
    report = {
        'schema': 'openbrowser-navigation-profile/1',
        'status': 'complete' if failed_phase is None and profile_metadata is not None and root_removed else 'failed',
        'failed_phase': failed_phase,
        'arm': 'openbrowser',
        'workload': request.workload,
        'delay_seconds': .15,
        'source': {
            'commit': source_commit_before,
            'manifest_start': source_start,
            'manifest_end': source_end,
            'unchanged': source_unchanged and harness_unchanged,
        },
        'harness': {'sha256': harness_start},
        'version_manifest_file': 'version-manifest.json',
        'version_manifest_sha256': sha256(copied_manifest),
        'tools': {
            'openbrowser_python': {
                'path': str(request.openbrowser_python),
                'version': python_record['version'],
                'sha256': python_record['sha256'],
            },
            'chrome': {
                'path': str(request.chrome),
                'version': chrome_record['version'],
                'sha256': chrome_record['sha256'],
            },
            'browser_runtime': browser_runtime or {},
            'packages': package_data if isinstance(package_data, dict) else {'packages': {}, 'loaded_packages': {}},
        },
        'warmup': warmup,
        'target': target,
        'profile': profile_metadata if failed_phase is None and root_removed else None,
        'cleanup': cleanup_report,
        'errors': errors,
        'verification_errors': [],
    }
    if report['status'] != 'complete' and failed_phase is None:
        failed_phase = report['failed_phase'] = 'copy'
        errors.append('profile artifacts did not reach a complete state')
        report['errors'] = errors
        report['profile'] = None
    if report['status'] == 'complete':
        report['verification_errors'] = profile_report_errors(report)
        if report['verification_errors']:
            failed_phase = report['failed_phase'] = 'copy'
            errors.extend(report['verification_errors'])
            report['errors'] = errors
            report['status'] = 'failed'
            report['profile'] = None
            discard_profile_stats()
    else:
        if not report['errors']:
            report['errors'] = [f'profile run failed during {report["failed_phase"] or "startup"}']
            if report['failed_phase'] is None:
                report['failed_phase'] = 'startup'
        report['failed_phase'] = report['failed_phase'] or 'startup'
        report['verification_errors'] = profile_report_errors(report)
        if not report['verification_errors']:
            report['verification_errors'] = ['profile report is failed']

    try:
        atomic_profile_json(destination / 'profile.json', report)
    except Exception as error:
        LOG.error('cannot write profile report: %s', error)
        return 1
    try:
        atomic_profile_json(destination / 'sha256.json', profile_hash_entries(destination))
    except Exception as error:
        LOG.error('cannot write profile hashes: %s', error)
        report['status'] = 'failed'
        report['failed_phase'] = report['failed_phase'] or 'copy'
        report['profile'] = None
        report['errors'] = [*report['errors'], f'cannot write profile hashes: {error}']
        report['verification_errors'] = profile_report_errors(report)
        try:
            atomic_profile_json(destination / 'profile.json', report)
        except Exception:
            pass
        return 1
    if report['status'] == 'complete' and not report['verification_errors']:
        try:
            verification_result = verify_profile(destination, request.result_root)
        except Exception as error:
            verification_result = None
            verification_error = f'offline artifact verification raised {type(error).__name__}: {error}'
        else:
            verification_error = 'saved profile artifacts failed offline verification'
        if type(verification_result) is int and verification_result == 0:
            return 0

        report['status'] = 'failed'
        report['failed_phase'] = 'artifact_verify'
        report['profile'] = None
        report['errors'].append(verification_error)
        report['verification_errors'] = [*report['verification_errors'], verification_error]
        discard_profile_stats()
        try:
            atomic_profile_json(destination / 'profile.json', report)
            atomic_profile_json(destination / 'sha256.json', profile_hash_entries(destination))
        except Exception as error:
            LOG.error('cannot write artifact verification failure report: %s', error)
        return 1
    return 1


def dispatch(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['profile']:
        return profile_cli(arguments[1:])
    parsed = parser().parse_args(arguments)
    try:
        if parsed.verify:
            code = verify(parsed.verify)
        elif parsed.worker:
            asyncio.run(worker(parsed))
            code = 0
        else:
            code = asyncio.run(main(parsed))
    except (ValueError, OSError, KeyError) as error:
        LOG.error('%s', error)
        code = 2
    return code


async def main(args):
    validate_config(args)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    destination.chmod(0o700)
    validate_private_directory(destination)
    root = Path(tempfile.mkdtemp(prefix='nb-', dir='/tmp' if os.name == 'posix' else None)).resolve()
    root.chmod(0o700)
    token = uuid.uuid4().hex
    atomic_profile_bytes(root / '.benchmark-owner', token.encode())
    source = Path(args.source)
    before = source_manifest(source)
    result = {
        'metadata': {
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'platform': platform.platform(), 'python': sys.version, 'cores': os.cpu_count(),
            'load_start': load_average(), 'script_sha256': sha256(SCRIPT),
            'samples_per_workload_arm': args.samples, 'source_commit': git_value(source, 'rev-parse', 'HEAD'),
            'source_dirty': git_value(source, 'status', '--porcelain'), 'source_manifest': before,
            'executables': {key: executable_metadata(getattr(args, key)) for key in
                            ('chrome', 'lightpanda', 'agent_browser', 'browser_use_python', 'openbrowser_python')},
            'scope': 'Local native warm-operation diagnostic. No LLM, cloud, or Docker workflow.',
            'performance_verdict': 'inconclusive: different engines, IPC and default wait semantics; no limiter profile',
            'delay_seconds': args.delay, 'runtime_root': str(root), 'owner_token': token,
        },
        'availability': availability(args), 'startup': {}, 'warmup': [], 'samples': [], 'cleanup': {},
    }
    if args.version_manifest:
        result['metadata']['release_discovery'] = json.loads(Path(args.version_manifest).read_text())
        result['metadata']['release_discovery_sha256'] = sha256(args.version_manifest)
    fixture = Fixture(args.delay)
    fixture_thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    fixture_thread.start()
    workers: dict[str, WorkerHandle] = {}
    try:
        for name, status in result['availability'].items():
            if status['state'] == 'unavailable':
                LOG.warning('%s unavailable: %s', name, status['reason'])
                continue
            run_root = root / name
            launch = make_worker_launch(
                arm=name, mode=WorkerMode.NAVIGATION, source_root=source.resolve(),
                candidate_python=Path(args.openbrowser_python).absolute(),
                browser_use_python=(Path(args.browser_use_python).absolute()
                                    if args.browser_use_python else None),
                runtime_root=root, worker_root=run_root, working_directory=run_root,
                log_directory=destination, log_prefix=name, owner_token=token,
                chrome=Path(args.chrome).absolute() if args.chrome else None,
                lightpanda=Path(args.lightpanda).absolute() if args.lightpanda else None,
                agent_browser=Path(args.agent_browser).absolute() if args.agent_browser else None,
            )
            try:
                handle = await launch_worker(launch)
                workers[name] = handle
                start = time.perf_counter_ns()
                response, start_errors = await start_worker(handle)
                result['startup'][name] = {**response, 'elapsed_ns': time.perf_counter_ns() - start}
                if start_errors:
                    status.update(state='failed', reason='; '.join(start_errors))
                else:
                    status['state'] = 'ready' if response.get('ok') else 'failed'
            except Exception as error:
                status.update(state='failed', reason=repr(error))
        active = [name for name, status in result['availability'].items() if status['state'] == 'ready']
        schedule = [(True, 0, name, 'fast') for name in active]
        for index in range(args.samples):
            ordered = active[index % len(active):] + active[:index % len(active)] if active else []
            for kind in ('fast', 'delayed'):
                schedule += [(False, index, name, kind) for name in ordered]
        for warmup, index, name, kind in schedule:
            nonce = f'{name}-{kind}-{index}-{uuid.uuid4().hex}'
            url = f'http://127.0.0.1:{fixture.server_address[1]}/fixture?{urlencode({"nonce": nonce, "delayed": int(kind == "delayed")})}'
            sample = {'arm': name, 'index': index, 'kind': kind, 'nonce': nonce, 'url': url}
            handle = workers[name]
            start = time.perf_counter_ns()
            try:
                response = await send_operation(handle, {'op': 'trial', **sample}, 40)
                if not response.get('ok'):
                    raise RuntimeError(response)
                sample.update(response['result'])
            except Exception as error:
                end = time.perf_counter_ns()
                sample.update(outcome='failure', exception=repr(error), timed_out=isinstance(error, TimeoutError),
                              start_ns=start, end_ns=end, elapsed_ns=end - start)
            sample['server'] = fixture.snapshot(nonce)
            sample['errors'] = workload_errors(sample, sample['server'], sample['end_ns'])
            if sample['errors']:
                sample['outcome'] = 'failure'
            result['warmup' if warmup else 'samples'].append(sample)
            with (destination / 'samples.jsonl').open('a') as stream:
                stream.write(json.dumps({'warmup': warmup, **sample}) + '\n')
            LOG.info('%s %s #%s %s %s ns', name, kind, index, sample['outcome'], sample['elapsed_ns'])
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        LOG.exception('comparison failed')
    finally:
        all_states = []
        for name, handle in workers.items():
            try:
                response = await cleanup_worker(handle)
            except Exception as error:
                response = {
                    'ok': False,
                    'error': f'{type(error).__name__}: {error}',
                    'worker_identity': {**handle.worker_identity, 'state': 'unknown'},
                    'post_worker_exit_identities': [],
                }
            result['cleanup'][name] = response
            all_states.extend(r.get('state', 'unknown') for r in response.get('post_worker_exit_identities', []))
            all_states.append(response.get('worker_identity', {}).get('state', 'unknown'))
            if not response.get('ok'):
                all_states.append('unknown')
            handle.stdout_log.close()
            handle.stderr_log.close()
        fixture.shutdown()
        fixture.server_close()
        fixture_thread.join(timeout=5)
        if fixture_thread.is_alive():
            all_states.append('unknown')
        for sample in result['warmup'] + result['samples']:
            sample['server'] = fixture.snapshot(sample['nonce'])
            sample['errors'] = workload_errors(sample, sample['server'], sample['end_ns'])
            if sample['errors']:
                sample['outcome'] = 'failure'
        result['metadata']['load_end'] = load_average()
        result['metadata']['source_manifest_end'] = source_manifest(source)
        result['metadata']['source_unchanged'] = before == result['metadata']['source_manifest_end']
        result['summary'] = summarize(result['samples'], result['availability'], args.samples)
        result['cleanup']['root_removed'] = removal_allowed(root, token, all_states)
        if result['cleanup']['root_removed']:
            shutil.rmtree(root)
        result['verification_errors'] = result_errors(result)
        (destination / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        hashes = {p.name: sha256(p) for p in destination.iterdir() if p.is_file()}
        (destination / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
        LOG.info('results: %s', destination)
        for error in result['verification_errors']:
            LOG.error('%s', error)
    return 1 if result['verification_errors'] else 0


def verify(directory):
    directory = Path(directory)
    hashes = json.loads((directory / 'sha256.json').read_text())
    errors = []
    if 'results.json' not in hashes:
        errors.append('missing results.json hash')
    for name, expected in hashes.items():
        if Path(name).name != name or not (directory / name).is_file() or sha256(directory / name) != expected:
            errors.append(f'file hash mismatch: {name}')
    result = json.loads((directory / 'results.json').read_text())
    errors.extend(result_errors(result))
    for error in errors:
        LOG.error('%s', error)
    if not errors:
        LOG.info('verified hashes, accounting, timed workload, source stability, and cleanup')
    return 1 if errors else 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stderr)
    raise SystemExit(dispatch())
