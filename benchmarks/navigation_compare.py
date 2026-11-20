"""Local current-release operation diagnostic. No LLM, cloud, Docker, or user profiles.

Each owned worker has one arm, a private profile, and an explicit command protocol.
The measured workload is navigation through load, observation, then one DOM mutation
and one POST. CLI transport is included only for the CLI-only agent-browser arm.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import importlib.metadata
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
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
from urllib.parse import parse_qs, urlencode, urlsplit

import psutil

LOG = logging.getLogger('comparison')
PREFIX = 'BENCH_RESULT:'
SCRIPT = Path(__file__).resolve()
ARMS = ('raw-cdp', 'lightpanda', 'agent-browser', 'browser-use', 'openbrowser')
CLAIM_TIERS = ('historical', 'pinned')
LANE = 'native-navigation'
CHROME_ARMS = {'raw-cdp', 'agent-browser', 'browser-use', 'openbrowser'}
OBSERVE = """JSON.stringify({url:location.href,title:document.title,
readyState:document.readyState,marker:document.getElementById('marker')?.textContent,
loaded:document.documentElement.dataset.loaded,scriptDone:document.documentElement.dataset.scriptDone,
field:document.getElementById('value')?.value})"""


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


def state(r):
    try:
        p = psutil.Process(r['pid'])
        if p.create_time() != r['created']:
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
                           '--use-mock-keychain', 'about:blank']
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
            self.browser = BrowserSession(browser_profile=BrowserProfile(headless=True, executable_path=str(self.chrome), args=['--use-mock-keychain'],
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
            return await self.cli('open', url)
        elif self.name == 'browser-use':
            def operation():
                result = self.helpers.goto_url(url)
                if not self.helpers.wait_for_load(timeout=15):
                    raise TimeoutError('browser-use prescribed wait_for_load')
                return result
            return await asyncio.to_thread(operation)
        else:
            return await self.browser.navigate_to(url)

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
        return {'errors': failures, 'identities': identities}


async def worker(args):
    arm = Arm(args)
    stopped = False
    try:
        while raw := await asyncio.to_thread(sys.stdin.readline):
            request = json.loads(raw)
            try:
                if request['op'] == 'start':
                    result = await arm.start()
                elif request['op'] == 'trial':
                    result = await arm.trial(request)
                elif request['op'] == 'stop':
                    result = await arm.stop()
                    stopped = True
                else:
                    raise ValueError('unknown worker operation')
                reply = {'ok': True, 'result': result}
            except Exception as error:
                reply = {'ok': False, 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            sys.stdout.write(PREFIX + json.dumps(reply) + '\n')
            sys.stdout.flush()
            if request['op'] == 'stop':
                break
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


async def cleanup_worker(proc, stdout, identity, initial, token):
    captured = {(r['pid'], r['created']): r for r in initial}
    cleanup = {}
    try:
        cleanup = await send(proc, {'op': 'stop'}, stdout, 60)
        captured.update({(r['pid'], r['created']): r
                         for r in cleanup.get('result', {}).get('identities', [])})
        proc.stdin.close()
        await asyncio.wait_for(proc.wait(), 15)
    except Exception as error:
        cleanup['error'] = repr(error)
        proc.stdin.close()
    cleanup['fallback_used'] = state(identity) == 'alive' or any(state(r) == 'alive' for r in captured.values())
    if cleanup['fallback_used']:
        captured.update({(r['pid'], r['created']): r for r in descendant_identities(identity, token)})
        await terminate_owned([identity, *captured.values()], token)
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except TimeoutError:
            cleanup['error'] = 'owned worker did not exit after verified termination'
    cleanup['post_worker_exit_identities'] = [{**r, 'state': state(r)} for r in captured.values()]
    cleanup['worker_identity'] = {**identity, 'state': state(identity)}
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


def claim_scope():
    """The only claim this lane can support."""
    return {
        'lane': LANE,
        'measured': [
            'local navigation through the adapter load wait',
            'one DOM input assignment',
            'one loopback POST response',
            'fast page and deferred-external-script page',
        ],
        'not_measured': [
            'CodeAgent task success', 'MCP tool behavior', 'CLI or daemon behavior',
            'agent framework behavior', 'cold start', 'public-site compatibility',
            'production behavior', 'CPU limiter or engine profile',
        ],
        'cross_tool_ranking': 'inconclusive',
        'publishable_to_comparison_page': False,
    }


def valid_timestamp(value):
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def load_pins(path):
    """Load and validate the canonical pin file."""
    data = json.loads(Path(path).read_text())
    pins = data.get('pins')
    if not isinstance(pins, dict):
        raise ValueError('pin file must contain a pins object')
    cleaned = {}
    for name, pin in pins.items():
        if not isinstance(name, str) or not name:
            raise ValueError('pin names must be non-empty strings')
        if not isinstance(pin, dict):
            raise ValueError(f'pin {name} must be an object')
        version = pin.get('version')
        commit = pin.get('source_commit')
        if not (isinstance(version, str) and version) and not (isinstance(commit, str) and commit):
            raise ValueError(f'pin {name} must have a version or source_commit string')
        cleaned[name] = dict(pin)
    captured = data.get('captured_utc')
    if captured is not None and not valid_timestamp(captured):
        raise ValueError('captured_utc must be an ISO 8601 timestamp')
    return {'captured_utc': captured, 'pins': cleaned}


def version_matches(pin, actual):
    """Match a pin as a complete version token, preserving suffixes."""
    if not isinstance(pin, str) or not pin or not isinstance(actual, str) or not actual:
        return False
    return re.search(rf'(?<![0-9A-Za-z.+-]){re.escape(pin)}(?![0-9A-Za-z.+-])', actual) is not None


def recorded_versions(result, arm):
    meta = result.get('metadata', {})
    executables = meta.get('executables') or {}
    startup = result.get('startup') or {}
    chrome = (executables.get('chrome') or {}).get('version')
    if arm == 'raw-cdp':
        return {'chrome': chrome}
    if arm == 'agent-browser':
        return {'agent-browser': (executables.get('agent_browser') or {}).get('version'), 'chrome': chrome}
    if arm == 'lightpanda':
        return {'lightpanda': (executables.get('lightpanda') or {}).get('version')}
    if arm == 'browser-use':
        loaded = ((startup.get('browser-use') or {}).get('result') or {}).get('loaded_packages') or {}
        return {'browser-use': loaded.get('browser-use'), 'chrome': chrome}
    if arm == 'openbrowser':
        loaded = ((startup.get('openbrowser') or {}).get('result') or {}).get('loaded_packages') or {}
        return {'openbrowser': loaded.get('openbrowser-ai'), 'chrome': chrome}
    return {}


def _pin_version_errors(errors, label, pin, actual):
    version = (pin or {}).get('version')
    if not version:
        errors.append(f'{label}: no pinned version for a pinned claim')
        return
    if not actual:
        errors.append(f'{label}: recorded version is missing')
        return
    if not version_matches(version, actual):
        errors.append(f'{label}: recorded version {actual!r} does not match pin {version!r}')


def claim_errors(result):
    errors = []
    meta = result.get('metadata', {})
    tier = meta.get('claim_tier', 'historical')
    if tier not in CLAIM_TIERS:
        errors.append(f'unknown claim tier: {tier!r}')
        return errors
    scope = meta.get('claim_scope')
    if scope is not None:
        expected = claim_scope()
        if scope != expected:
            errors.append('claim scope does not match the native-navigation canonical scope')
        if meta.get('scope') != 'Local native warm-operation diagnostic. No LLM, cloud, or Docker workflow.':
            errors.append('legacy scope statement was changed')
        if meta.get('performance_verdict') != 'inconclusive: different engines, IPC and default wait semantics; no limiter profile':
            errors.append('legacy performance verdict must remain the canonical inconclusive verdict')
    elif tier == 'pinned':
        errors.append('pinned claim requires a claim_scope')
    if tier != 'pinned':
        return errors
    pin_record = meta.get('pins') or {}
    if not valid_timestamp(pin_record.get('captured_utc')):
        errors.append('pinned claim requires a valid captured_utc')
    pins = pin_record.get('pins') or {}
    availability = result.get('availability') or {}
    requested = meta.get('requested_arms') or list(availability)
    if set(requested) != set(availability):
        errors.append('requested_arms does not match availability')
    for arm in requested:
        if arm not in ARMS:
            errors.append(f'unknown requested arm: {arm}')
            continue
        if availability.get(arm, {}).get('state') != 'ready':
            continue
        versions = recorded_versions(result, arm)
        if arm == 'raw-cdp':
            _pin_version_errors(errors, 'raw-cdp/chrome', pins.get('chrome'), versions.get('chrome'))
        elif arm == 'lightpanda':
            _pin_version_errors(errors, 'lightpanda', pins.get('lightpanda'), versions.get('lightpanda'))
        elif arm == 'agent-browser':
            _pin_version_errors(errors, 'agent-browser', pins.get('agent-browser'), versions.get('agent-browser'))
            _pin_version_errors(errors, 'agent-browser/chrome', pins.get('chrome'), versions.get('chrome'))
        elif arm == 'browser-use':
            _pin_version_errors(errors, 'browser-use', pins.get('browser-use'), versions.get('browser-use'))
            _pin_version_errors(errors, 'browser-use/chrome', pins.get('chrome'), versions.get('chrome'))
        elif arm == 'openbrowser':
            pin = pins.get('openbrowser') or {}
            if pin.get('version') and pin.get('source_commit'):
                errors.append('openbrowser: pin must use version or source_commit, not both')
            elif pin.get('version'):
                _pin_version_errors(errors, 'openbrowser', pin, versions.get('openbrowser'))
            elif pin.get('source_commit'):
                actual = meta.get('source_commit') or ''
                want = pin['source_commit']
                if len(want) < 7:
                    errors.append('openbrowser/source: source_commit pin must be at least 7 characters')
                elif not actual or len(actual) < len(want) or not actual.startswith(want):
                    errors.append(f'openbrowser/source: commit {actual!r} does not match pin {want!r}')
            else:
                errors.append('openbrowser: no pinned version or source commit for a pinned claim')
            if meta.get('source_dirty'):
                errors.append('openbrowser/source: source tree was dirty during the pinned run')
            _pin_version_errors(errors, 'openbrowser/chrome', pins.get('chrome'), versions.get('chrome'))
    if any(arm in CHROME_ARMS for arm in requested) and 'chrome' not in pins:
        errors.append('pinned claim requires a chrome pin for Chrome-based arms')
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
    errors.extend(claim_errors(result))
    return errors


def local_open(url, timeout):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=timeout)


def worker_environment(root, token, source):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('BU_', 'BH_', 'BROWSER_USE_', 'OPENBROWSER_', 'AGENT_BROWSER_', 'LIGHTPANDA_'))}
    env.update(
        HOME=str(root / 'home'), XDG_CONFIG_HOME=str(root / 'config'), XDG_CACHE_HOME=str(root / 'cache'),
        PYTHONPATH=str(source / 'src'), NAVIGATION_BENCHMARK_OWNER=token,
        ANONYMIZED_TELEMETRY='false', LIGHTPANDA_DISABLE_TELEMETRY='true', BH_TELEMETRY='false',
        BROWSER_HARNESS_TELEMETRY='false', BROWSER_USE_SETUP_LOGGING='false', BROWSER_USE_NO_UPDATE_CHECK='1',
        BH_HOME=str(root / 'bh'), BH_RUNTIME_DIR=str(root / 'runtime'), BH_TMP_DIR=str(root / 'tmp'),
        BH_TAB_MARKER='0', BH_RECORD='0', AGENT_BROWSER_SOCKET_DIR=str(root / 'agent'),
        AGENT_BROWSER_NO_VERSION_CHECK='1', AGENT_BROWSER_NO_WEBMCP='1', DO_NOT_TRACK='1',
        OPENBROWSER_HEADLESS='true', OPENBROWSER_DISABLE_EXTENSIONS='true',
    )
    return env


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default=os.getenv('NAV_BENCH_SOURCE'))
    p.add_argument('--output')
    p.add_argument('--chrome', default=os.getenv('NAV_BENCH_CHROME'))
    p.add_argument('--lightpanda', default=os.getenv('NAV_BENCH_LIGHTPANDA'))
    p.add_argument('--agent-browser', default=os.getenv('NAV_BENCH_AGENT_BROWSER'))
    p.add_argument('--browser-use-python', default=os.getenv('NAV_BENCH_BROWSER_USE_PYTHON'))
    p.add_argument('--openbrowser-python', default=os.getenv('NAV_BENCH_OPENBROWSER_PYTHON', sys.executable))
    p.add_argument('--version-manifest', help='Optional JSON release-discovery evidence, copied into results')
    p.add_argument('--claim-tier', choices=CLAIM_TIERS, default='historical',
                   help='historical (default) or pinned; pinned requires --pins and exact version matches')
    p.add_argument('--pins', help='Canonical pin file required for a pinned claim')
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--delay', type=float, default=.15)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--verify', metavar='RESULT_DIRECTORY', help='Verify saved hashes and result contracts without browsers')
    p.add_argument('--worker', choices=ARMS, help=argparse.SUPPRESS)
    p.add_argument('--run-root', help=argparse.SUPPRESS)
    return p


def validate_config(args):
    if args.samples < 1 or not math.isfinite(args.delay) or args.delay < 0:
        raise ValueError('samples must be positive and delay must be finite and non-negative')
    if len(args.arms) != len(set(args.arms)):
        raise ValueError('duplicate arms are not allowed')
    if not args.source or not (Path(args.source) / 'src/openbrowser/__init__.py').is_file():
        raise ValueError('--source must identify the source repository')
    if not args.output or Path(args.output).exists() or Path(args.output).is_symlink():
        raise ValueError('--output must be a new directory')
    args.source = str(Path(args.source).resolve())
    for key in ('chrome', 'lightpanda', 'agent_browser', 'browser_use_python', 'openbrowser_python'):
        value = getattr(args, key)
        if value:
            path = Path(value).expanduser().absolute()
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f'{key} is not an executable file: {path}')
            setattr(args, key, str(path))
    if args.version_manifest:
        json.loads(Path(args.version_manifest).read_text())
    if args.claim_tier == 'pinned' and not args.pins:
        raise ValueError('a pinned claim requires --pins')
    if args.pins:
        load_pins(args.pins)


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


def load_average():
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        return None


async def main(args):
    validate_config(args)
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    root = Path(tempfile.mkdtemp(prefix='nb-', dir='/tmp' if os.name == 'posix' else None)).resolve()
    root.chmod(0o700)
    token = uuid.uuid4().hex
    (root / '.benchmark-owner').write_text(token)
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
            'claim_tier': args.claim_tier, 'requested_arms': list(args.arms), 'claim_scope': claim_scope(),
            'delay_seconds': args.delay, 'runtime_root': str(root), 'owner_token': token,
        },
        'availability': availability(args), 'startup': {}, 'warmup': [], 'samples': [], 'cleanup': {},
    }
    if args.version_manifest:
        result['metadata']['release_discovery'] = json.loads(Path(args.version_manifest).read_text())
        result['metadata']['release_discovery_sha256'] = sha256(args.version_manifest)
    if args.pins:
        result['metadata']['pins'] = load_pins(args.pins)
        result['metadata']['pins_sha256'] = sha256(args.pins)
        shutil.copyfile(args.pins, destination / 'pins.json')
    fixture = Fixture(args.delay)
    fixture_thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    fixture_thread.start()
    workers = {}
    try:
        for name, status in result['availability'].items():
            if status['state'] == 'unavailable':
                LOG.warning('%s unavailable: %s', name, status['reason'])
                continue
            run_root = root / name
            run_root.mkdir(mode=0o700)
            env = worker_environment(run_root, token, source)
            for variable in ('HOME', 'XDG_CONFIG_HOME', 'XDG_CACHE_HOME'):
                Path(env[variable]).mkdir(mode=0o700)
            python = args.openbrowser_python if name == 'openbrowser' else (args.browser_use_python if name == 'browser-use' else sys.executable)
            env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
            stderr = (destination / f'{name}.stderr.log').open('w')
            stdout = (destination / f'{name}.stdout.log').open('w')
            command = [str(python), '-u', str(SCRIPT), '--worker', name, '--run-root', str(run_root), '--source', str(source)]
            for option in ('chrome', 'lightpanda', 'agent_browser'):
                if getattr(args, option):
                    command += ['--' + option.replace('_', '-'), getattr(args, option)]
            try:
                proc = await asyncio.create_subprocess_exec(*command, cwd=run_root, env=env,
                            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr)
                identity = record(psutil.Process(proc.pid), token)
                workers[name] = (proc, stdout, stderr, identity)
                start = time.perf_counter_ns()
                response = await send(proc, {'op': 'start'}, stdout)
                result['startup'][name] = {**response, 'elapsed_ns': time.perf_counter_ns() - start}
                status['state'] = 'ready' if response.get('ok') else 'failed'
            except Exception as error:
                status.update(state='failed', reason=repr(error))
                if name not in workers:
                    stdout.close()
                    stderr.close()
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
            proc, stdout, _, _ = workers[name]
            start = time.perf_counter_ns()
            try:
                response = await send(proc, {'op': 'trial', **sample}, stdout, 40)
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
        for name, (proc, stdout, stderr, identity) in workers.items():
            initial = result['startup'].get(name, {}).get('result', {}).get('owned', [])
            response = await cleanup_worker(proc, stdout, identity, initial, token)
            result['cleanup'][name] = response
            all_states.extend(r['state'] for r in response['post_worker_exit_identities'])
            all_states.append(response['worker_identity']['state'])
            if not response.get('ok'):
                all_states.append('unknown')
            stdout.close()
            stderr.close()
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
    arguments = parser().parse_args()
    try:
        if arguments.verify:
            code = verify(arguments.verify)
        elif arguments.worker:
            asyncio.run(worker(arguments))
            code = 0
        else:
            code = asyncio.run(main(arguments))
    except (ValueError, OSError, KeyError) as error:
        LOG.error('%s', error)
        code = 2
    raise SystemExit(code)
