import importlib.util
import asyncio
import contextlib
import io
import json
import os
import shutil
import statistics
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/navigation_compare.py'
spec = importlib.util.spec_from_file_location('navigation_compare', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


@pytest.fixture
def private_result_root():
    root = Path('/tmp').resolve(strict=True) / f'openbrowser-navigation-test-{benchmark.uuid.uuid4().hex}'
    yield root
    shutil.rmtree(root, ignore_errors=True)


def make_worker_launch(tmp_path, *, arm='openbrowser', mode=None, candidate_python=None,
                       browser_use_python=None, log_directory=None, log_prefix='worker'):
    runtime_root = Path(tmp_path) / 'runtime'
    runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    worker_root = runtime_root / arm
    log_directory = Path(log_directory or tmp_path)
    log_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return benchmark.make_worker_launch(
        arm=arm,
        mode=mode or benchmark.WorkerMode.NAVIGATION,
        source_root=SCRIPT.parents[1],
        candidate_python=Path(candidate_python or sys.executable),
        browser_use_python=Path(browser_use_python or sys.executable),
        runtime_root=runtime_root,
        worker_root=worker_root,
        working_directory=worker_root,
        log_directory=log_directory,
        log_prefix=log_prefix,
        owner_token='private-token',
    )


def test_normal_benchmark_import_does_not_load_profile_modules(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    env = {
        **{key: value for key, value in os.environ.items() if not key.startswith('AWS_')},
        'HOME': str(home),
        'XDG_CONFIG_HOME': str(home / 'config'),
        'XDG_CACHE_HOME': str(home / 'cache'),
        'AWS_EC2_METADATA_DISABLED': 'true',
    }
    code = (
        'import importlib.util, sys; '
        'spec = importlib.util.spec_from_file_location("navigation_compare_import_probe", sys.argv[1]); '
        'module = importlib.util.module_from_spec(spec); '
        'sys.modules[spec.name] = module; '
        'spec.loader.exec_module(module); '
        'assert "cProfile" not in sys.modules; '
        'assert "pstats" not in sys.modules'
    )
    result = subprocess.run(
        [sys.executable, '-c', code, str(SCRIPT)],
        capture_output=True,
        check=False,
        cwd=SCRIPT.parents[1],
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def completed_sample(kind='delayed'):
    nonce = 'unique-request'
    return {
        'arm': 'raw-cdp', 'kind': kind, 'index': 0, 'nonce': nonce,
        'url': 'http://127.0.0.1:1234/fixture?nonce=unique-request',
        'start_ns': 100, 'end_ns': 200, 'checked_ns': 195, 'elapsed_ns': 100,
        'outcome': 'success',
        'navigation_observation': {
            'url': 'http://127.0.0.1:1234/fixture?nonce=unique-request',
            'title': f'fixture-{nonce}', 'marker': nonce, 'loaded': nonce,
            'readyState': 'complete', 'scriptDone': nonce,
        },
        'effect_observation': {'field': nonce, 'response': {'count': 1, 'nonce': nonce, 'payload': nonce}},
        'final_observation': {'url': 'http://127.0.0.1:1234/fixture?nonce=unique-request',
                              'title': f'fixture-{nonce}', 'marker': nonce, 'loaded': nonce,
                              'readyState': 'complete', 'scriptDone': nonce, 'field': nonce},
        'server': {'effects': 1, 'requests': [
            {'path': path, 'method': method, 'nonce': nonce, 'status': 200,
             'received_ns': 110, 'completed_ns': 190}
            for path, method in [('/fixture', 'GET'), ('/slow.js', 'GET'), ('/effect', 'POST')]
        ]},
    }


def test_workload_accepts_exact_observation_and_server_work_inside_timer():
    sample = completed_sample()
    assert benchmark.workload_errors(sample, sample['server'], 200) == []


@pytest.mark.asyncio
async def test_openbrowser_navigate_explicitly_waits_for_load():
    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.name = 'openbrowser'
    calls = []

    class Browser:
        async def navigate_to(self, url, *, wait_until=None):
            calls.append((url, wait_until))
            return {'url': url}

    arm.browser = Browser()
    url = 'http://127.0.0.1:12345/fixture?nonce=load-check'

    await arm.navigate(url)

    assert calls == [(url, 'load')]


@pytest.mark.asyncio
async def test_arm_trial_reports_elapsed_navigation_and_post_navigation_phases(monkeypatch):
    nonce = 'phase-proof'
    url = f'http://127.0.0.1:12345/fixture?nonce={nonce}'
    observation = {
        'url': url, 'title': f'fixture-{nonce}', 'readyState': 'complete',
        'marker': nonce, 'loaded': nonce, 'scriptDone': nonce,
    }
    server = {'effects': 1, 'requests': [
        {'path': '/fixture', 'method': 'GET', 'nonce': nonce, 'status': 200,
         'received_ns': 110, 'completed_ns': 120},
        {'path': '/effect', 'method': 'POST', 'nonce': nonce, 'status': 200,
         'received_ns': 160, 'completed_ns': 170},
    ]}

    class AuditResponse(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    arm = benchmark.Arm.__new__(benchmark.Arm)
    observations = [observation, {**observation, 'field': nonce}]

    async def navigate(_url):
        return None

    async def evaluate(expression):
        if expression == benchmark.OBSERVE:
            return json.dumps(observations.pop(0))
        return json.dumps({
            'field': nonce,
            'response': {'count': 1, 'nonce': nonce, 'payload': nonce},
        })

    arm.navigate = navigate
    arm.evaluate = evaluate
    arm.ownership = lambda: []
    ticks = iter((100, 150, 180, 200))
    monkeypatch.setattr(benchmark.time, 'perf_counter_ns', lambda: next(ticks))
    monkeypatch.setattr(
        benchmark, 'local_open',
        lambda *_args, **_kwargs: AuditResponse(json.dumps(server)),
    )

    result = await arm.trial({
        'arm': 'openbrowser', 'kind': 'fast', 'index': 0,
        'nonce': nonce, 'url': url,
    })

    assert result['outcome'] == 'success'
    assert result['elapsed_ns'] == 100
    assert result['navigation_ns'] == 50
    assert result['post_navigation_ns'] == 50


@pytest.mark.asyncio
async def test_agent_browser_navigate_waits_for_load_after_open():
    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.name = 'agent-browser'
    calls = []

    async def cli(*command):
        calls.append(command)
        return {'success': True}

    arm.cli = cli
    url = 'http://127.0.0.1:12345/fixture?nonce=load-check'

    await arm.navigate(url)

    assert calls == [('open', url), ('wait', '--load', 'load')]


@pytest.mark.parametrize('field,value', [('url', 'about:blank'), ('readyState', 'interactive'),
                                        ('loaded', None), ('scriptDone', 'pending'),
                                        ('marker', 'old'), ('title', 'wrong')])
def test_workload_rejects_incomplete_navigation(field, value):
    sample = completed_sample()
    sample['navigation_observation'][field] = value
    assert benchmark.workload_errors(sample, sample['server'], 200)


def test_workload_rejects_duplicate_post_and_wrong_field():
    sample = completed_sample()
    sample['server']['requests'].append(dict(sample['server']['requests'][-1]))
    sample['effect_observation']['field'] = 'wrong'
    sample['server']['effects'] = 2
    assert len(benchmark.workload_errors(sample, sample['server'], 200)) == 3


def test_workload_rechecks_final_document_after_side_effect():
    sample = completed_sample()
    sample['final_observation']['url'] = 'about:blank'
    sample['final_observation']['field'] = 'lost-value'
    assert len(benchmark.workload_errors(sample, sample['server'], 200)) == 2


@pytest.mark.parametrize('field,value', [('received_ns', 99), ('completed_ns', 201), ('status', 500)])
def test_workload_rejects_server_work_outside_timer_or_http_errors(field, value):
    sample = completed_sample()
    sample['server']['requests'][0][field] = value
    assert benchmark.workload_errors(sample, sample['server'], 200)


def test_summary_keeps_failures_unavailable_and_missing_attempts_distinct():
    sample = completed_sample('fast')
    failed = {**sample, 'index': 1, 'outcome': 'failure', 'timed_out': True, 'elapsed_ns': 500}
    result = benchmark.summarize([sample, failed], {
        'raw-cdp': {'state': 'ready'}, 'lightpanda': {'state': 'unavailable'},
    }, 3)
    assert result['raw-cdp/fast'] == {
        'expected': 3, 'attempted': 2, 'successful': 1, 'failures': 1,
        'timeouts': 1, 'unavailable': 0, 'not_run': 1,
        'median_ns': 100, 'min_ns': 100, 'max_ns': 100,
    }
    assert result['lightpanda/fast']['unavailable'] == 3
    assert result['lightpanda/fast']['successful'] == 0
    assert result['lightpanda/fast']['median_ns'] is None


def test_identity_guard_rejects_pid_reuse_changed_command_and_unowned_process():
    identity = {'pid': 12, 'created': 2.0, 'cmdline': ['chrome', '--user-data-dir=private'], 'owner_token': 'run'}
    assert benchmark.same_owned_identity(identity, dict(identity), 'run')
    for change in ({'created': 3.0}, {'pid': 13}, {'cmdline': ['chrome', '--user-data-dir=user']}, {'owner_token': 'other'}):
        assert not benchmark.same_owned_identity(identity, {**identity, **change}, 'run')
    assert not benchmark.same_owned_identity({**identity, 'owner_token': None}, identity, 'run')


def test_identity_guard_rejects_missing_process_identity_fields():
    assert not benchmark.same_owned_identity({'owner_token': 'run'}, {'owner_token': 'run'}, 'run')


def test_state_preserves_valid_live_and_reused_identity_behavior(monkeypatch):
    class Process:
        def create_time(self):
            return 2.0

        def status(self):
            return benchmark.psutil.STATUS_RUNNING

    process = Process()
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda _pid: process)
    identity = {'pid': 12, 'created': 2.0, 'owner_token': 'run'}
    assert benchmark.state(identity) == 'alive'
    assert benchmark.state({**identity, 'created': 1.0}) == 'reused'


@pytest.mark.parametrize(('has_created', 'created'), [
    (False, None),
    (True, None),
    (True, 'not-a-number'),
    (True, float('nan')),
    (True, float('inf')),
    (True, float('-inf')),
])
def test_live_identity_with_missing_or_invalid_creation_time_is_unknown(monkeypatch, has_created, created):
    class Process:
        def create_time(self):
            return 2.0

        def status(self):
            return benchmark.psutil.STATUS_RUNNING

    process = Process()
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda _pid: process)
    identity = {'pid': 12, 'owner_token': 'run'}
    if has_created:
        identity['created'] = created
    assert benchmark.state(identity) == 'unknown'


def test_state_still_reports_definitely_absent_pid_with_missing_creation_time(monkeypatch):
    def absent_process(pid):
        raise benchmark.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(benchmark.psutil, 'Process', absent_process)
    assert benchmark.state({'pid': 12}) == 'absent'


@pytest.mark.asyncio
async def test_owned_cleanup_escalates_only_after_rechecking_identity(tmp_path, monkeypatch):
    _, env = runtime_environment(tmp_path)
    process = ProcessIdentity(12, env, ['owned-browser'])
    calls = process.signals
    saved = benchmark.record(process, 'run')
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: process)
    monkeypatch.setattr(benchmark, 'state', lambda identity: 'absent' if 'kill' in calls else 'alive')
    await benchmark.terminate_owned([saved], 'run', grace=0)
    assert calls == ['terminate', 'kill']

    calls.clear()
    monkeypatch.setattr(process, 'create_time', lambda: 2.0 if not calls else 3.0)
    await benchmark.terminate_owned([saved], 'run', grace=0)
    assert calls == ['terminate']


@pytest.mark.asyncio
async def test_graceful_shutdown_timeout_still_reaches_owned_cleanup(monkeypatch):
    calls = []
    timeout = asyncio.timeout
    monkeypatch.setattr(benchmark.asyncio, 'timeout', lambda seconds: timeout(.001))

    class Browser:
        async def kill(self):
            await asyncio.Event().wait()

    async def cleanup(identities, token, grace=2):
        calls.append(token)
        return []

    monkeypatch.setattr(benchmark, 'terminate_owned', cleanup)
    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.name, arm.browser, arm.proc, arm.cdp = 'openbrowser', Browser(), None, None
    arm.token, arm.owned = 'run', {}
    arm.ownership = lambda: []
    result = await arm.stop()
    assert calls == ['run']
    assert result['errors']


@pytest.mark.asyncio
async def test_stop_reports_identity_state_after_reaping_browser(monkeypatch):
    reaped = False
    identity = {'pid': 1234, 'created': 1.0, 'cmdline': ['owned-browser'],
                'owner_token': 'run', 'owner_root': '/private/tmp/run/raw-cdp'}

    class Process:
        def wait(self, timeout):
            nonlocal reaped
            reaped = True

    async def terminate(identities, token):
        return [{**identity, 'state': 'zombie'}]

    monkeypatch.setattr(benchmark, 'terminate_owned', terminate)
    monkeypatch.setattr(benchmark, 'state', lambda _identity: 'absent' if reaped else 'zombie')
    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.name, arm.browser, arm.cdp, arm.proc = 'raw-cdp', None, None, Process()
    arm.token, arm.owned = 'run', {identity['pid']: identity}
    arm.ownership = lambda: [identity]

    result = await arm.stop()

    assert reaped is True
    assert result['errors'] == []
    assert result['identities'][0]['state'] == 'absent'


@pytest.mark.asyncio
async def test_coordinator_stop_failure_still_reaps_only_verified_owned_identities(monkeypatch):
    worker = {
        'pid': 12, 'created': 2.0, 'cmdline': ['worker'],
        'owner_token': 'run', 'owner_root': '/private/tmp/run/openbrowser',
    }
    browser = {**worker, 'pid': 13, 'cmdline': ['owned-browser']}
    calls = []
    alive = True

    class Input:
        def close(self):
            calls.append('close-input')

    class Process:
        stdin = Input()
        returncode = None

        async def wait(self):
            self.returncode = 0

    async def failed_send(*args, **kwargs):
        raise TimeoutError('unresponsive worker')

    async def terminate(identities, token, grace=2):
        nonlocal alive
        calls.append(('terminate', {i['pid'] for i in identities}, token))
        alive = False

    monkeypatch.setattr(benchmark, 'send', failed_send)
    monkeypatch.setattr(benchmark, 'descendant_identities', lambda identity, token: [browser])
    monkeypatch.setattr(benchmark, 'terminate_owned', terminate)
    monkeypatch.setattr(benchmark, 'state', lambda identity: 'alive' if alive else 'absent')
    handle = benchmark.WorkerHandle(
        process=Process(), stdout_log=io.StringIO(), stderr_log=io.StringIO(),
        worker_identity=worker, initial_owned=(browser,), owner_token='run',
        mode=benchmark.WorkerMode.NAVIGATION, state=benchmark.WorkerState.READY,
    )
    result = await benchmark.cleanup_worker(handle)
    assert ('terminate', {12, 13}, 'run') in calls
    assert result['worker_identity']['state'] == 'absent'
    assert result['post_worker_exit_identities'][0]['state'] == 'absent'
    assert result['fallback_used']
    assert result['error']


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['raw-cdp', 'openbrowser'])
async def test_private_home_chrome_never_uses_host_keychain(tmp_path, monkeypatch, name):
    root = tmp_path / name
    root.mkdir()
    (tmp_path / '.benchmark-owner').write_text('run')
    monkeypatch.setenv('NAVIGATION_BENCHMARK_OWNER', 'run')
    args = SimpleNamespace(worker=name, worker_mode=benchmark.WorkerMode.NAVIGATION.value,
                           run_root=str(root), source=str(SCRIPT.parents[1]),
                           chrome='chrome', lightpanda=None, agent_browser=None)
    arm = benchmark.Arm(args)
    captured = {}

    if name == 'raw-cdp':
        def no_launch(command, **kwargs):
            captured['args'] = command
            raise RuntimeError('browser launch prevented by test')
        monkeypatch.setattr(benchmark.subprocess, 'Popen', no_launch)
    else:
        import openbrowser.browser

        def profile(**kwargs):
            captured.update(kwargs)
            return kwargs

        def no_session(**kwargs):
            raise RuntimeError('browser launch prevented by test')
        monkeypatch.setattr(openbrowser.browser, 'BrowserProfile', profile)
        monkeypatch.setattr(openbrowser.browser, 'BrowserSession', no_session)
    try:
        with pytest.raises(RuntimeError, match='browser launch prevented by test'):
            await arm.start()
        assert '--use-mock-keychain' in captured['args']
    finally:
        if hasattr(arm, 'log'):
            arm.log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['raw-cdp', 'openbrowser'])
@pytest.mark.parametrize(
    ('mode', 'expected_srgb'),
    [(benchmark.WorkerMode.NAVIGATION, False), (benchmark.WorkerMode.SCREENSHOT, True)],
)
async def test_chrome_start_uses_srgb_only_for_screenshot_mode(tmp_path, monkeypatch, name, mode, expected_srgb):
    root = tmp_path / name
    root.mkdir()
    (tmp_path / '.benchmark-owner').write_text('run')
    monkeypatch.setenv('NAVIGATION_BENCHMARK_OWNER', 'run')
    args = SimpleNamespace(
        worker=name, worker_mode=mode.value, run_root=str(root), source=str(SCRIPT.parents[1]),
        chrome='chrome', lightpanda=None, agent_browser=None,
    )
    captured = {}
    messages = [{'op': 'start'}, {'op': 'stop'}]
    stdout = io.StringIO()
    monkeypatch.setattr(benchmark.sys, 'stdin', io.StringIO(''.join(json.dumps(item) + '\n' for item in messages)))
    monkeypatch.setattr(benchmark.sys, 'stdout', stdout)

    if name == 'raw-cdp':
        def no_launch(command, **kwargs):
            captured['args'] = command
            raise RuntimeError('browser launch prevented by test')
        monkeypatch.setattr(benchmark.subprocess, 'Popen', no_launch)
    else:
        import openbrowser.browser

        def profile(**kwargs):
            captured.update(kwargs)
            return kwargs

        def no_session(**kwargs):
            raise RuntimeError('browser launch prevented by test')
        monkeypatch.setattr(openbrowser.browser, 'BrowserProfile', profile)
        monkeypatch.setattr(openbrowser.browser, 'BrowserSession', no_session)

    await benchmark.worker(args)

    chrome_args = captured['args']
    assert ('--force-color-profile=srgb' in chrome_args) is expected_srgb
    replies = [json.loads(line[len(benchmark.PREFIX):]) for line in stdout.getvalue().splitlines()]
    assert [reply['ok'] for reply in replies] == [False, True]


def test_cleanup_refuses_live_process_wrong_owner_and_symlink(tmp_path):
    root = tmp_path / 'owned'
    root.mkdir()
    (root / '.benchmark-owner').write_text('run')
    assert benchmark.removal_allowed(root, 'run', ['absent', 'reused'])
    assert not benchmark.removal_allowed(root, 'run', ['alive'])
    assert not benchmark.removal_allowed(root, 'run', ['zombie'])
    assert not benchmark.removal_allowed(root, 'other', [])
    link = tmp_path / 'alias'
    link.symlink_to(root, target_is_directory=True)
    assert not benchmark.removal_allowed(link, 'run', [])
    assert root.exists()


def test_worker_refuses_unowned_root_even_with_supplied_environment_token(tmp_path):
    root = tmp_path / 'raw-cdp'
    root.mkdir()
    with pytest.raises(ValueError):
        benchmark.validate_worker_root(root, 'raw-cdp', 'run')
    (tmp_path / '.benchmark-owner').write_text('run')
    benchmark.validate_worker_root(root, 'raw-cdp', 'run')
    with pytest.raises(ValueError):
        benchmark.validate_worker_root(root, 'openbrowser', 'run')


@pytest.mark.asyncio
@pytest.mark.parametrize('states', [
    [FileNotFoundError(), '', '12345\n/devtools/browser/owned'],
    ['', 'not-yet-a-port', '12345\n/devtools/browser/owned'],
])
async def test_startup_retries_absent_empty_and_partial_port_file(tmp_path, monkeypatch, states):
    arm, waits = mock_raw_startup(tmp_path, monkeypatch, states)
    try:
        await arm.start()
        assert waits == [.05, .05]
        assert arm.session == 'owned-session'
    finally:
        arm.log.close()


@pytest.mark.asyncio
async def test_empty_port_file_still_obeys_startup_deadline(tmp_path, monkeypatch):
    arm, waits = mock_raw_startup(tmp_path, monkeypatch, [''])
    clock = iter([0, 31])
    monkeypatch.setattr(benchmark, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    try:
        with pytest.raises(TimeoutError, match=r'^CDP startup$'):
            await arm.start()
        assert waits == []
    finally:
        arm.log.close()


def mock_raw_startup(tmp_path, monkeypatch, states):
    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.name, arm.root, arm.chrome, arm.token = 'raw-cdp', tmp_path, 'chrome', 'run'
    arm.mode = benchmark.WorkerMode.NAVIGATION
    arm.owned, arm.ownership = {}, lambda: []
    arm.proc = None
    monkeypatch.setattr(benchmark.subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(poll=lambda: None))
    port_file = tmp_path / 'profile/DevToolsActivePort'
    read_text = Path.read_text
    states = iter(states)

    def read(path, *args, **kwargs):
        if path != port_file:
            return read_text(path, *args, **kwargs)
        value = next(states)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(Path, 'read_text', read)
    version = json.dumps({'webSocketDebuggerUrl': 'ws://127.0.0.1:12345/devtools/browser/owned'})
    monkeypatch.setattr(benchmark, 'local_open', lambda *args, **kwargs: contextlib.closing(io.StringIO(version)))
    waits = []

    async def sleep(delay):
        waits.append(delay)
    monkeypatch.setattr(benchmark.asyncio, 'sleep', sleep)

    class Cdp:
        async def connect(self, url):
            assert url == 'ws://127.0.0.1:12345/devtools/browser/owned'

        async def send(self, method, params=None, session=None):
            return {'Target.createTarget': {'targetId': 'owned-target'},
                    'Target.attachToTarget': {'sessionId': 'owned-session'}}.get(method, {})
    monkeypatch.setattr(benchmark, 'Cdp', Cdp)
    return arm, waits


def test_external_paths_and_missing_optional_arms_are_explicit(tmp_path, monkeypatch, private_result_root):
    for key in ('NAV_BENCH_SOURCE', 'NAV_BENCH_CHROME', 'NAV_BENCH_LIGHTPANDA',
                'NAV_BENCH_AGENT_BROWSER', 'NAV_BENCH_BROWSER_USE_PYTHON', 'NAV_BENCH_OPENBROWSER_PYTHON'):
        monkeypatch.delenv(key, raising=False)
    source = tmp_path / 'source'
    (source / 'src/openbrowser').mkdir(parents=True)
    (source / 'src/openbrowser/__init__.py').touch()
    chrome = tmp_path / 'chrome'
    chrome.write_text('#!/bin/sh\nexit 0\n')
    chrome.chmod(0o700)
    args = benchmark.parser().parse_args(['--source', str(source), '--chrome', str(chrome),
                                         '--output', str(private_result_root)])
    benchmark.validate_config(args)
    states = benchmark.availability(args)
    assert states['raw-cdp']['state'] == 'configured'
    assert states['lightpanda']['state'] == 'unavailable'
    assert states['agent-browser']['state'] == 'unavailable'
    assert states['browser-use']['state'] == 'unavailable'


@pytest.mark.parametrize('extra', [['--samples', '0'], ['--delay', 'nan'], ['--delay', '-1'],
                                   ['--arms', 'raw-cdp', 'raw-cdp']])
def test_config_rejects_invalid_measurement_and_duplicate_arms(tmp_path, extra):
    args = benchmark.parser().parse_args(['--source', str(tmp_path), '--output', str(tmp_path / 'out'), *extra])
    with pytest.raises(ValueError):
        benchmark.validate_config(args)


def test_config_refuses_existing_output_and_user_profile_options(tmp_path, private_result_root):
    source = tmp_path / 'source'
    (source / 'src/openbrowser').mkdir(parents=True)
    (source / 'src/openbrowser/__init__.py').touch()
    chrome = tmp_path / 'chrome'
    chrome.write_text('#!/bin/sh\nexit 0\n')
    chrome.chmod(0o700)
    output = private_result_root
    output.mkdir(mode=0o700)
    args = benchmark.parser().parse_args(['--source', str(source), '--chrome', str(chrome), '--output', str(output)])
    with pytest.raises(ValueError, match=r'^--output must be a new directory$'):
        benchmark.validate_config(args)
    with pytest.raises(SystemExit):
        benchmark.parser().parse_args(['--profile', str(tmp_path)])


def runtime_environment(tmp_path):
    (tmp_path / '.benchmark-owner').write_text('run')
    root = tmp_path / 'raw-cdp'
    root.mkdir()
    return root, {
        'NAVIGATION_BENCHMARK_OWNER': 'run',
        'BH_RUNTIME_DIR': str(root / 'runtime'),
        'HOME': str(root / 'home'),
    }


class ProcessIdentity:
    def __init__(self, pid, environment, command=None):
        self.pid, self.environment = pid, environment
        self.command = command or ['private-runtime-process']
        self.descendants, self.signals = [], []
        self.created = 2.0

    def create_time(self):
        return self.created

    def cmdline(self):
        return self.command

    def environ(self):
        if self.environment is None:
            raise benchmark.psutil.AccessDenied(self.pid)
        return self.environment

    def children(self, recursive=False):
        return self.descendants

    def status(self):
        return benchmark.psutil.STATUS_RUNNING

    def terminate(self):
        self.signals.append('terminate')

    def kill(self):
        self.signals.append('kill')


def test_record_verifies_private_runtime_and_owner_marker(tmp_path):
    root, env = runtime_environment(tmp_path)
    identity = benchmark.record(ProcessIdentity(12, env), 'run')
    assert identity['owner_token'] == 'run'
    assert identity['owner_root'] == str(root)


@pytest.mark.parametrize('failure', ['token', 'runtime', 'marker', 'profile', 'environment_access'])
def test_record_does_not_stamp_unverified_process_as_owned(tmp_path, failure):
    root, env = runtime_environment(tmp_path)
    command = ['chrome', f'--user-data-dir={root / "profile"}']
    if failure == 'token':
        env['NAVIGATION_BENCHMARK_OWNER'] = 'different-owner'
    elif failure == 'runtime':
        env['BH_RUNTIME_DIR'] = str(tmp_path / 'somewhere-else')
    elif failure == 'marker':
        (tmp_path / '.benchmark-owner').unlink()
    elif failure == 'profile':
        command = ['chrome', f'--user-data-dir={tmp_path / "user-profile"}']
    else:
        env = None
    identity = benchmark.record(ProcessIdentity(12, env, command), 'run')
    assert identity['owner_token'] is None
    assert identity['ownership_error']


def test_descendant_cannot_borrow_verified_parent_ownership(tmp_path, monkeypatch):
    _, env = runtime_environment(tmp_path)
    parent = ProcessIdentity(12, env)
    child = ProcessIdentity(13, {})
    parent.descendants = [child]
    processes = {12: parent, 13: child}
    monkeypatch.setattr(benchmark.psutil, 'Process', processes.__getitem__)
    descendants = benchmark.descendant_identities(benchmark.record(parent, 'run'), 'run')
    assert len(descendants) == 1
    assert descendants[0]['owner_token'] is None
    assert benchmark.state(descendants[0]) == 'unknown'


@pytest.mark.asyncio
@pytest.mark.parametrize('changed', ['token', 'runtime', 'marker'])
async def test_signal_boundary_rechecks_each_process_ownership(tmp_path, monkeypatch, changed):
    _, env = runtime_environment(tmp_path)
    process = ProcessIdentity(12, env)
    saved = benchmark.record(process, 'run')
    if changed == 'token':
        env['NAVIGATION_BENCHMARK_OWNER'] = 'different-owner'
    elif changed == 'runtime':
        env['BH_RUNTIME_DIR'] = str(tmp_path / 'foreign-runtime')
    else:
        (tmp_path / '.benchmark-owner').unlink()
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: process)
    await benchmark.terminate_owned([saved], 'run', grace=0)
    assert process.signals == []


@pytest.mark.asyncio
async def test_kill_rechecks_ownership_after_terminate(tmp_path, monkeypatch):
    _, env = runtime_environment(tmp_path)
    process = ProcessIdentity(12, env)
    saved = benchmark.record(process, 'run')

    def terminate():
        process.signals.append('terminate')
        (tmp_path / '.benchmark-owner').unlink()
    monkeypatch.setattr(process, 'terminate', terminate)
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: process)
    identities = await benchmark.terminate_owned([saved], 'run', grace=0)
    assert process.signals == ['terminate']
    assert identities[0]['state'] not in ('absent', 'reused')


def test_worker_collection_checks_each_child_not_only_its_parent(tmp_path, monkeypatch):
    root, env = runtime_environment(tmp_path)
    parent = ProcessIdentity(benchmark.os.getpid(), env)
    owned = ProcessIdentity(12, env)
    foreign = ProcessIdentity(13, {})
    parent.descendants = [owned, foreign]
    processes = {parent.pid: parent, 12: owned, 13: foreign}
    monkeypatch.setattr(benchmark.psutil, 'Process', processes.__getitem__)
    arm = benchmark.Arm.__new__(benchmark.Arm)
    arm.root, arm.token, arm.agent_browser, arm.owned = root, 'run', None, {}
    captured = {identity['pid']: identity for identity in arm.ownership()}
    assert captured[12]['owner_token'] == 'run'
    assert captured[13]['owner_token'] is None
    assert benchmark.state(captured[13]) == 'unknown'


@pytest.mark.asyncio
async def test_unverified_live_process_is_not_signaled_or_reported_clean(tmp_path, monkeypatch):
    runtime_environment(tmp_path)
    process = ProcessIdentity(13, {})
    identity = benchmark.record(process, 'run')
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: process)
    identities = await benchmark.terminate_owned([identity], 'run', grace=0)
    assert process.signals == []
    assert identities[0]['state'] == 'unknown'
    result = verified_result()
    result['cleanup']['raw-cdp']['post_worker_exit_identities'] = identities
    assert any('cleanup identity proof is incomplete' in error for error in benchmark.result_errors(result))


def test_load_average_is_none_when_platform_has_no_capability(monkeypatch):
    monkeypatch.delattr(benchmark.os, 'getloadavg', raising=False)
    assert benchmark.load_average() is None


def test_load_average_preserves_available_values_and_handles_os_error(monkeypatch):
    monkeypatch.setattr(benchmark.os, 'getloadavg', lambda: (1.0, 2.0, 3.0), raising=False)
    assert benchmark.load_average() == (1.0, 2.0, 3.0)

    def unavailable():
        raise OSError('load average is unavailable')
    monkeypatch.setattr(benchmark.os, 'getloadavg', unavailable)
    assert benchmark.load_average() is None


def test_verify_rejects_false_summary_and_missing_attempts():
    sample = completed_sample('fast')
    result = {'metadata': {'samples_per_workload_arm': 1, 'source_unchanged': True},
              'availability': {'raw-cdp': {'state': 'ready'}}, 'samples': [sample],
              'warmup': [], 'cleanup': {'root_removed': True}, 'summary': {}}
    assert benchmark.result_errors(result)
    result['summary'] = benchmark.summarize(result['samples'], result['availability'], 1)
    assert any('not_run' in error for error in benchmark.result_errors(result))


def verified_result():
    samples = [completed_sample(kind) for kind in ('fast', 'delayed')]
    result = {
        'metadata': {'samples_per_workload_arm': 1, 'source_unchanged': True,
                     'source_manifest': {'source.py': 'hash'}, 'source_manifest_end': {'source.py': 'hash'}},
        'availability': {'raw-cdp': {'state': 'ready'}}, 'samples': samples, 'warmup': [completed_sample('fast')],
        'startup': {'raw-cdp': {'ok': True}},
        'cleanup': {'root_removed': True, 'raw-cdp': {'ok': True,
                    'worker_identity': {'state': 'absent'}, 'post_worker_exit_identities': [{'state': 'absent'}]}},
    }
    result['summary'] = benchmark.summarize(samples, result['availability'], 1)
    return result


def test_verifier_accepts_all_workloads_and_proven_cleanup():
    assert benchmark.result_errors(verified_result()) == []


@pytest.mark.parametrize('failure', [
    'missing', 'duplicate', 'failed', 'kind', 'index', 'unknown_arm',
    'title', 'side_effect', 'post_count', 'request_timer', 'checked_timer', 'elapsed',
])
def test_verifier_rejects_invalid_warmup(failure):
    result = verified_result()
    warmup = completed_sample('fast')
    result['warmup'] = [warmup]
    if failure == 'missing':
        result['warmup'] = []
    elif failure == 'duplicate':
        result['warmup'].append(completed_sample('fast'))
    elif failure == 'failed':
        warmup['outcome'] = 'failure'
    elif failure == 'kind':
        warmup['kind'] = 'delayed'
    elif failure == 'index':
        warmup['index'] = 1
    elif failure == 'unknown_arm':
        warmup['arm'] = 'unrequested-arm'
    elif failure == 'title':
        warmup['navigation_observation']['title'] = 'wrong-document'
    elif failure == 'side_effect':
        warmup['server']['effects'] = 2
    elif failure == 'post_count':
        warmup['server']['requests'].append(dict(warmup['server']['requests'][-1]))
    elif failure == 'request_timer':
        warmup['server']['requests'][0]['completed_ns'] = 201
    elif failure == 'checked_timer':
        warmup['checked_ns'] = 201
    else:
        warmup['elapsed_ns'] = 1
    assert any('warmup' in error for error in benchmark.result_errors(result))


def test_warmup_never_enters_measured_latency_summary():
    result = verified_result()
    warmup = completed_sample('fast')
    warmup.update(end_ns=10000, elapsed_ns=9900)
    result['warmup'] = [warmup]
    assert benchmark.result_errors(result) == []
    assert result['summary']['raw-cdp/fast']['median_ns'] == 100


def test_verifier_does_not_trust_cleanup_boolean_over_live_identity():
    result = verified_result()
    result['cleanup']['raw-cdp']['post_worker_exit_identities'][0]['state'] = 'alive'
    assert any('cleanup' in error for error in benchmark.result_errors(result))


def test_verifier_checks_saved_source_hashes_not_just_stability_boolean():
    result = verified_result()
    result['metadata']['source_manifest_end']['source.py'] = 'changed'
    assert any('source' in error for error in benchmark.result_errors(result))


def test_saved_artifact_verification_detects_tampered_raw_file(tmp_path):
    result = verified_result()
    (tmp_path / 'results.json').write_text(json.dumps(result))
    (tmp_path / 'samples.jsonl').write_text('raw sample stream\n')
    hashes = {name: benchmark.sha256(tmp_path / name) for name in ('results.json', 'samples.jsonl')}
    (tmp_path / 'sha256.json').write_text(json.dumps(hashes))
    assert benchmark.verify(tmp_path) == 0
    (tmp_path / 'samples.jsonl').write_text('changed\n')
    assert benchmark.verify(tmp_path) == 1


def test_worker_environment_does_not_inherit_user_browser_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv('AGENT_BROWSER_CDP', 'http://remote.example:9222')
    monkeypatch.setenv('OPENBROWSER_USER_DATA_DIR', '/user/profile')
    monkeypatch.setenv('BU_NAME', 'user-browser')
    monkeypatch.setenv('BH_HOME', '/user/browser-use')
    launch = make_worker_launch(tmp_path, log_directory=tmp_path / 'logs')
    env = benchmark.worker_environment(launch)
    assert 'AGENT_BROWSER_CDP' not in env
    assert 'OPENBROWSER_USER_DATA_DIR' not in env
    assert 'BU_NAME' not in env
    assert Path(env['BH_HOME']).is_relative_to(tmp_path)
    assert Path(env['HOME']).is_relative_to(tmp_path)
    assert env['NAVIGATION_BENCHMARK_OWNER'] == 'private-token'


def test_cli_rejects_result_root_outside_physical_temp_before_creating_output(tmp_path, monkeypatch):
    source = SCRIPT.parents[1]
    chrome = tmp_path / 'chrome'
    chrome.write_text('#!/bin/sh\necho test-version\n')
    chrome.chmod(0o700)
    output = tmp_path / 'navigation-results'
    monkeypatch.setattr(
        benchmark.asyncio, 'create_subprocess_exec',
        lambda *_args, **_kwargs: pytest.fail('worker launch ran for an invalid result root'),
    )

    code = benchmark.dispatch([
        '--source', str(source), '--output', str(output), '--chrome', str(chrome),
        '--arms', 'openbrowser', '--openbrowser-python', sys.executable,
    ])

    assert code == 2
    assert not output.exists()


@pytest.mark.parametrize('arm', ['openbrowser', 'raw-cdp', 'lightpanda', 'agent-browser', 'browser-use'])
def test_worker_launch_factory_selects_the_pinned_interpreter_for_each_arm(tmp_path, arm):
    candidate_python = Path('/candidate/bin/python')
    browser_use_python = Path('/browser-use/bin/python')
    launch = make_worker_launch(
        tmp_path,
        arm=arm,
        candidate_python=candidate_python,
        browser_use_python=browser_use_python,
        log_directory=tmp_path / 'logs',
    )

    expected = browser_use_python if arm == 'browser-use' else candidate_python
    assert launch.interpreter == expected


def test_worker_launch_factory_rejects_lightpanda_screenshot_mode(tmp_path):
    with pytest.raises(ValueError, match='screenshot'):
        make_worker_launch(
            tmp_path, arm='lightpanda', mode=benchmark.WorkerMode.SCREENSHOT,
            log_directory=tmp_path / 'logs',
        )


@pytest.mark.asyncio
async def test_launch_worker_creates_private_logs_and_records_starting_handle(tmp_path, monkeypatch):
    physical_tmp = Path('/tmp').resolve(strict=True)
    with tempfile.TemporaryDirectory(dir=physical_tmp) as runtime_name, \
            tempfile.TemporaryDirectory(dir=physical_tmp) as result_name:
        runtime_root = Path(runtime_name)
        result_root = Path(result_name)
        runtime_root.chmod(0o700)
        result_root.chmod(0o700)
        marker = runtime_root / '.benchmark-owner'
        marker.write_text('private-token')
        marker.chmod(0o600)
        log_directory = result_root / 'logs'
        log_directory.mkdir(mode=0o700)
        launch = benchmark.make_worker_launch(
            arm='openbrowser', mode=benchmark.WorkerMode.NAVIGATION,
            source_root=SCRIPT.parents[1], candidate_python=Path(sys.executable),
            browser_use_python=Path(sys.executable), runtime_root=runtime_root,
            worker_root=runtime_root / 'openbrowser', working_directory=runtime_root / 'openbrowser',
            log_directory=log_directory, log_prefix='openbrowser-fast', owner_token='private-token',
        )
        captured = {}

        class Process:
            pid = 81234

        async def create_process(*command, **kwargs):
            captured.update(command=command, kwargs=kwargs)
            return Process()

        identity = {
            'pid': 81234, 'created': 1234.5, 'cmdline': ['worker'],
            'owner_token': launch.owner_token, 'owner_root': str(launch.worker_root),
        }
        monkeypatch.setattr(benchmark.asyncio, 'create_subprocess_exec', create_process)
        monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: SimpleNamespace(pid=pid))
        monkeypatch.setattr(benchmark, 'record', lambda _process, _token: identity)

        handle = await benchmark.launch_worker(launch)

        assert handle.process.pid == 81234
        assert handle.worker_identity == identity
        assert handle.initial_owned is None
        assert handle.state is benchmark.WorkerState.STARTING
        assert captured['command'][0] == str(launch.interpreter)
        assert str(SCRIPT) in captured['command']
        assert captured['kwargs']['cwd'] == launch.working_directory
        assert captured['kwargs']['env']['NAVIGATION_BENCHMARK_OWNER'] == launch.owner_token
        stdout_path = log_directory / 'openbrowser-fast.stdout.log'
        stderr_path = log_directory / 'openbrowser-fast.stderr.log'
        assert stdout_path.is_file()
        assert stderr_path.is_file()
        assert stdout_path.stat().st_mode & 0o777 == 0o600
        assert stderr_path.stat().st_mode & 0o777 == 0o600
        handle.stdout_log.close()
        handle.stderr_log.close()


def make_worker_handle(tmp_path, initial_owned):
    class Input:
        def close(self):
            return None

    class Process:
        def __init__(self):
            self.pid = 81235
            self.stdin = Input()
            self.returncode = None

        async def wait(self):
            self.returncode = 0

    identity = {
        'pid': 81235, 'created': 1234.5, 'cmdline': ['worker'],
        'owner_token': 'private-token', 'owner_root': str(Path(tmp_path) / 'openbrowser'),
    }
    return benchmark.WorkerHandle(
        process=Process(), stdout_log=io.StringIO(), stderr_log=io.StringIO(),
        worker_identity=identity, initial_owned=initial_owned,
        owner_token='private-token', mode=benchmark.WorkerMode.NAVIGATION,
        state=benchmark.WorkerState.STARTING,
    )


@pytest.mark.asyncio
async def test_cleanup_worker_marks_unknown_start_inventory_incomplete(tmp_path, monkeypatch):
    handle = make_worker_handle(tmp_path, None)
    calls = []

    async def send(_process, request, _log, timeout=90):
        calls.append((request['op'], timeout))
        return {'ok': True, 'result': {'identities': [], 'errors': []}}

    monkeypatch.setattr(benchmark, 'send', send)
    monkeypatch.setattr(benchmark, 'state', lambda _identity: 'absent')

    result = await benchmark.cleanup_worker(handle)

    assert calls == [('stop', 60)]
    assert result['ok'] is False
    assert handle.state is benchmark.WorkerState.INCOMPLETE
    assert result['worker_identity']['state'] == 'absent'
    assert 'inventory' in result['error']


@pytest.mark.asyncio
async def test_cleanup_worker_is_idempotent_after_verified_stop(tmp_path, monkeypatch):
    handle = make_worker_handle(tmp_path, ())
    calls = []

    async def send(_process, request, _log, timeout=90):
        calls.append(request['op'])
        return {'ok': True, 'result': {'identities': [], 'errors': []}}

    monkeypatch.setattr(benchmark, 'send', send)
    monkeypatch.setattr(benchmark, 'state', lambda _identity: 'absent')

    first = await benchmark.cleanup_worker(handle)
    second = await benchmark.cleanup_worker(handle)

    assert first['ok'] is True
    assert second == first
    assert calls == ['stop']
    assert handle.state is benchmark.WorkerState.STOPPED


@pytest.mark.asyncio
async def test_cleanup_worker_accepts_stateful_stop_identity_records(tmp_path, monkeypatch):
    handle = make_worker_handle(tmp_path, ())
    child = {**handle.worker_identity, 'pid': 81236}

    async def send(_process, request, _log, timeout=90):
        return {'ok': True, 'result': {'identities': [{**child, 'state': 'absent'}], 'errors': []}}

    monkeypatch.setattr(benchmark, 'send', send)
    monkeypatch.setattr(benchmark, 'state', lambda _identity: 'absent')
    cleanup = await benchmark.cleanup_worker(handle)

    assert cleanup['ok'] is True, cleanup
    assert cleanup['post_worker_exit_identities'] == [{**child, 'state': 'absent'}]


def test_worker_state_transition_returns_ready_after_failed_navigation_sample():
    transition = getattr(benchmark, 'transition_worker_state', None)
    assert callable(transition), 'worker state transition table is not defined'

    state = 'created'
    for event in ('start', 'start_succeeded', 'valid_operation', 'navigation_result'):
        state = transition(state, event)

    assert state == 'ready'


def test_worker_protocol_error_blocks_later_operations_but_keeps_stop(monkeypatch):
    instances = []

    class FakeArm:
        def __init__(self, _args):
            self.calls = []
            instances.append(self)

        async def start(self, _mode=None):
            self.calls.append('start')
            return {'owned': []}

        async def trial(self, _request):
            self.calls.append('trial')
            return {'outcome': 'failure', 'exception': 'fixture assertion failed'}

        async def stop(self):
            self.calls.append('stop')
            return {'errors': [], 'identities': []}

    requests = [
        {'op': 'start'},
        {'op': 'unknown_operation'},
        {'op': 'trial', 'url': 'http://127.0.0.1/fixture', 'nonce': 'sample'},
        {'op': 'stop'},
    ]
    stdout = io.StringIO()
    monkeypatch.setattr(benchmark, 'Arm', FakeArm)
    monkeypatch.setattr(benchmark.sys, 'stdin', io.StringIO(''.join(json.dumps(item) + '\n' for item in requests)))
    monkeypatch.setattr(benchmark.sys, 'stdout', stdout)

    asyncio.run(benchmark.worker(SimpleNamespace(
        worker='openbrowser', worker_mode=benchmark.WorkerMode.NAVIGATION,
    )))

    replies = [json.loads(line[len(benchmark.PREFIX):]) for line in stdout.getvalue().splitlines()]
    assert [reply['ok'] for reply in replies] == [True, False, False, True]
    assert instances[0].calls == ['start', 'stop']


def _assert_published_paired_statistics(data):
    assert data['schema_version'] == 2
    for kind, comparison in data['paired_comparison'].items():
        baseline = data['paired']['baseline'][kind]['elapsed_ns']
        candidate = data['paired']['candidate'][kind]['elapsed_ns']
        assert len(baseline) == len(candidate) >= 5
        differences = [old - new for old, new in zip(baseline, candidate)]
        assert comparison['paired_savings_ns'] == differences
        assert comparison['difference_of_medians_ns'] == statistics.median(baseline) - statistics.median(candidate)
        assert comparison['median_paired_saving_ns'] == statistics.median(differences)
        assert 'median_saving_ns' not in comparison


def test_published_paired_statistics_match_unrounded_samples():
    files = sorted(SCRIPT.parent.glob('navigation_results_*.json'))
    assert files
    for path in files:
        _assert_published_paired_statistics(json.loads(path.read_text()))


def test_paired_statistics_reject_difference_of_medians_as_median_paired_saving():
    path = SCRIPT.parent / 'navigation_results_2026-11-15.json'
    data = json.loads(path.read_text())
    comparison = data['paired_comparison']['fast']
    comparison['median_paired_saving_ns'] = comparison['difference_of_medians_ns']
    with pytest.raises(AssertionError):
        _assert_published_paired_statistics(data)


def test_paired_statistics_reject_legacy_schema_version():
    path = SCRIPT.parent / 'navigation_results_2026-11-15.json'
    data = json.loads(path.read_text())
    data['schema_version'] = 1
    with pytest.raises(AssertionError):
        _assert_published_paired_statistics(data)


def test_normal_dispatch_preserves_default_schedule_and_artifacts(tmp_path, monkeypatch, private_result_root):
    dispatch = getattr(benchmark, 'dispatch', None)
    assert callable(dispatch), 'normal command dispatcher is missing'

    def executable(name):
        path = tmp_path / name
        path.write_text('#!/bin/sh\necho test-version\n')
        path.chmod(0o700)
        return str(path)

    source = SCRIPT.parents[1]
    chrome = executable('chrome')
    lightpanda = executable('lightpanda')
    agent_browser = executable('agent-browser')
    output = private_result_root
    processes = []
    schedule = []

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid
            self.stdin = SimpleNamespace(close=lambda: None)
            self.returncode = 0

    async def create_process(*command, **_kwargs):
        process = FakeProcess(1000 + len(processes))
        processes.append(process)
        return process

    async def send(_process, request, _log, timeout=90):
        if request['op'] == 'start':
            return {'ok': True, 'result': {'owned': []}}
        schedule.append((request['arm'], request['kind'], request['index']))
        sample = {key: value for key, value in request.items() if key != 'op'}
        sample.update(start_ns=100, end_ns=200, checked_ns=195, elapsed_ns=100, outcome='success')
        return {'ok': True, 'result': sample}

    def record(process, token):
        return {'pid': process.pid, 'created': float(process.pid), 'cmdline': ['fake-worker'],
                'owner_token': token, 'owner_root': '/private/tmp/fake-runtime'}

    async def cleanup(handle):
        return {'ok': True, 'worker_identity': {**handle.worker_identity, 'state': 'absent'},
                'post_worker_exit_identities': [], 'fallback_used': False}

    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: SimpleNamespace(pid=pid))
    monkeypatch.setattr(benchmark, 'record', record)
    monkeypatch.setattr(benchmark, 'source_manifest', lambda _source: {'src/openbrowser/test.py': 'a' * 64})
    monkeypatch.setattr(benchmark, 'git_value', lambda _source, *args: '' if args[0] == 'status' else '0' * 40)
    monkeypatch.setattr(benchmark, 'executable_metadata', lambda path: {'path': path, 'sha256': 'a' * 64, 'version': 'test'})
    monkeypatch.setattr(benchmark, 'workload_errors', lambda *_args: [])
    monkeypatch.setattr(benchmark, 'result_errors', lambda _result: [])
    monkeypatch.setattr(benchmark, 'cleanup_worker', cleanup)
    monkeypatch.setattr(benchmark.asyncio, 'create_subprocess_exec', create_process)

    async def fake_send_recording(process, request, log, timeout=90):
        return await send(process, request, log, timeout)

    monkeypatch.setattr(benchmark, 'send', fake_send_recording)
    code = dispatch([
        '--source', str(source), '--output', str(output), '--chrome', chrome,
        '--lightpanda', lightpanda, '--agent-browser', agent_browser,
        '--browser-use-python', sys.executable, '--openbrowser-python', sys.executable,
    ])

    expected = [(name, 'fast', 0) for name in benchmark.ARMS]
    for index in range(5):
        ordered = list(benchmark.ARMS[index:]) + list(benchmark.ARMS[:index])
        for kind in ('fast', 'delayed'):
            expected.extend((name, kind, index) for name in ordered)
    assert code == 0
    assert len(processes) == len(benchmark.ARMS)
    assert schedule == expected
    result = json.loads((output / 'results.json').read_text())
    assert result['metadata']['samples_per_workload_arm'] == 5
    assert result['metadata']['delay_seconds'] == .15
    assert set(result) == {
        'metadata', 'availability', 'startup', 'warmup', 'samples', 'cleanup', 'summary', 'verification_errors',
    }
    assert len(result['warmup']) == len(benchmark.ARMS)
    assert len(result['samples']) == 2 * 5 * len(benchmark.ARMS)
    assert set(result['availability']) == set(benchmark.ARMS)
    expected_files = {
        'results.json', 'samples.jsonl', 'sha256.json',
        *(f'{name}.{stream}.log' for name in benchmark.ARMS for stream in ('stdout', 'stderr')),
    }
    assert {path.name for path in output.iterdir()} == expected_files

    def has_profile_field(value):
        if isinstance(value, dict):
            return 'profile' in value or any(has_profile_field(item) for item in value.values())
        if isinstance(value, list):
            return any(has_profile_field(item) for item in value)
        return False

    assert not has_profile_field(result)
