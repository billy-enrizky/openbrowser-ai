import hashlib
import importlib.util
import asyncio
import contextlib
import io
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/navigation_compare.py'
spec = importlib.util.spec_from_file_location('navigation_compare', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


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
async def test_coordinator_stop_failure_still_reaps_only_verified_owned_identities(monkeypatch):
    worker = {'pid': 12, 'created': 2.0, 'cmdline': ['worker'], 'owner_token': 'run'}
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
    result = await benchmark.cleanup_worker(Process(), None, worker, [browser], 'run')
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
    args = SimpleNamespace(worker=name, run_root=str(root), source=str(SCRIPT.parents[1]),
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


def test_external_paths_and_missing_optional_arms_are_explicit(tmp_path, monkeypatch):
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
                                         '--output', str(tmp_path / 'new-results')])
    benchmark.validate_config(args)
    states = benchmark.availability(args)
    assert states['raw-cdp']['state'] == 'configured'
    assert states['lightpanda']['state'] == 'unavailable'
    assert states['agent-browser']['state'] == 'unavailable'
    assert states['browser-use']['state'] == 'unavailable'


def test_executable_metadata_accepts_an_arm_specific_version_command(tmp_path):
    binary = tmp_path / 'lightpanda'
    binary.write_text('#!/bin/sh\nif [ "$1" = "version" ]; then echo 1.0.4; else exit 1; fi\n')
    binary.chmod(0o700)
    assert benchmark.executable_metadata(str(binary), ('version',))['version'] == '1.0.4'
    assert 'unavailable' in benchmark.executable_metadata(str(binary))['version']


def test_config_requires_pins_for_a_pinned_claim(tmp_path):
    source = tmp_path / 'source'
    (source / 'src/openbrowser').mkdir(parents=True)
    (source / 'src/openbrowser/__init__.py').touch()
    base = ['--source', str(source), '--output', str(tmp_path / 'out'), '--claim-tier', 'pinned']
    with pytest.raises(ValueError, match='pinned claim requires --pins'):
        benchmark.validate_config(benchmark.parser().parse_args(base))

    pins = tmp_path / 'pins.json'
    pins.write_text(json.dumps({'captured_utc': '2026-11-20T00:00:00Z', 'pins': {'chrome': {'version': '130.0.6723.58'}}}))
    benchmark.validate_config(benchmark.parser().parse_args([*base, '--pins', str(pins)]))

    broken = tmp_path / 'broken.json'
    broken.write_text(json.dumps({'pins': {'chrome': {}}}))
    with pytest.raises(ValueError, match='version or source_commit'):
        benchmark.validate_config(benchmark.parser().parse_args([*base, '--pins', str(broken)]))


@pytest.mark.parametrize('extra', [['--samples', '0'], ['--delay', 'nan'], ['--delay', '-1'],
                                   ['--arms', 'raw-cdp', 'raw-cdp']])
def test_config_rejects_invalid_measurement_and_duplicate_arms(tmp_path, extra):
    args = benchmark.parser().parse_args(['--source', str(tmp_path), '--output', str(tmp_path / 'out'), *extra])
    with pytest.raises(ValueError):
        benchmark.validate_config(args)


def test_config_refuses_existing_output_and_user_profile_options(tmp_path):
    source = tmp_path / 'source'
    (source / 'src/openbrowser').mkdir(parents=True)
    (source / 'src/openbrowser/__init__.py').touch()
    chrome = tmp_path / 'chrome'
    chrome.write_text('#!/bin/sh\nexit 0\n')
    chrome.chmod(0o700)
    output = tmp_path / 'existing-results'
    output.mkdir()
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
                     'source_manifest': {'source.py': 'hash'}, 'source_manifest_end': {'source.py': 'hash'},
                     'claim_tier': 'historical', 'requested_arms': ['raw-cdp'],
                     'claim_scope': benchmark.claim_scope(),
                     'scope': 'Local native warm-operation diagnostic. No LLM, cloud, or Docker workflow.',
                     'performance_verdict': 'inconclusive: different engines, IPC and default wait semantics; no limiter profile'},
        'availability': {'raw-cdp': {'state': 'ready'}}, 'samples': samples, 'warmup': [completed_sample('fast')],
        'startup': {'raw-cdp': {'ok': True}},
        'cleanup': {'root_removed': True, 'raw-cdp': {'ok': True,
                    'worker_identity': {'state': 'absent'}, 'post_worker_exit_identities': [{'state': 'absent'}]}},
    }
    result['summary'] = benchmark.summarize(samples, result['availability'], 1)
    return result


def pinned_result():
    result = verified_result()
    result['metadata'].update({
        'claim_tier': 'pinned',
        'requested_arms': ['raw-cdp'],
        'executables': {'chrome': {'version': 'Google Chrome 130.0.6723.58'}},
        'source_commit': 'a9022ad43db84fa8f37c9e77bb767ad5c57f5132',
        'pins': {'captured_utc': '2026-11-20T00:00:00Z', 'pins': {'chrome': {'version': '130.0.6723.58'}}},
    })
    return result


def test_pinned_claim_accepts_matching_pins():
    result = pinned_result()
    assert benchmark.claim_errors(result) == []
    assert benchmark.result_errors(result) == []


def test_pinned_claim_requires_a_pin_for_each_ready_arm():
    result = pinned_result()
    result['metadata']['pins'] = {'captured_utc': '2026-11-20T00:00:00Z', 'pins': {}}
    assert any('raw-cdp/chrome' in error for error in benchmark.claim_errors(result))


def test_pinned_claim_rejects_a_version_mismatch():
    result = pinned_result()
    result['metadata']['executables']['chrome']['version'] = 'Google Chrome 129.0.0.1'
    assert any('does not match pin' in error for error in benchmark.claim_errors(result))
    assert any('does not match pin' in error for error in benchmark.result_errors(result))


def test_pinned_claim_rejects_an_invalid_capture_time():
    result = pinned_result()
    result['metadata']['pins']['captured_utc'] = 'unknown'
    assert any('captured_utc' in error for error in benchmark.claim_errors(result))


def test_pinned_claim_rejects_a_missing_recorded_version():
    result = pinned_result()
    result['metadata']['executables']['chrome'] = {}
    assert any('recorded version is missing' in error for error in benchmark.claim_errors(result))


def test_captured_utc_requires_an_aware_utc_timestamp():
    assert benchmark.valid_timestamp('2026-11-20T00:00:00Z')
    assert benchmark.valid_timestamp('2026-11-20T00:00:00+00:00')
    assert not benchmark.valid_timestamp('2026-11-20T00:00:00')
    assert not benchmark.valid_timestamp('2026-11-20')
    assert not benchmark.valid_timestamp('2026-11-20T00:00:00+05:00')
    assert not benchmark.valid_timestamp('unknown')


def test_version_matches_full_tokens_only():
    assert benchmark.version_matches('1.0.6', '1.0.6')
    assert benchmark.version_matches('130.0.6723.58', 'Google Chrome 130.0.6723.58')
    assert not benchmark.version_matches('1.0.6', '1.0.6rc1')
    assert not benchmark.version_matches('1.0.6', '1.0.6-rc1')
    assert not benchmark.version_matches('130.0.6723.58', '130.0.6723.581')


def test_version_match_is_exact_and_preserves_suffixes():
    result = pinned_result()
    result['metadata']['executables']['chrome']['version'] = 'Google Chrome 130.0.6723.581'
    assert any('does not match pin' in error for error in benchmark.claim_errors(result))

    result = verified_result()
    result['availability'] = {'browser-use': {'state': 'ready'}}
    result['metadata'].update({
        'claim_tier': 'pinned', 'requested_arms': ['browser-use'],
        'executables': {'chrome': {'version': 'Google Chrome 130.0.6723.58'}},
        'pins': {'captured_utc': '2026-11-20T00:00:00Z', 'pins': {'browser-use': {'version': '1.0.6'}, 'chrome': {'version': '130.0.6723.58'}}},
    })
    result['startup'] = {'browser-use': {'result': {'loaded_packages': {'browser-use': '1.0.6'}}}}
    assert benchmark.claim_errors(result) == []
    result['startup']['browser-use']['result']['loaded_packages']['browser-use'] = '1.0.6rc1'
    assert any('does not match pin' in error for error in benchmark.claim_errors(result))


def test_pinned_claim_reconciles_requested_arms_with_availability():
    result = pinned_result()
    result['metadata']['requested_arms'] = ['raw-cdp']
    result['availability'] = {'raw-cdp': {'state': 'ready'}, 'browser-use': {'state': 'ready'}}
    assert any('requested_arms does not match availability' in error for error in benchmark.claim_errors(result))


def test_pinned_claim_falls_back_to_availability_when_requested_arms_is_missing():
    result = pinned_result()
    result['metadata'].pop('requested_arms')
    assert benchmark.claim_errors(result) == []


def test_legacy_claim_statements_are_checked_without_a_claim_scope():
    result = verified_result()
    result['metadata'].pop('claim_scope')
    assert benchmark.claim_errors(result) == []
    result['metadata']['performance_verdict'] = 'openbrowser wins'
    assert benchmark.claim_errors(result)
    result['metadata']['performance_verdict'] = 'inconclusive: OpenBrowser wins universally'
    assert benchmark.claim_errors(result)
    result['metadata'].pop('performance_verdict')
    result['metadata']['scope'] = 'production winner'
    assert benchmark.claim_errors(result)


def test_pinned_lightpanda_claim_does_not_require_chrome():
    result = verified_result()
    result['availability'] = {'lightpanda': {'state': 'ready'}}
    result['metadata'].update({
        'claim_tier': 'pinned', 'requested_arms': ['lightpanda'],
        'executables': {'lightpanda': {'version': 'Lightpanda 1.0.4'}},
        'pins': {'captured_utc': '2026-11-20T00:00:00Z', 'pins': {'lightpanda': {'version': '1.0.4'}}},
    })
    assert benchmark.claim_errors(result) == []


def test_openbrowser_source_pin_requires_a_clean_tree_and_a_real_prefix():
    def result(commit='a9022ad43db84fa8f37c9e77bb767ad5c57f5132', pin='a9022ad4', dirty=''):
        value = verified_result()
        value['availability'] = {'openbrowser': {'state': 'ready'}}
        value['metadata'].update({
            'claim_tier': 'pinned', 'requested_arms': ['openbrowser'],
            'source_commit': commit, 'source_dirty': dirty,
            'executables': {'chrome': {'version': 'Google Chrome 130.0.6723.58'}},
            'pins': {'captured_utc': '2026-11-20T00:00:00Z',
                     'pins': {'openbrowser': {'source_commit': pin},
                              'chrome': {'version': '130.0.6723.58'}}},
        })
        return value

    assert benchmark.claim_errors(result()) == []
    assert any('at least 7' in error for error in benchmark.claim_errors(result(pin='a902')))
    assert any('not verified clean' in error for error in benchmark.claim_errors(result(dirty=' M src/x.py')))
    assert any('does not match pin' in error for error in benchmark.claim_errors(
        result(commit='b0000000000000000000000000000000000000000')))
    version_only = result()
    version_only['metadata']['pins']['pins']['openbrowser'] = {'version': '0.1.57'}
    assert any('must use source_commit' in error for error in benchmark.claim_errors(version_only))
    dirty = result(dirty=' M src/x.py')
    assert any('was not verified clean' in error for error in benchmark.claim_errors(dirty))
    unverified = result(dirty=None)
    assert any('was not verified clean' in error for error in benchmark.claim_errors(unverified))


def test_historical_run_may_retain_pins_without_enforcing_them():
    result = verified_result()
    result['metadata']['pins'] = {'captured_utc': '2026-11-20T00:00:00Z', 'pins': {'chrome': {'version': '999.0.0'}}}
    assert benchmark.claim_errors(result) == []


def test_legacy_historical_artifact_without_claim_fields_is_still_verifiable():
    result = verified_result()
    for key in ('claim_tier', 'claim_scope', 'requested_arms', 'scope', 'performance_verdict'):
        result['metadata'].pop(key, None)
    assert benchmark.result_errors(result) == []


def test_historical_claim_does_not_require_pins():
    result = verified_result()
    assert result['metadata']['claim_tier'] == 'historical'
    assert benchmark.claim_errors(result) == []


@pytest.mark.parametrize('change', [
    {'publishable_to_comparison_page': True},
    {'cross_tool_ranking': 'openbrowser wins'},
    {'lane': 'all-surfaces'},
    {'not_measured': []},
    {'measured': ['CodeAgent task success']},
])
def test_claim_scope_cannot_be_widened(change):
    result = verified_result()
    result['metadata']['claim_scope'] = {**benchmark.claim_scope(), **change}
    assert benchmark.claim_errors(result)


def test_comparison_page_does_not_cite_the_native_navigation_lane():
    page = (Path(__file__).resolve().parents[1] / 'docs/comparison.mdx').read_text()
    assert 'navigation_compare' not in page
    assert 'navigation_results' not in page


def test_checked_in_pin_example_matches_the_schema():
    example = Path(__file__).resolve().parents[1] / 'benchmarks/navigation_pins.example.json'
    pins = benchmark.load_pins(example)
    assert set(pins['pins']) == {'lightpanda', 'agent-browser', 'browser-use', 'chrome', 'openbrowser'}
    assert pins['pins']['browser-use']['version'] == '1.0.6'
    assert pins['pins']['openbrowser']['source_commit']


def test_verify_cross_checks_the_retained_pin_snapshot(tmp_path):
    pins = {'captured_utc': '2026-11-20T00:00:00Z', 'pins': {'chrome': {'version': '130.0.6723.58'}}}
    raw = json.dumps(pins).encode()
    (tmp_path / 'pins.json').write_bytes(raw)
    result = {'metadata': {'pins': pins, 'pins_sha256': hashlib.sha256(raw).hexdigest()}}
    assert benchmark.pin_snapshot_errors(tmp_path, result) == []
    (tmp_path / 'pins.json').write_text('{"pins": {}}')
    assert benchmark.pin_snapshot_errors(tmp_path, result)


def test_pinned_config_requires_a_capture_time_before_measurement(tmp_path):
    source = tmp_path / 'source'
    (source / 'src/openbrowser').mkdir(parents=True)
    (source / 'src/openbrowser/__init__.py').touch()
    pins = tmp_path / 'pins.json'
    pins.write_text(json.dumps({'pins': {'chrome': {'version': '130.0.6723.58'}}}))
    args = benchmark.parser().parse_args([
        '--source', str(source), '--output', str(tmp_path / 'out'),
        '--claim-tier', 'pinned', '--pins', str(pins),
    ])
    with pytest.raises(ValueError, match='captured_utc'):
        benchmark.validate_config(args)


def test_pin_file_requires_a_pins_object_and_an_identity(tmp_path):
    good = tmp_path / 'good.json'
    good.write_text(json.dumps({'pins': {'browser-use': {'version': '1.0.6'}}}))
    assert benchmark.load_pins(good)['pins']['browser-use']['version'] == '1.0.6'

    missing = tmp_path / 'missing.json'
    missing.write_text(json.dumps({}))
    with pytest.raises(ValueError, match='pins object'):
        benchmark.load_pins(missing)

    empty = tmp_path / 'empty.json'
    empty.write_text(json.dumps({'pins': {'browser-use': {}}}))
    with pytest.raises(ValueError, match='version or source_commit'):
        benchmark.load_pins(empty)

    bad_time = tmp_path / 'bad-time.json'
    bad_time.write_text(json.dumps({'captured_utc': 'unknown', 'pins': {'browser-use': {'version': '1.0.6'}}}))
    with pytest.raises(ValueError, match='ISO 8601'):
        benchmark.load_pins(bad_time)


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
    env = benchmark.worker_environment(tmp_path, 'private-token', tmp_path / 'source')
    assert 'AGENT_BROWSER_CDP' not in env
    assert 'OPENBROWSER_USER_DATA_DIR' not in env
    assert 'BU_NAME' not in env
    assert Path(env['BH_HOME']).is_relative_to(tmp_path)
    assert Path(env['HOME']).is_relative_to(tmp_path)
    assert env['NAVIGATION_BENCHMARK_OWNER'] == 'private-token'


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
