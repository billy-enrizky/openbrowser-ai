import importlib.util
import asyncio
import json
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
async def test_owned_cleanup_escalates_only_after_rechecking_identity(monkeypatch):
    calls = []
    saved = {'pid': 12, 'created': 2.0, 'cmdline': ['owned-browser'], 'owner_token': 'run'}

    class Process:
        pid = 12

        def create_time(self):
            return 2.0

        def cmdline(self):
            return ['owned-browser']

        def terminate(self):
            calls.append('terminate')

        def kill(self):
            calls.append('kill')

    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: Process())
    monkeypatch.setattr(benchmark, 'state', lambda identity: 'absent' if 'kill' in calls else 'alive')
    await benchmark.terminate_owned([saved], 'run', grace=0)
    assert calls == ['terminate', 'kill']

    calls.clear()
    monkeypatch.setattr(Process, 'create_time', lambda self: 2.0 if not calls else 3.0)
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


def test_external_paths_and_missing_optional_arms_are_explicit(tmp_path):
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


@pytest.mark.parametrize('extra', [['--samples', '0'], ['--delay', 'nan'], ['--delay', '-1'],
                                   ['--arms', 'raw-cdp', 'raw-cdp']])
def test_config_rejects_invalid_measurement_and_duplicate_arms(tmp_path, extra):
    args = benchmark.parser().parse_args(['--source', str(tmp_path), '--output', str(tmp_path / 'out'), *extra])
    with pytest.raises(ValueError):
        benchmark.validate_config(args)


def test_config_refuses_existing_output_and_user_profile_options(tmp_path):
    args = benchmark.parser().parse_args(['--source', str(tmp_path), '--output', str(tmp_path)])
    with pytest.raises(ValueError):
        benchmark.validate_config(args)
    with pytest.raises(SystemExit):
        benchmark.parser().parse_args(['--profile', str(tmp_path)])


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
        'availability': {'raw-cdp': {'state': 'ready'}}, 'samples': samples, 'warmup': [],
        'startup': {'raw-cdp': {'ok': True}},
        'cleanup': {'root_removed': True, 'raw-cdp': {'ok': True,
                    'worker_identity': {'state': 'absent'}, 'post_worker_exit_identities': [{'state': 'absent'}]}},
    }
    result['summary'] = benchmark.summarize(samples, result['availability'], 1)
    return result


def test_verifier_accepts_all_workloads_and_proven_cleanup():
    assert benchmark.result_errors(verified_result()) == []


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
