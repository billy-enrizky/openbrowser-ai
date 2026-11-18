import asyncio
import cProfile
import hashlib
import importlib.util
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks/navigation_compare.py'
spec = importlib.util.spec_from_file_location('navigation_compare_profile_tests', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def profile_function(name):
    function = getattr(benchmark, name, None)
    assert callable(function), f'{name} is not implemented yet'
    return function


def run_cli(arguments, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


def make_executable(path, content):
    path.write_text(content)
    path.chmod(0o700)
    return path


def profile_output_path(result_root, workload='fast', run_id='a' * 32):
    return Path(result_root) / 'profile' / workload / run_id


@pytest.fixture
def profile_parent():
    with tempfile.TemporaryDirectory(
        prefix='openbrowser-profile-parent-',
        dir=Path('/tmp').resolve(),
    ) as directory:
        parent = Path(directory)
        parent.chmod(0o700)
        profile_directory = parent / 'profile'
        profile_directory.mkdir(mode=0o700)
        profile_directory.chmod(0o700)
        for workload in ('fast', 'delayed'):
            output_parent = profile_directory / workload
            output_parent.mkdir(mode=0o700, parents=True)
            output_parent.chmod(0o700)
        yield parent


def private_cli_environment(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    return {
        **{key: value for key, value in os.environ.items() if not key.startswith(('AWS_', 'NAV_BENCH_'))},
        'HOME': str(home),
        'XDG_CONFIG_HOME': str(home / 'config'),
        'XDG_CACHE_HOME': str(home / 'cache'),
        'AWS_EC2_METADATA_DISABLED': 'true',
    }


def dirty_profile_source(tmp_path, change):
    source = tmp_path / 'source'
    package = source / 'src' / 'openbrowser'
    package.mkdir(parents=True)
    initializer = package / '__init__.py'
    initializer.write_text('__version__ = "0.1.56"\n')
    subprocess.run(['git', 'init', '--quiet', str(source)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(source), 'add', 'src/openbrowser/__init__.py'], check=True)
    subprocess.run(
        ['git', '-C', str(source), '-c', 'user.name=Profile tests', '-c',
         'user.email=profile-tests@example.test', 'commit', '--quiet', '-m', 'initial source'],
        check=True,
        capture_output=True,
    )
    if change == 'tracked':
        initializer.write_text('__version__ = "changed"\n')
    elif change == 'untracked':
        (package / 'untracked.py').write_text('value = 1\n')
    else:
        raise ValueError(f'unsupported test change: {change}')
    return source


def current_manifest(source, python, chrome):
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        'fetched_at_utc': '2026-11-16T00:00:00+00:00',
        'latest_versions': {'chrome_for_testing': 'Chrome for Testing 1.2.3', 'openbrowser-ai': '0.1.56'},
        'installed_versions': {'chrome_for_testing': 'Chrome for Testing 1.2.3', 'openbrowser-ai': '0.1.56'},
        'installed_executables': {
            'chrome_for_testing': {'version': 'Chrome for Testing 1.2.3', 'sha256': digest(chrome)},
        },
        'python_interpreters': {
            'candidate_openbrowser': {'version': sys.version.split()[0], 'sha256': digest(python)},
        },
        'source': {'candidate_commit': benchmark.git_value(source, 'rev-parse', 'HEAD')},
        'harness': {'sha256': digest(SCRIPT)},
    }


def run_arguments(source, python, chrome, manifest, output, *, result_root, workload='fast'):
    return [
        '--source', str(source),
        '--openbrowser-python', str(python),
        '--chrome', str(chrome),
        '--version-manifest', str(manifest),
        '--workload', workload,
        '--result-root', str(result_root),
        '--output', str(output),
    ]


def profile_pair_request(workload='delayed'):
    return {
        'op': 'profile_pair',
        'workload': workload,
        'fixture_port': 12345,
        'warmup': {'arm': 'openbrowser', 'kind': 'fast', 'index': 0, 'nonce': 'warmup-nonce'},
        'target': {'arm': 'openbrowser', 'kind': workload, 'index': 0, 'nonce': 'target-nonce'},
    }


def successful_trial(request):
    nonce, url = request['nonce'], request['url']
    observation = {
        'url': url,
        'title': f'fixture-{nonce}',
        'readyState': 'complete',
        'marker': nonce,
        'loaded': nonce,
        'scriptDone': nonce,
    }
    requests = [
        {'path': '/fixture', 'method': 'GET', 'nonce': nonce, 'status': 200,
         'received_ns': 105, 'completed_ns': 106},
    ]
    if request['kind'] == 'delayed':
        requests.append({'path': '/slow.js', 'method': 'GET', 'nonce': nonce, 'status': 200,
                         'received_ns': 107, 'completed_ns': 108})
    requests.append({'path': '/effect', 'method': 'POST', 'nonce': nonce, 'status': 200,
                     'received_ns': 150, 'completed_ns': 160})
    return {
        **request,
        'start_ns': 100,
        'navigation_return_ns': 110,
        'checked_ns': 190,
        'end_ns': 200,
        'elapsed_ns': 100,
        'outcome': 'success',
        'navigation_observation': observation,
        'effect_observation': {'field': nonce, 'response': {'count': 1, 'nonce': nonce, 'payload': nonce}},
        'final_observation': {**observation, 'field': nonce},
        'server': {'effects': 1, 'requests': requests},
    }


class FakeArm:
    def __init__(self, *, fail_warmup=False, fail_target=False, timeout_target=False, block_target=False):
        self.name = 'openbrowser'
        self.trials = []
        self.profile_states = []
        self.profile_probe = lambda: False
        self.fail_warmup = fail_warmup
        self.fail_target = fail_target
        self.timeout_target = timeout_target
        self.block_target = block_target
        self.target_started = asyncio.Event()
        self.stopped = False

    async def start(self):
        return {
            'version': {'product': 'Chrome/1.2.3'},
            'owned': [],
            'packages': {'psutil': '7.2.0', 'websockets': '15.0.1'},
            'loaded_packages': {'openbrowser-ai': '0.1.56'},
        }

    async def trial(self, request):
        self.trials.append(request)
        self.profile_states.append(self.profile_probe())
        if request['kind'] != 'fast' or request['nonce'] == 'target-nonce':
            self.target_started.set()
            if self.block_target:
                await asyncio.Event().wait()
            if self.fail_target:
                raise RuntimeError('target trial failed')
            if self.timeout_target:
                return {**request, 'start_ns': 100, 'end_ns': 200, 'elapsed_ns': 100,
                        'outcome': 'failure', 'timed_out': True, 'exception': 'TimeoutError: trial timed out'}
        if self.fail_warmup and request['nonce'] == 'warmup-nonce':
            return {**request, 'start_ns': 100, 'end_ns': 200, 'elapsed_ns': 100,
                    'outcome': 'failure', 'exception': 'navigation failed'}
        return successful_trial(request)

    async def stop(self):
        self.stopped = True
        return {'errors': [], 'identities': []}


class ProfileSpy:
    def __init__(self, profile_type):
        self.real = profile_type()
        self.enabled = False

    def enable(self):
        self.enabled = True
        self.real.enable()

    def disable(self):
        self.enabled = False
        self.real.disable()

    def dump_stats(self, path):
        self.real.dump_stats(path)


def install_profile_spy(monkeypatch, arm):
    real_profile_type = cProfile.Profile
    spy = ProfileSpy(real_profile_type)
    monkeypatch.setattr(cProfile, 'Profile', lambda: spy)
    arm.profile_probe = lambda: spy.enabled
    return spy


def worker_messages(tmp_path, monkeypatch, arm, messages):
    run_root = tmp_path / 'openbrowser'
    run_root.mkdir()
    stdin = io.StringIO(''.join(json.dumps(message) + '\n' for message in messages))
    stdout = io.StringIO()
    monkeypatch.setattr(benchmark.sys, 'stdin', stdin)
    monkeypatch.setattr(benchmark.sys, 'stdout', stdout)
    monkeypatch.setattr(benchmark, 'Arm', lambda _args: arm)
    args = SimpleNamespace(
        worker='openbrowser', worker_mode=benchmark.WorkerMode.PROFILE, run_root=str(run_root),
    )
    asyncio.run(benchmark.worker(args))
    replies = [
        json.loads(line[len(benchmark.PREFIX):])
        for line in stdout.getvalue().splitlines()
        if line.startswith(benchmark.PREFIX)
    ]
    return run_root, replies


def test_profile_command_has_an_isolated_help_screen():
    result = run_cli(['profile', '--help'])
    assert result.returncode == 0, result.stderr
    assert '--workload' in result.stdout
    assert '--verify' in result.stdout
    assert '--samples' not in result.stdout
    assert '--delay' not in result.stdout


@pytest.mark.parametrize('change', ['tracked', 'untracked'])
def test_profile_cli_rejects_dirty_runtime_source_before_output_or_worker(
    tmp_path, profile_parent, change,
):
    source = dirty_profile_source(tmp_path, change)
    chrome = make_executable(tmp_path / 'chrome', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    worker_started = tmp_path / 'worker-started'
    python = make_executable(
        tmp_path / 'python-wrapper',
        '#!/bin/sh\nprintf started > "$PROFILE_WORKER_START_MARKER"\n'
        'exec "$PROFILE_REAL_PYTHON" "$@"\n',
    )
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(current_manifest(source, python, chrome)))
    output = profile_output_path(profile_parent)
    environment = private_cli_environment(tmp_path)
    environment['PROFILE_WORKER_START_MARKER'] = str(worker_started)
    environment['PROFILE_REAL_PYTHON'] = sys.executable

    result = run_cli(
        ['profile', *run_arguments(
            source, python, chrome, manifest_path, output, result_root=profile_parent,
        )],
        env=environment,
    )

    assert result.returncode == 2, result.stderr
    assert 'profile source has changes under src/openbrowser' in result.stderr
    assert not worker_started.exists()
    assert not output.exists()


@pytest.mark.parametrize('workload', ['fast', 'delayed'])
def test_profile_parser_accepts_all_seven_run_options(workload, tmp_path):
    parser = profile_function('profile_parser')()
    arguments = [
        '--source', str(SCRIPT.parents[1]),
        '--openbrowser-python', sys.executable,
        '--chrome', str(tmp_path / 'chrome'),
        '--version-manifest', str(tmp_path / 'manifest.json'),
        '--workload', workload,
        '--result-root', str(tmp_path / 'result-root'),
        '--output', str(tmp_path / 'profile'),
    ]
    parsed = parser.parse_args(arguments)
    assert parsed.workload == workload
    assert parsed.source == str(SCRIPT.parents[1])
    assert parsed.openbrowser_python == sys.executable
    assert parsed.chrome == str(tmp_path / 'chrome')
    assert parsed.version_manifest == str(tmp_path / 'manifest.json')
    assert parsed.result_root == str(tmp_path / 'result-root')
    assert parsed.output == str(tmp_path / 'profile')
    assert not hasattr(parsed, 'samples')
    assert not hasattr(parsed, 'delay')
    assert not hasattr(parsed, 'arms')


def test_profile_verify_parser_accepts_only_verify_directory(tmp_path):
    parser = profile_function('profile_parser')()
    parsed = parser.parse_args([
        '--verify', str(tmp_path / 'profile'), '--result-root', str(tmp_path / 'result-root'),
    ])
    assert parsed.verify == str(tmp_path / 'profile')
    assert parsed.result_root == str(tmp_path / 'result-root')
    assert parsed.source is None
    assert parsed.openbrowser_python is None
    assert parsed.chrome is None
    assert parsed.version_manifest is None
    assert parsed.workload is None
    assert parsed.output is None


@pytest.mark.parametrize('arguments,expected_text', [
    (['profile'], '--source'),
    (['profile', '--unknown'], '--unknown'),
    (['profile', '--samples', '1'], '--samples'),
    (['profile', '--delay', '0.2'], '--delay'),
    (['profile', '--arms', 'openbrowser'], '--arms'),
    (['profile', '--verify', '/tmp/profile', '--workload', 'fast'], '--verify'),
    (['profile', '--verify', '/tmp/profile'], '--result-root'),
    (['profile', '--workload', 'unsupported'], 'workload'),
])
def test_profile_cli_rejects_invalid_invocations(arguments, expected_text):
    result = run_cli(arguments)
    assert result.returncode == 2
    assert expected_text in result.stderr


def test_profile_output_rejects_path_outside_result_root_before_source_validation(
    tmp_path, profile_parent,
):
    outside = tmp_path / 'outside-profile'
    chrome = tmp_path / 'chrome'
    manifest = tmp_path / 'manifest.json'

    result = run_cli([
        'profile', '--source', '', '--openbrowser-python', sys.executable,
        '--chrome', str(chrome), '--version-manifest', str(manifest), '--workload', 'fast',
        '--result-root', str(profile_parent), '--output', str(outside),
    ], env=private_cli_environment(tmp_path))

    assert result.returncode == 2
    assert 'outside RESULT_ROOT' in result.stderr
    assert not outside.exists()


def test_profile_verify_rejects_path_outside_result_root(tmp_path, profile_parent):
    outside = tmp_path / 'outside-profile'
    outside.mkdir(mode=0o700)

    result = run_cli([
        'profile', '--verify', str(outside), '--result-root', str(profile_parent),
    ], env=private_cli_environment(tmp_path))

    assert result.returncode == 2
    assert 'outside RESULT_ROOT' in result.stderr


def test_profile_output_rejects_a_non_run_directory_inside_result_root(tmp_path, profile_parent):
    non_run_directory = profile_parent / 'profile' / 'fast'
    result = run_cli([
        'profile', '--source', '', '--openbrowser-python', sys.executable,
        '--chrome', str(tmp_path / 'chrome'), '--version-manifest', str(tmp_path / 'manifest.json'),
        '--workload', 'fast', '--result-root', str(profile_parent),
        '--output', str(non_run_directory),
    ], env=private_cli_environment(tmp_path))

    assert result.returncode == 2
    assert 'RESULT_ROOT/profile/<workload>/<run-id>' in result.stderr
    assert list(non_run_directory.iterdir()) == []


def test_profile_request_accepts_required_manifest_and_preserves_python_symlink(
    tmp_path, profile_parent,
):
    validate = profile_function('validate_profile_request')
    source = SCRIPT.parents[1]
    parent = profile_parent
    chrome = make_executable(tmp_path / 'chrome', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    python_alias = tmp_path / 'candidate-python'
    python_alias.symlink_to(Path(sys.executable))
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(current_manifest(source, python_alias, chrome)))
    output = profile_output_path(parent)
    parser = profile_function('profile_parser')()
    parsed = parser.parse_args(run_arguments(
        source, python_alias, chrome, manifest_path, output, result_root=parent,
    ))
    request = validate(parsed)
    assert request.source == source.resolve()
    assert request.openbrowser_python == python_alias.absolute()
    assert request.openbrowser_python != python_alias.resolve()
    assert request.chrome == chrome.absolute()
    assert request.output == output
    assert request.workload == 'fast'
    assert not output.exists()


def test_profile_request_rejects_manifest_missing_current_run_identity(tmp_path, profile_parent):
    validate = profile_function('validate_profile_request')
    source = SCRIPT.parents[1]
    parent = profile_parent
    chrome = make_executable(tmp_path / 'chrome', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    manifest = current_manifest(source, sys.executable, chrome)
    del manifest['installed_executables']
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    parsed = profile_function('profile_parser')().parse_args(
        run_arguments(
            source, sys.executable, chrome, manifest_path, profile_output_path(parent), result_root=parent,
        )
    )
    with pytest.raises(ValueError, match='manifest'):
        validate(parsed)


def test_profile_request_rejects_existing_output_and_traversal(tmp_path, profile_parent):
    validate = profile_function('validate_profile_request')
    source = SCRIPT.parents[1]
    parent = profile_parent
    chrome = make_executable(tmp_path / 'chrome', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(current_manifest(source, sys.executable, chrome)))
    existing = profile_output_path(parent)
    existing.mkdir(mode=0o700)
    parser = profile_function('profile_parser')()
    existing_args = parser.parse_args(run_arguments(
        source, sys.executable, chrome, manifest_path, existing, result_root=parent,
    ))
    with pytest.raises(ValueError, match='output'):
        validate(existing_args)
    traversal_args = parser.parse_args(
        run_arguments(
            source, sys.executable, chrome, manifest_path, parent / '..' / 'escaped', result_root=parent,
        )
    )
    with pytest.raises(ValueError, match='output|path|temporary|RESULT_ROOT'):
        validate(traversal_args)


def test_profile_request_rejects_symlink_component_below_physical_temp_root(
    tmp_path, profile_parent,
):
    validate = profile_function('validate_profile_request')
    source = SCRIPT.parents[1]
    parent = profile_parent
    chrome = make_executable(tmp_path / 'chrome', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(current_manifest(source, sys.executable, chrome)))
    target = profile_parent / 'target-parent'
    target.mkdir(mode=0o700)
    link = parent / 'link'
    link.symlink_to(target, target_is_directory=True)
    parsed = profile_function('profile_parser')().parse_args(
        run_arguments(
            source, sys.executable, chrome, manifest_path,
            link / 'profile' / 'fast' / ('a' * 32), result_root=parent,
        )
    )
    with pytest.raises(ValueError, match='symlink'):
        validate(parsed)


def test_profile_request_rejects_output_outside_result_root(tmp_path, profile_parent):
    validate = profile_function('validate_profile_request')
    source = SCRIPT.parents[1]
    chrome = make_executable(tmp_path / 'chrome', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(current_manifest(source, sys.executable, chrome)))
    outside = SCRIPT.parents[1] / 'outside-profile'
    parsed = profile_function('profile_parser')().parse_args(
        run_arguments(
            source, sys.executable, chrome, manifest_path, outside, result_root=profile_parent,
        )
    )
    with pytest.raises(ValueError, match='outside RESULT_ROOT'):
        validate(parsed)
    assert not outside.exists()


def test_profile_request_rejects_empty_source_even_when_current_directory_is_a_repository(
    tmp_path, profile_parent,
):
    validate = profile_function('validate_profile_request')
    chrome = make_executable(tmp_path / 'chrome', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(current_manifest(SCRIPT.parents[1], sys.executable, chrome)))
    parsed = profile_function('profile_parser')().parse_args(
        run_arguments(
            '', sys.executable, chrome, manifest_path,
            profile_output_path(profile_parent), result_root=profile_parent,
        )
    )
    with pytest.raises(ValueError, match='source'):
        validate(parsed)


def test_worker_profiles_only_selected_trial_after_fast_unprofiled_warmup(tmp_path, monkeypatch):
    arm = FakeArm()
    profiler = install_profile_spy(monkeypatch, arm)
    request = profile_pair_request('delayed')
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm, [{'op': 'start'}, request, {'op': 'stop'}]
    )
    result = replies[1]['result']
    assert replies[1]['ok'] is True
    assert [(sample['kind'], sample['index']) for sample in arm.trials] == [('fast', 0), ('delayed', 0)]
    assert arm.profile_states == [False, True]
    assert profiler.enabled is False
    assert result['warmup']['outcome'] == 'success'
    assert result['target']['outcome'] == 'success'
    assert result['warmup']['sample']['nonce'] != result['target']['sample']['nonce']
    assert result['profile']['enabled'] is True
    assert result['profile']['disabled'] is True
    assert result['profile']['dumped'] is True
    assert (run_root / 'trial.pstats').is_file()


@pytest.mark.parametrize(('label', 'invalid_index'), [
    ('warmup', False),
    ('target', 0.0),
])
def test_worker_rejects_non_integer_request_sample_indices(tmp_path, monkeypatch, label, invalid_index):
    arm = FakeArm()
    request = profile_pair_request('fast')
    request[label]['index'] = invalid_index

    _run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm, [{'op': 'start'}, request, {'op': 'stop'}],
    )

    assert replies[1]['ok'] is False
    assert arm.trials == []


@pytest.mark.parametrize('invalid_index', [False, 0.0])
def test_worker_rejects_non_integer_returned_sample_indices(tmp_path, monkeypatch, invalid_index):
    arm = FakeArm()
    original_trial = arm.trial

    async def trial_with_invalid_index(request):
        sample = await original_trial(request)
        sample['index'] = invalid_index
        return sample

    arm.trial = trial_with_invalid_index
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )

    result = replies[1]['result']
    assert replies[1]['ok'] is True
    assert result['failed_phase'] == 'warmup'
    assert result['warmup']['outcome'] == 'failure'
    assert result['target'] == {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    assert result['profile']['enabled'] is False
    assert not (run_root / 'trial.pstats').exists()


@pytest.mark.parametrize(('record_name', 'field', 'value'), [
    ('warmup', 'expected', True),
    ('target', 'expected', 1.0),
    ('warmup', 'attempted', True),
    ('target', 'attempted', 1.0),
    ('warmup.sample', 'index', False),
    ('target.sample', 'index', 0.0),
])
def test_profile_pair_errors_reject_non_integer_accounting(record_name, field, value):
    request = profile_pair_request('fast')
    warmup_sample = successful_trial(benchmark.profile_trial_request(request['warmup'], request['fixture_port']))
    target_sample = successful_trial(benchmark.profile_trial_request(request['target'], request['fixture_port']))
    warmup = {'expected': 1, 'attempted': 1, 'outcome': 'success', 'sample': warmup_sample}
    target = {'expected': 1, 'attempted': 1, 'outcome': 'success', 'sample': target_sample}

    record = warmup if record_name.startswith('warmup') else target
    if '.' in record_name:
        record = record['sample']
    record[field] = value

    assert benchmark.profile_pair_errors(warmup, target, 'fast')


def test_worker_atomically_records_profile_pair_progress(tmp_path, monkeypatch):
    arm = FakeArm()
    write_json = benchmark.atomic_profile_json
    progress_writes = []

    def record_progress_writes(path, value):
        if Path(path).name == 'profile-progress.json':
            progress_writes.append(json.loads(json.dumps(value)))
        return write_json(path, value)

    monkeypatch.setattr(benchmark, 'atomic_profile_json', record_progress_writes)
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )

    assert replies[1]['ok'] is True
    assert [write['phase'] for write in progress_writes] == [
        'ready', 'warmup', 'warmup', 'profile', 'target', 'profile', 'complete',
    ]
    assert progress_writes[0] == {
        'schema': 'openbrowser-navigation-progress/1',
        'phase': 'ready',
        'warmup': {'attempted': 0, 'outcome': 'not_run'},
        'target': {'attempted': 0, 'outcome': 'not_run'},
    }
    assert progress_writes[1]['warmup'] == {'attempted': 1, 'outcome': 'in_progress'}
    assert progress_writes[2]['warmup'] == {'attempted': 1, 'outcome': 'success'}
    assert progress_writes[3]['warmup'] == {'attempted': 1, 'outcome': 'success'}
    assert progress_writes[4]['target'] == {'attempted': 1, 'outcome': 'in_progress'}
    assert progress_writes[5]['target'] == {'attempted': 1, 'outcome': 'success'}
    assert progress_writes[-1]['phase'] == 'complete'
    assert json.loads((run_root / 'profile-progress.json').read_text()) == progress_writes[-1]


def test_profile_progress_writes_and_sample_validation_run_with_profiler_disabled(tmp_path, monkeypatch):
    arm = FakeArm()
    profiler = install_profile_spy(monkeypatch, arm)
    write_json = benchmark.atomic_profile_json
    record_trial = benchmark.profile_trial_record
    progress_observations = []
    validation_observations = []

    def observe_progress_write(path, value):
        if Path(path).name == 'profile-progress.json':
            progress_observations.append({
                'phase': value['phase'],
                'target': dict(value['target']),
                'profiler_enabled': profiler.enabled,
            })
        return write_json(path, value)

    def observe_sample_validation(sample, request, label):
        validation_observations.append((label, profiler.enabled))
        return record_trial(sample, request, label)

    monkeypatch.setattr(benchmark, 'atomic_profile_json', observe_progress_write)
    monkeypatch.setattr(benchmark, 'profile_trial_record', observe_sample_validation)
    _run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )

    assert replies[1]['ok'] is True
    target_start = next(item for item in progress_observations if item['phase'] == 'target')
    target_completion = next(
        item for item in progress_observations
        if item['phase'] == 'profile' and item['target']['attempted'] == 1
    )
    assert target_start['target'] == {'attempted': 1, 'outcome': 'in_progress'}
    assert target_start['profiler_enabled'] is False
    assert target_completion['target'] == {'attempted': 1, 'outcome': 'success'}
    assert target_completion['profiler_enabled'] is False
    assert validation_observations == [('warmup', False), ('target', False)]
    assert all(item['profiler_enabled'] is False for item in progress_observations)


def test_worker_returns_profile_failure_if_warmup_progress_write_fails(tmp_path, monkeypatch):
    arm = FakeArm()
    write_json = benchmark.atomic_profile_json

    def fail_after_warmup(path, value):
        if Path(path).name == 'profile-progress.json' and value['warmup']['outcome'] == 'success':
            raise OSError('warmup progress write failed')
        return write_json(path, value)

    monkeypatch.setattr(benchmark, 'atomic_profile_json', fail_after_warmup)
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )

    assert replies[1]['ok'] is True
    result = replies[1]['result']
    assert result['failed_phase'] == 'profile'
    assert result['warmup']['attempted'] == 1
    assert result['warmup']['outcome'] == 'success'
    assert result['warmup']['sample']['nonce'] == 'warmup-nonce'
    assert result['target'] == {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    assert result['errors']
    assert len(arm.trials) == 1
    assert arm.trials[0]['nonce'] == 'warmup-nonce'
    assert json.loads((run_root / 'profile-progress.json').read_text())['warmup'] == {
        'attempted': 1, 'outcome': 'in_progress',
    }


def test_worker_preserves_target_sample_if_completion_progress_write_fails(tmp_path, monkeypatch):
    arm = FakeArm()
    write_json = benchmark.atomic_profile_json

    def fail_target_completion(path, value):
        if Path(path).name == 'profile-progress.json' and value['target'] == {
            'attempted': 1, 'outcome': 'success',
        }:
            raise OSError('target completion progress write failed')
        return write_json(path, value)

    monkeypatch.setattr(benchmark, 'atomic_profile_json', fail_target_completion)
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )

    assert replies[1]['ok'] is True
    result = replies[1]['result']
    assert result['failed_phase'] == 'profile'
    assert result['warmup']['outcome'] == 'success'
    assert result['target']['attempted'] == 1
    assert result['target']['outcome'] == 'success'
    assert result['target']['sample']['nonce'] == 'target-nonce'
    assert result['profile']['enabled'] is True
    assert result['profile']['disabled'] is True
    assert result['profile']['dumped'] is False
    assert any('target completion progress write failed' in error for error in result['errors'])
    assert len(arm.trials) == 2
    assert json.loads((run_root / 'profile-progress.json').read_text())['target'] == {
        'attempted': 1, 'outcome': 'in_progress',
    }
    assert not (run_root / 'trial.pstats').exists()


def test_worker_does_not_profile_target_when_fast_warmup_fails(tmp_path, monkeypatch):
    arm = FakeArm(fail_warmup=True)
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )
    result = replies[1]['result']
    assert replies[1]['ok'] is True
    assert len(arm.trials) == 1
    assert arm.profile_states == [False]
    assert result['warmup']['outcome'] == 'failure'
    assert result['target'] == {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
    assert result['profile']['enabled'] is False
    assert not (run_root / 'trial.pstats').exists()


def test_worker_rejects_second_pair_and_requests_that_choose_paths(tmp_path, monkeypatch):
    arm = FakeArm()
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), profile_pair_request('delayed'), {'op': 'stop'}],
    )
    assert replies[1]['ok'] is True
    assert replies[2]['ok'] is False
    assert len(arm.trials) == 2
    assert (run_root / 'trial.pstats').is_file()

    invalid_root = tmp_path / 'invalid-request'
    invalid_root.mkdir()
    invalid_arm = FakeArm()
    invalid = profile_pair_request('fast')
    invalid['stats_path'] = '/tmp/user-controlled.pstats'
    _, invalid_replies = worker_messages(
        invalid_root, monkeypatch, invalid_arm, [{'op': 'start'}, invalid, {'op': 'stop'}]
    )
    assert invalid_replies[1]['ok'] is False
    assert invalid_arm.trials == []


def test_worker_disables_profiler_after_target_exception(tmp_path, monkeypatch):
    arm = FakeArm(fail_target=True)
    profiler = install_profile_spy(monkeypatch, arm)
    _run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )
    assert replies[1]['ok'] is True
    assert replies[1]['result']['target']['outcome'] == 'failure'
    assert replies[1]['result']['failed_phase'] == 'target'
    assert arm.profile_states == [False, True]
    assert profiler.enabled is False
    assert arm.stopped is True


def test_worker_disables_profiler_after_target_timeout(tmp_path, monkeypatch):
    arm = FakeArm(timeout_target=True)
    profiler = install_profile_spy(monkeypatch, arm)
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )
    result = replies[1]['result']
    assert replies[1]['ok'] is True
    assert result['target']['outcome'] == 'failure'
    assert result['target']['sample']['timed_out'] is True
    assert arm.profile_states == [False, True]
    assert profiler.enabled is False
    assert profiler.real.getstats()
    assert (run_root / 'trial.pstats').is_file()


@pytest.mark.parametrize('failure', ['enable', 'disable', 'dump'])
def test_worker_reports_profile_failures_without_a_stats_artifact(tmp_path, monkeypatch, failure):
    arm = FakeArm()
    real_profile_type = cProfile.Profile

    class FailingProfile:
        def __init__(self):
            self.real = real_profile_type()
            self.enabled = False
            self.disabled = False

        def enable(self):
            if failure == 'enable':
                raise RuntimeError('enable failed')
            self.enabled = True
            self.real.enable()

        def disable(self):
            self.real.disable()
            self.enabled = False
            self.disabled = True
            if failure == 'disable':
                raise RuntimeError('disable failed')

        def dump_stats(self, path):
            if failure == 'dump':
                raise RuntimeError('dump failed')
            self.real.dump_stats(path)

    profiler = FailingProfile()
    arm.profile_probe = lambda: profiler.enabled
    monkeypatch.setattr(cProfile, 'Profile', lambda: profiler)
    run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )
    result = replies[1]['result']
    assert replies[1]['ok'] is True
    assert result['failed_phase'] == 'profile'
    assert result['errors']
    if failure == 'enable':
        assert result['target'] == {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
        progress = json.loads((run_root / 'profile-progress.json').read_text())
        assert progress['target'] == {'attempted': 0, 'outcome': 'not_run'}
        assert len(arm.trials) == 1
    else:
        assert result['target']['outcome'] == 'success'
        assert len(arm.trials) == 2
    assert (run_root / 'trial.pstats').exists() is False


def test_target_failure_stays_first_when_profiler_disable_also_fails(tmp_path, monkeypatch):
    arm = FakeArm(timeout_target=True)
    real_profile_type = cProfile.Profile

    class DisableFailingProfile:
        def __init__(self):
            self.real = real_profile_type()
            self.enabled = False

        def enable(self):
            self.enabled = True
            self.real.enable()

        def disable(self):
            self.real.disable()
            self.enabled = False
            raise RuntimeError('disable failed')

        def dump_stats(self, path):
            self.real.dump_stats(path)

    profiler = DisableFailingProfile()
    arm.profile_probe = lambda: profiler.enabled
    monkeypatch.setattr(cProfile, 'Profile', lambda: profiler)
    _run_root, replies = worker_messages(
        tmp_path, monkeypatch, arm,
        [{'op': 'start'}, profile_pair_request('fast'), {'op': 'stop'}],
    )

    result = replies[1]['result']
    assert result['target']['outcome'] == 'failure'
    assert any('disable failed' in error for error in result['errors'])
    assert result['failed_phase'] == 'target'


@pytest.mark.asyncio
async def test_worker_disables_profiler_when_target_is_cancelled(tmp_path, monkeypatch):
    arm = FakeArm(block_target=True)
    profiler = install_profile_spy(monkeypatch, arm)
    run_root = tmp_path / 'openbrowser'
    run_root.mkdir()
    messages = [{'op': 'start'}, profile_pair_request('fast')]
    stdin = io.StringIO(''.join(json.dumps(message) + '\n' for message in messages))
    stdout = io.StringIO()
    monkeypatch.setattr(benchmark.sys, 'stdin', stdin)
    monkeypatch.setattr(benchmark.sys, 'stdout', stdout)
    monkeypatch.setattr(benchmark, 'Arm', lambda _args: arm)
    task = asyncio.create_task(benchmark.worker(SimpleNamespace(
        worker='openbrowser', worker_mode=benchmark.WorkerMode.PROFILE, run_root=str(run_root),
    )))
    try:
        await asyncio.wait_for(arm.target_started.wait(), timeout=2)
    except TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        pytest.fail('profile_pair did not start its selected target')
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert arm.profile_states == [False, True]
    assert profiler.enabled is False
    assert arm.stopped is True


@pytest.fixture
def profile_input(tmp_path, profile_parent):
    parent = profile_parent
    source = SCRIPT.parents[1].resolve()
    python = Path(sys.executable).absolute()
    chrome = make_executable(tmp_path / 'chrome-for-testing', '#!/bin/sh\necho "Chrome for Testing 1.2.3"\n')
    manifest = current_manifest(source, python, chrome)
    manifest_path = tmp_path / 'current-run-manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    request = benchmark.ProfileRequest(
        source=source,
        openbrowser_python=python,
        chrome=chrome.absolute(),
        version_manifest=manifest_path.absolute(),
        workload='delayed',
        result_root=parent,
        output=profile_output_path(parent, workload='delayed'),
    )
    return request, parent, manifest_path, chrome


class FakeWorkerProcess:
    def __init__(self, pid, run_root):
        self.pid = pid
        self.run_root = run_root
        self.stdin = SimpleNamespace(close=lambda: None)
        self.returncode = 0

    async def wait(self):
        return self.returncode


def write_fake_stats(path):
    profiler = cProfile.Profile()
    profiler.runcall(lambda: sum(range(4)))
    profiler.dump_stats(str(path))
    Path(path).chmod(0o600)


def complete_cleanup_identity(pid, state='absent'):
    return {
        'pid': pid,
        'created': 1234.5,
        'cmdline': ['python', '--worker', 'openbrowser'],
        'owner_token': 'profile-owner-token',
        'owner_root': '/tmp/profile-runtime/openbrowser',
        'state': state,
    }


def install_profile_run_fakes(monkeypatch, request, scenario='success'):
    captured = {'pair_requests': [], 'cleanup_called': False, 'snapshots': {}}
    worker_identity = {}

    async def create_process(*command, cwd, env, **_kwargs):
        run_root = Path(command[command.index('--run-root') + 1])
        proc = FakeWorkerProcess(43127, run_root)
        captured.update(command=command, cwd=cwd, env=dict(env), process=proc, run_root=run_root,
                        runtime_root=run_root.parent)
        captured['owner_marker'] = (run_root.parent / '.benchmark-owner').read_text()
        return proc

    def make_trial(spec, port):
        sample_request = benchmark.profile_trial_request(spec, port)
        sample = successful_trial(sample_request)
        captured['snapshots'][sample['nonce']] = sample['server']
        return sample

    async def send(proc, message, _log, timeout=90):
        if message['op'] == 'start':
            if scenario == 'startup':
                return {'ok': False, 'error': 'startup failed'}
            if scenario == 'startup_inventory_lost':
                raise RuntimeError('worker exited before startup identity response')
            started = {
                'version': {'product': 'Chrome/1.2.3'},
                'owned': [],
                'packages': {'psutil': '7.2.0', 'websockets': '15.0.1'},
                'loaded_packages': {'openbrowser-ai': '0.1.56'},
            }
            if scenario == 'startup_inventory_missing_owned':
                del started['owned']
            elif scenario == 'startup_inventory_invalid_owned':
                started['owned'] = None
            return {'ok': True, 'result': started}
        if message['op'] == 'stop':
            stop_errors = ['worker stop reported nested errors'] if scenario == 'worker_stop_nested_errors' else []
            return {
                'ok': scenario != 'worker_stop',
                'error': 'worker stop failed' if scenario == 'worker_stop' else None,
                'result': {'identities': [], 'errors': stop_errors},
            }
        if message['op'] != 'profile_pair':
            raise AssertionError(f'unexpected worker request: {message!r}')
        captured['pair_requests'].append(message)
        if scenario == 'pair_eof_no_progress':
            raise RuntimeError('worker stdout closed without reply')
        if scenario == 'pair_nonloss_error_stale_target':
            benchmark.atomic_profile_json(proc.run_root / 'profile-progress.json', {
                'schema': 'openbrowser-navigation-progress/1',
                'phase': 'target',
                'warmup': {'attempted': 1, 'outcome': 'success'},
                'target': {'attempted': 1, 'outcome': 'in_progress'},
            })
            raise RuntimeError('profile_pair transport error')
        if scenario == 'pair_error_envelope_stale_target':
            benchmark.atomic_profile_json(proc.run_root / 'profile-progress.json', {
                'schema': 'openbrowser-navigation-progress/1',
                'phase': 'target',
                'warmup': {'attempted': 1, 'outcome': 'success'},
                'target': {'attempted': 1, 'outcome': 'in_progress'},
            })
            return {'ok': False, 'error': 'profile progress write failed'}
        if scenario == 'pair_eof_warmup':
            benchmark.atomic_profile_json(proc.run_root / 'profile-progress.json', {
                'schema': 'openbrowser-navigation-progress/1',
                'phase': 'warmup',
                'warmup': {'attempted': 1, 'outcome': 'in_progress'},
                'target': {'attempted': 0, 'outcome': 'not_run'},
            })
            raise RuntimeError('worker stdout closed without reply')
        if scenario == 'pair_timeout_target':
            benchmark.atomic_profile_json(proc.run_root / 'profile-progress.json', {
                'schema': 'openbrowser-navigation-progress/1',
                'phase': 'target',
                'warmup': {'attempted': 1, 'outcome': 'success'},
                'target': {'attempted': 1, 'outcome': 'in_progress'},
            })
            raise TimeoutError('profile_pair timed out')
        warm_sample = make_trial(message['warmup'], message['fixture_port'])
        target_sample = make_trial(message['target'], message['fixture_port'])
        warmup = {'expected': 1, 'attempted': 1, 'outcome': 'success', 'sample': warm_sample}
        target = {'expected': 1, 'attempted': 1, 'outcome': 'success', 'sample': target_sample}
        profile_state = {'enabled': True, 'disabled': True, 'dumped': True, 'file': 'trial.pstats'}
        failed_phase = None
        errors = []
        if scenario == 'warmup':
            warm_sample.update(outcome='failure', exception='RuntimeError: warmup failed')
            warmup['outcome'] = 'failure'
            target = {'expected': 1, 'attempted': 0, 'outcome': 'not_run', 'sample': None}
            profile_state = {'enabled': False, 'disabled': False, 'dumped': False, 'file': 'trial.pstats'}
            failed_phase = 'warmup'
            errors.append('warmup failed')
        elif scenario == 'bad_attempt':
            warmup['attempted'] = 0
        elif scenario in ('target', 'target_timeout'):
            target_sample.update(outcome='failure', exception='TimeoutError: target failed',
                                 timed_out=scenario == 'target_timeout')
            target['outcome'] = 'failure'
            failed_phase = 'target'
            errors.append('target failed')
        elif scenario == 'profile':
            profile_state = {'enabled': True, 'disabled': False, 'dumped': False, 'file': 'trial.pstats'}
            failed_phase = 'profile'
            errors.append('profile dump failed')
        elif scenario == 'progress_write_failure':
            profile_state = {'enabled': True, 'disabled': True, 'dumped': False, 'file': 'trial.pstats'}
            failed_phase = 'profile'
            errors.append('target completion progress write failed')
            benchmark.atomic_profile_json(proc.run_root / 'profile-progress.json', {
                'schema': 'openbrowser-navigation-progress/1',
                'phase': 'target',
                'warmup': {'attempted': 1, 'outcome': 'success'},
                'target': {'attempted': 1, 'outcome': 'in_progress'},
            })
        if profile_state['dumped']:
            write_fake_stats(proc.run_root / 'trial.pstats')
        return {'ok': True, 'result': {
            'warmup': warmup,
            'target': target,
            'profile': profile_state,
            'browser_runtime': {'product': 'Chrome/1.2.3'},
            'packages': {
                'packages': {'psutil': '7.2.0', 'websockets': '15.0.1'},
                'loaded_packages': {'openbrowser-ai': '0.1.56'},
            },
            'failed_phase': failed_phase,
            'errors': errors,
        }}

    def record(_process, token):
        identity = {
            'pid': 43127,
            'created': 1234.5,
            'cmdline': list(captured['command']),
            'owner_token': token,
            'owner_root': str(captured['run_root']),
        }
        worker_identity.update(identity)
        return identity

    async def cleanup(handle):
        captured['cleanup_called'] = True
        state = 'unknown' if scenario == 'cleanup' else 'absent'
        post = [{'pid': 43128, 'state': 'unknown'}] if scenario == 'cleanup' else []
        return {
            'ok': scenario != 'worker_stop',
            'result': {
                'identities': [],
                **({'errors': ['worker stop reported nested errors']}
                   if scenario == 'worker_stop_nested_errors' else {}),
            },
            'worker_identity': {**handle.worker_identity, 'state': state},
            'post_worker_exit_identities': post,
            'fallback_used': False,
            **({'error': 'worker stop failed'} if scenario == 'worker_stop' else {}),
        }

    monkeypatch.setattr(benchmark.asyncio, 'create_subprocess_exec', create_process)
    monkeypatch.setattr(benchmark, 'send', send)
    monkeypatch.setattr(benchmark, 'record', record)
    monkeypatch.setattr(benchmark.psutil, 'Process', lambda pid: SimpleNamespace(pid=pid))
    monkeypatch.setattr(benchmark, 'cleanup_worker', cleanup)
    monkeypatch.setattr(benchmark, 'state', lambda _identity: 'absent')
    monkeypatch.setattr(benchmark, 'descendant_identities', lambda *_args: [])
    monkeypatch.setattr(
        benchmark.Fixture, 'snapshot',
        lambda _fixture, nonce: captured['snapshots'].get(nonce, {'effects': 0, 'requests': []}),
    )
    if scenario == 'source':
        calls = 0

        def source_manifest(_source):
            nonlocal calls
            calls += 1
            return {'src/openbrowser/test.py': ('a' if calls == 1 else 'b') * 64}

        monkeypatch.setattr(benchmark, 'source_manifest', source_manifest)
    return captured


def test_profile_coordinator_runs_one_worker_and_removes_aws_environment(profile_input, monkeypatch):
    request, _parent, manifest_path, _chrome = profile_input
    for name in (
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_PROFILE',
        'AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_SHARED_CREDENTIALS_FILE', 'AWS_CONFIG_FILE',
    ):
        monkeypatch.setenv(name, 'sentinel-' + name.lower())
    captured = install_profile_run_fakes(monkeypatch, request)
    run_profile = profile_function('run_profile')
    code = asyncio.run(run_profile(request))
    assert code == 0
    assert captured['command'][captured['command'].index('--worker') + 1] == 'openbrowser'
    assert Path(captured['cwd']) == captured['run_root']
    assert captured['run_root'].name == 'openbrowser'
    assert captured['owner_marker'] == captured['env']['NAVIGATION_BENCHMARK_OWNER']
    assert {key: value for key, value in captured['env'].items() if key.startswith('AWS_')} == {
        'AWS_EC2_METADATA_DISABLED': 'true',
        'AWS_CONFIG_FILE': '/dev/null',
        'AWS_SHARED_CREDENTIALS_FILE': '/dev/null',
    }
    assert len(captured['pair_requests']) == 1
    pair = captured['pair_requests'][0]
    assert pair['warmup']['kind'] == 'fast'
    assert pair['target']['kind'] == request.workload
    assert pair['warmup']['nonce'] != pair['target']['nonce']
    assert not captured['runtime_root'].exists()

    output = request.output
    report_path = output / 'profile.json'
    report = json.loads(report_path.read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert report['schema'] == 'openbrowser-navigation-profile/1'
    assert report['status'] == 'complete'
    assert report['failed_phase'] is None
    assert report['version_manifest_file'] == 'version-manifest.json'
    assert (output / 'version-manifest.json').read_bytes() == manifest_path.read_bytes()
    assert report['profile']['file'] == 'trial.pstats'
    assert set(hashes) == {
        'profile.json', 'version-manifest.json', 'worker.stdout.log', 'worker.stderr.log', 'trial.pstats',
    }
    assert {path.name for path in output.iterdir()} == set(hashes) | {'sha256.json'}
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_profile_coordinator_retains_root_when_captured_worker_identity_is_incomplete(
    profile_input, monkeypatch,
):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request)

    def incomplete_record(_process, token):
        return {
            'pid': 43127,
            'created': None,
            'cmdline': ['python', '--worker', 'openbrowser'],
            'owner_token': token,
            'owner_root': str(captured['run_root']),
        }

    async def cleanup_with_reused_fallback(handle):
        captured['cleanup_called'] = True
        return {
            'ok': True,
            'result': {'identities': []},
            'worker_identity': {**handle.worker_identity, 'state': 'reused'},
            'post_worker_exit_identities': [],
        }

    monkeypatch.setattr(benchmark, 'record', incomplete_record)
    monkeypatch.setattr(benchmark, 'cleanup_worker', cleanup_with_reused_fallback)

    run_profile = profile_function('run_profile')
    assert asyncio.run(run_profile(request)) == 1

    report = json.loads((request.output / 'profile.json').read_text())
    assert report['status'] == 'failed'
    assert report['cleanup']['worker_identity']['state'] == 'unknown'
    assert report['cleanup']['root_removed'] is False
    assert captured['cleanup_called']
    assert captured['runtime_root'].exists()
    shutil.rmtree(captured['runtime_root'], ignore_errors=True)


def test_profile_coordinator_retains_root_when_captured_worker_identity_has_extra_key(
    profile_input, monkeypatch,
):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request)
    record = benchmark.record

    def record_with_extra_key(process, token):
        identity = record(process, token)
        identity['unexpected'] = 'not part of process identity'
        return identity

    monkeypatch.setattr(benchmark, 'record', record_with_extra_key)
    run_profile = profile_function('run_profile')

    try:
        assert asyncio.run(run_profile(request)) == 1
        report = json.loads((request.output / 'profile.json').read_text())
        assert report['status'] == 'failed'
        assert report['cleanup']['root_removed'] is False
        assert captured['runtime_root'].exists()
    finally:
        shutil.rmtree(captured['runtime_root'], ignore_errors=True)


@pytest.mark.parametrize('mutation', ['missing', 'non_list'])
def test_profile_coordinator_retains_root_when_post_exit_identities_are_invalid(
    profile_input, monkeypatch, mutation,
):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request)
    cleanup_worker = benchmark.cleanup_worker

    async def cleanup_with_invalid_post_exit_identities(handle):
        response = await cleanup_worker(handle)
        if mutation == 'missing':
            del response['post_worker_exit_identities']
        else:
            response['post_worker_exit_identities'] = {'pid': 43128, 'state': 'absent'}
        return response

    monkeypatch.setattr(benchmark, 'cleanup_worker', cleanup_with_invalid_post_exit_identities)
    run_profile = profile_function('run_profile')

    try:
        assert asyncio.run(run_profile(request)) == 1
        report = json.loads((request.output / 'profile.json').read_text())
        cleanup = report['cleanup']
        identities = [cleanup['worker_identity'], *cleanup['post_worker_exit_identities']]
        assert report['status'] == 'failed'
        assert cleanup['ok'] is False
        assert cleanup['root_removed'] is False
        assert any(identity.get('state') == 'unknown' for identity in identities)
        assert captured['runtime_root'].exists()
    finally:
        runtime_root = captured.get('runtime_root')
        if runtime_root is not None:
            shutil.rmtree(runtime_root, ignore_errors=True)


@pytest.mark.parametrize(('field', 'value'), [
    ('created', None),
    ('ownership_error', 'process ownership could not be verified'),
])
def test_profile_coordinator_retains_root_for_incomplete_post_exit_identity(
    profile_input, monkeypatch, field, value,
):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request)
    cleanup_worker = benchmark.cleanup_worker

    async def cleanup_with_incomplete_post_exit_identity(handle):
        response = await cleanup_worker(handle)
        post_identity = {
            'pid': 43128,
            'created': 1234.5,
            'cmdline': ['chrome', '--user-data-dir=private'],
            'owner_token': handle.owner_token,
            'owner_root': str(captured['run_root']),
            'state': 'absent',
        }
        post_identity[field] = value
        response['post_worker_exit_identities'] = [post_identity]
        return response

    monkeypatch.setattr(benchmark, 'cleanup_worker', cleanup_with_incomplete_post_exit_identity)
    run_profile = profile_function('run_profile')

    try:
        assert asyncio.run(run_profile(request)) == 1
        report = json.loads((request.output / 'profile.json').read_text())
        cleanup = report['cleanup']
        assert report['status'] == 'failed'
        assert cleanup['ok'] is False
        assert cleanup['root_removed'] is False
        assert cleanup['post_worker_exit_identities'][0]['state'] == 'unknown'
        assert captured['runtime_root'].exists()
    finally:
        runtime_root = captured.get('runtime_root')
        if runtime_root is not None:
            shutil.rmtree(runtime_root, ignore_errors=True)


@pytest.mark.parametrize('scenario', [
    'startup_inventory_lost', 'startup_inventory_missing_owned', 'startup_inventory_invalid_owned',
])
def test_profile_coordinator_retains_root_when_startup_identity_inventory_is_unknown(
    profile_input, monkeypatch, scenario,
):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request, scenario)
    run_profile = profile_function('run_profile')

    try:
        assert asyncio.run(run_profile(request)) == 1
        report = json.loads((request.output / 'profile.json').read_text())

        assert report['status'] == 'failed'
        assert report['failed_phase'] == 'startup'
        assert report['cleanup']['ok'] is False
        assert report['cleanup']['root_removed'] is False
        assert captured['runtime_root'].exists()
        assert captured['cleanup_called']
    finally:
        runtime_root = captured.get('runtime_root')
        if runtime_root is not None and runtime_root.exists():
            shutil.rmtree(runtime_root, ignore_errors=True)


def test_profile_coordinator_trusts_worker_profile_phase_and_keeps_target_sample(profile_input, monkeypatch):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request, 'progress_write_failure')
    run_profile = profile_function('run_profile')

    assert asyncio.run(run_profile(request)) == 1

    report = json.loads((request.output / 'profile.json').read_text())
    assert report['status'] == 'failed'
    assert report['failed_phase'] == 'profile'
    assert (report['target']['attempted'], report['target']['outcome']) == (1, 'success')
    assert report['target']['sample']['nonce'].startswith('target-')
    assert report['profile'] is None
    assert not (request.output / 'trial.pstats').exists()
    assert captured['cleanup_called']


def test_profile_run_reverifies_saved_artifacts_and_removes_stats_on_failure(profile_input, monkeypatch):
    request, _parent, _manifest_path, _chrome = profile_input
    install_profile_run_fakes(monkeypatch, request)
    verify_profile = benchmark.verify_profile
    verification_calls = []

    def corrupt_stats_then_verify(directory, result_root):
        verification_calls.append(Path(directory))
        stats_path = Path(directory) / 'trial.pstats'
        stats_path.write_bytes(stats_path.read_bytes() + b'corrupted after the initial hash')
        return verify_profile(directory, result_root)

    monkeypatch.setattr(benchmark, 'verify_profile', corrupt_stats_then_verify)
    run_profile = profile_function('run_profile')

    assert asyncio.run(run_profile(request)) == 1

    report = json.loads((request.output / 'profile.json').read_text())
    hashes = json.loads((request.output / 'sha256.json').read_text())
    assert verification_calls == [request.output]
    assert report['status'] == 'failed'
    assert report['failed_phase'] == 'artifact_verify'
    assert report['profile'] is None
    assert report['errors']
    assert report['verification_errors']
    assert not (request.output / 'trial.pstats').exists()
    assert set(hashes) == {
        'profile.json', 'version-manifest.json', 'worker.stdout.log', 'worker.stderr.log',
    }


@pytest.mark.parametrize(('scenario', 'phase', 'warmup', 'target', 'retained'), [
    ('startup', 'startup', ('not_run', 0), ('not_run', 0), True),
    ('warmup', 'warmup', ('failure', 1), ('not_run', 0), False),
    ('target', 'target', ('success', 1), ('failure', 1), False),
    ('target_timeout', 'target', ('success', 1), ('failure', 1), False),
    ('profile', 'profile', ('success', 1), ('success', 1), False),
    ('pair_error_envelope_stale_target', 'profile', ('not_run', 0), ('not_run', 0), False),
    ('pair_nonloss_error_stale_target', 'profile', ('not_run', 0), ('not_run', 0), False),
    ('pair_eof_no_progress', 'profile', ('not_run', 0), ('not_run', 0), False),
    ('pair_eof_warmup', 'warmup', ('failure', 1), ('not_run', 0), False),
    ('pair_timeout_target', 'target', ('success', 1), ('failure', 1), False),
    ('source', 'source', ('success', 1), ('success', 1), False),
    ('worker_stop', 'worker_stop', ('success', 1), ('success', 1), False),
    ('worker_stop_nested_errors', 'worker_stop', ('success', 1), ('success', 1), False),
    ('cleanup', 'cleanup', ('success', 1), ('success', 1), True),
    ('stats_copy', 'copy', ('success', 1), ('success', 1), False),
])
def test_profile_coordinator_writes_failure_report_for_lifecycle_errors(
    profile_input, monkeypatch, scenario, phase, warmup, target, retained,
):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request, scenario)
    if scenario == 'stats_copy':
        copyfile = benchmark.shutil.copyfile

        def fail_stats_copy(source, destination, *args, **kwargs):
            if Path(source).name == 'trial.pstats':
                raise OSError('stats copy failed')
            return copyfile(source, destination, *args, **kwargs)

        monkeypatch.setattr(benchmark.shutil, 'copyfile', fail_stats_copy)
    run_profile = profile_function('run_profile')
    try:
        assert asyncio.run(run_profile(request)) == 1
        report = json.loads((request.output / 'profile.json').read_text())
        assert report['status'] == 'failed'
        assert report['failed_phase'] == phase
        assert report['profile'] is None
        assert report['errors']
        assert report['verification_errors']
        assert (report['warmup']['outcome'], report['warmup']['attempted']) == warmup
        assert (report['target']['outcome'], report['target']['attempted']) == target
        assert report['cleanup']['root_removed'] is not retained
        assert captured['cleanup_called']
        assert captured['runtime_root'].exists() is retained
    finally:
        if captured.get('runtime_root') and captured['runtime_root'].exists():
            shutil.rmtree(captured['runtime_root'], ignore_errors=True)


def test_profile_manifest_copy_failure_removes_owned_output_before_worker_start(profile_input, monkeypatch):
    request, _parent, manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request)
    copyfile = benchmark.shutil.copyfile

    def fail_manifest_copy(source, destination, *args, **kwargs):
        if Path(source) == manifest_path:
            raise OSError('manifest copy failed')
        return copyfile(source, destination, *args, **kwargs)

    monkeypatch.setattr(benchmark.shutil, 'copyfile', fail_manifest_copy)
    run_profile = profile_function('run_profile')
    assert asyncio.run(run_profile(request)) == 1
    assert not request.output.exists()
    assert not captured.get('run_root')
    assert not captured['cleanup_called']


def test_profile_coordinator_does_not_copy_stats_when_sample_accounting_is_invalid(profile_input, monkeypatch):
    request, _parent, _manifest_path, _chrome = profile_input
    install_profile_run_fakes(monkeypatch, request, 'bad_attempt')
    run_profile = profile_function('run_profile')
    assert asyncio.run(run_profile(request)) == 1
    report = json.loads((request.output / 'profile.json').read_text())
    assert report['status'] == 'failed'
    assert report['failed_phase'] == 'warmup'
    assert report['profile'] is None
    assert not (request.output / 'trial.pstats').exists()


def test_profile_run_removes_stats_when_report_invariant_fails_after_copy(profile_input, monkeypatch):
    request, _parent, _manifest_path, _chrome = profile_input
    install_profile_run_fakes(monkeypatch, request)
    profile_report_errors = benchmark.profile_report_errors

    def reject_complete_report(report):
        errors = profile_report_errors(report)
        if report.get('status') == 'complete':
            return [*errors, 'forced report invariant failure']
        return errors

    monkeypatch.setattr(benchmark, 'profile_report_errors', reject_complete_report)
    run_profile = profile_function('run_profile')

    assert asyncio.run(run_profile(request)) == 1

    report = json.loads((request.output / 'profile.json').read_text())
    hashes = json.loads((request.output / 'sha256.json').read_text())
    assert report['status'] == 'failed'
    assert report['failed_phase'] == 'copy'
    assert report['profile'] is None
    assert report['errors']
    assert report['verification_errors']
    assert not (request.output / 'trial.pstats').exists()
    assert 'trial.pstats' not in hashes


@pytest.mark.parametrize('filename', ['profile.json', 'sha256.json'])
def test_profile_report_and_hash_write_failures_never_exit_success(profile_input, monkeypatch, filename):
    request, _parent, _manifest_path, _chrome = profile_input
    captured = install_profile_run_fakes(monkeypatch, request)
    write_json = benchmark.atomic_profile_json

    def fail_write(path, value):
        if Path(path).name == filename:
            raise OSError(f'{filename} write failed')
        return write_json(path, value)

    monkeypatch.setattr(benchmark, 'atomic_profile_json', fail_write)
    run_profile = profile_function('run_profile')
    assert asyncio.run(run_profile(request)) == 1
    assert not captured['runtime_root'].exists()
    if filename == 'profile.json':
        assert not (request.output / 'profile.json').exists()
        assert not (request.output / 'sha256.json').exists()
    else:
        assert not (request.output / 'sha256.json').exists()
        report = json.loads((request.output / 'profile.json').read_text())
        assert report['status'] == 'failed'
        assert report['failed_phase'] == 'copy'


def complete_profile_artifacts(profile_input):
    request, _parent, _manifest_path, chrome = profile_input
    output = request.output
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    manifest = current_manifest(request.source, request.openbrowser_python, chrome)
    manifest_path = output / 'version-manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    manifest_path.chmod(0o600)
    for name in ('worker.stdout.log', 'worker.stderr.log'):
        path = output / name
        path.write_bytes(b'')
        path.chmod(0o600)
    stats_path = output / 'trial.pstats'
    write_fake_stats(stats_path)
    port = 23456
    warmup_sample = successful_trial(benchmark.profile_trial_request(
        {'arm': 'openbrowser', 'kind': 'fast', 'index': 0, 'nonce': 'warmup-proof'}, port,
    ))
    target_sample = successful_trial(benchmark.profile_trial_request(
        {'arm': 'openbrowser', 'kind': request.workload, 'index': 0, 'nonce': 'target-proof'}, port,
    ))
    source_manifest = {'src/openbrowser/test.py': 'a' * 64}
    report = {
        'schema': 'openbrowser-navigation-profile/1',
        'status': 'complete',
        'failed_phase': None,
        'arm': 'openbrowser',
        'workload': request.workload,
        'delay_seconds': 0.15,
        'source': {
            'commit': manifest['source']['candidate_commit'],
            'manifest_start': source_manifest,
            'manifest_end': source_manifest,
            'unchanged': True,
        },
        'harness': {'sha256': manifest['harness']['sha256']},
        'version_manifest_file': 'version-manifest.json',
        'version_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'tools': {
            'openbrowser_python': {
                'path': str(request.openbrowser_python),
                'version': manifest['python_interpreters']['candidate_openbrowser']['version'],
                'sha256': manifest['python_interpreters']['candidate_openbrowser']['sha256'],
            },
            'chrome': {
                'path': str(request.chrome),
                'version': manifest['installed_executables']['chrome_for_testing']['version'],
                'sha256': manifest['installed_executables']['chrome_for_testing']['sha256'],
            },
            'browser_runtime': {'product': 'Chrome/1.2.3'},
            'packages': {
                'packages': {'psutil': '7.2.0', 'websockets': '15.0.1'},
                'loaded_packages': {'openbrowser-ai': '0.1.56'},
            },
        },
        'warmup': {'expected': 1, 'attempted': 1, 'outcome': 'success', 'sample': warmup_sample},
        'target': {'expected': 1, 'attempted': 1, 'outcome': 'success', 'sample': target_sample},
        'profile': {
            'file': 'trial.pstats',
            'bytes': stats_path.stat().st_size,
            'sha256': hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        },
        'cleanup': {
            'worker_identity': complete_cleanup_identity(12345),
            'post_worker_exit_identities': [],
            'ok': True,
            'root_removed': True,
            'fixture_thread_stopped': True,
        },
        'errors': [],
        'verification_errors': [],
    }
    assert benchmark.profile_report_errors(report) == []
    benchmark.atomic_profile_json(output / 'profile.json', report)
    write_profile_fixture_hashes(output)
    return output, report


def write_profile_fixture_hashes(output):
    output = Path(output)
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.name != 'sha256.json'
    }
    benchmark.atomic_profile_json(output / 'sha256.json', hashes)


def mutate_profile_report(output, mutation):
    report_path = output / 'profile.json'
    report = json.loads(report_path.read_text())
    mutation(report)
    benchmark.atomic_profile_json(report_path, report)
    write_profile_fixture_hashes(output)


@pytest.mark.parametrize('mutation', [
    'schema', 'target_kind', 'target_index', 'failed_warmup', 'zero_samples', 'multiple_samples',
    'matching_nonces', 'bad_task_proof', 'source_drift', 'incomplete_cleanup', 'wrong_size', 'wrong_profile_hash',
])
def test_offline_verifier_rejects_invalid_profile_reports(profile_input, mutation):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        if mutation == 'schema':
            report['schema'] = 'openbrowser-navigation-profile/0'
        elif mutation == 'target_kind':
            report['target']['sample']['kind'] = 'fast'
        elif mutation == 'target_index':
            report['target']['sample']['index'] = 1
        elif mutation == 'failed_warmup':
            report['warmup']['outcome'] = 'failure'
        elif mutation == 'zero_samples':
            report['warmup']['attempted'] = 0
        elif mutation == 'multiple_samples':
            report['target']['attempted'] = 2
        elif mutation == 'matching_nonces':
            report['target']['sample']['nonce'] = report['warmup']['sample']['nonce']
        elif mutation == 'bad_task_proof':
            report['target']['sample']['server']['effects'] = 2
        elif mutation == 'source_drift':
            report['source']['manifest_end']['src/openbrowser/test.py'] = 'b' * 64
            report['source']['unchanged'] = False
        elif mutation == 'incomplete_cleanup':
            report['cleanup']['worker_identity']['state'] = 'alive'
            report['cleanup']['ok'] = False
        elif mutation == 'wrong_size':
            report['profile']['bytes'] += 1
        else:
            report['profile']['sha256'] = 'f' * 64

    mutate_profile_report(output, change)
    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


def test_offline_verifier_rejects_rehashed_unrecognized_report_keys(profile_input):
    output, _report = complete_profile_artifacts(profile_input)
    mutate_profile_report(output, lambda report: report.update(competitor_win=True))

    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


def test_offline_verifier_rejects_rehashed_complete_cleanup_identity_with_extra_key(profile_input):
    output, _report = complete_profile_artifacts(profile_input)

    def add_extra_identity_key(report):
        report['cleanup']['worker_identity']['unexpected'] = 'not part of process identity'

    mutate_profile_report(output, add_extra_identity_key)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    assert any('cleanup' in error and 'identity' in error for error in benchmark.profile_report_errors(report))

    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


@pytest.mark.parametrize(('record_name', 'field', 'value'), [
    ('warmup', 'expected', True),
    ('target', 'expected', 1.0),
    ('warmup', 'attempted', False),
    ('target', 'attempted', 1.0),
    ('target.sample', 'index', False),
    ('warmup.sample', 'index', 0.0),
])
def test_offline_verifier_rejects_rehashed_non_integer_sample_accounting(profile_input, record_name, field, value):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        record = report
        for part in record_name.split('.'):
            record = record[part]
        record[field] = value

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    assert any(record_name.split('.')[0] in error for error in benchmark.profile_report_errors(report))
    assert profile_function('verify_profile')(output, profile_input[0].result_root) == 1


@pytest.mark.parametrize('invalid_name', [
    '/src/openbrowser/test.py',
    '../src/openbrowser/test.py',
    'src/openbrowser/../private.py',
    'src/openbrowser/README.md',
    'src/openbrowser/./test.py',
    'src//openbrowser/test.py',
    'src/openbrowser\\test.py',
])
def test_offline_verifier_rejects_rehashed_non_normalized_source_manifest_paths(profile_input, invalid_name):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        manifest = {invalid_name: 'a' * 64}
        report['source']['manifest_start'] = manifest
        report['source']['manifest_end'] = manifest

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    errors = benchmark.profile_report_errors(report)
    assert any('source' in error for error in errors)
    assert profile_function('verify_profile')(output, profile_input[0].result_root) == 1


@pytest.mark.parametrize('mutation', ['null', 'non_list', 'missing'])
def test_offline_verifier_rejects_rehashed_missing_post_exit_identity_lists(profile_input, mutation):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        cleanup = report['cleanup']
        if mutation == 'null':
            cleanup['post_worker_exit_identities'] = None
        elif mutation == 'non_list':
            cleanup['post_worker_exit_identities'] = {'pid': 12346, 'state': 'absent'}
        else:
            del cleanup['post_worker_exit_identities']

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    assert any('cleanup' in error for error in benchmark.profile_report_errors(report))

    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


@pytest.mark.parametrize('location', ['worker_identity', 'post_worker_exit_identities'])
@pytest.mark.parametrize('field', ['pid', 'created', 'cmdline', 'owner_token', 'owner_root'])
def test_complete_profile_rejects_rehashed_cleanup_identity_with_missing_required_field(
    profile_input, location, field,
):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        if location == 'worker_identity':
            identity = report['cleanup'][location]
        else:
            identity = complete_cleanup_identity(12346)
            report['cleanup'][location] = [identity]
        del identity[field]

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    assert any('cleanup' in error and 'identity' in error for error in benchmark.profile_report_errors(report))
    assert profile_function('verify_profile')(output, profile_input[0].result_root) == 1


@pytest.mark.parametrize(('location', 'field', 'value'), [
    ('worker_identity', 'created', float('nan')),
    ('worker_identity', 'owner_root', 'relative/openbrowser'),
    ('post_worker_exit_identities', 'pid', 0),
    ('post_worker_exit_identities', 'created', 'not-a-number'),
    ('post_worker_exit_identities', 'cmdline', []),
    ('post_worker_exit_identities', 'owner_token', ''),
])
def test_complete_profile_rejects_rehashed_invalid_cleanup_identity_field_values(
    profile_input, location, field, value,
):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        if location == 'worker_identity':
            identity = report['cleanup'][location]
        else:
            identity = complete_cleanup_identity(12346)
            report['cleanup'][location] = [identity]
        identity[field] = value

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    assert any('cleanup' in error and 'identity' in error for error in benchmark.profile_report_errors(report))
    assert profile_function('verify_profile')(output, profile_input[0].result_root) == 1


@pytest.mark.parametrize('location', ['worker_identity', 'post_worker_exit_identities'])
@pytest.mark.parametrize('ownership_error', [None, 'process ownership could not be verified'])
def test_complete_profile_rejects_rehashed_cleanup_identity_ownership_error(
    profile_input, location, ownership_error,
):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        if location == 'worker_identity':
            identity = report['cleanup'][location]
        else:
            identity = complete_cleanup_identity(12346)
            report['cleanup'][location] = [identity]
        identity['ownership_error'] = ownership_error

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    assert any('cleanup' in error and 'ownership' in error for error in benchmark.profile_report_errors(report))
    assert profile_function('verify_profile')(output, profile_input[0].result_root) == 1


@pytest.mark.parametrize(('location', 'identity'), [
    ('worker_identity', None),
    ('worker_identity', {'pid': 12345}),
    ('post_worker_exit_identities', [None]),
    ('post_worker_exit_identities', [{'pid': 12346, 'state': 'alive'}]),
])
def test_complete_profile_rejects_rehashed_invalid_cleanup_identities(profile_input, location, identity):
    output, _report = complete_profile_artifacts(profile_input)
    mutate_profile_report(output, lambda report: report['cleanup'].__setitem__(location, identity))
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    assert any('cleanup' in error for error in benchmark.profile_report_errors(report))

    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


def test_complete_profile_accepts_reused_post_exit_identity(profile_input):
    output, _report = complete_profile_artifacts(profile_input)
    mutate_profile_report(
        output,
        lambda report: report['cleanup'].__setitem__(
            'post_worker_exit_identities', [complete_cleanup_identity(12346, 'reused')],
        ),
    )
    report = json.loads((output / 'profile.json').read_text())
    assert benchmark.profile_report_errors(report) == []

    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 0


@pytest.mark.parametrize(('location', 'identity'), [
    ('worker_identity', None),
    ('worker_identity', {'pid': 12345}),
    ('post_worker_exit_identities', [None]),
    ('post_worker_exit_identities', [{'pid': 12346}]),
    ('post_worker_exit_identities', [{'pid': 12346, 'state': []}]),
    ('post_worker_exit_identities', [{'pid': 12346, 'state': 'not-a-process-state'}]),
])
def test_failed_profile_rejects_rehashed_malformed_cleanup_identities(profile_input, location, identity):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        report['status'] = 'failed'
        report['failed_phase'] = 'cleanup'
        report['profile'] = None
        report['errors'] = ['owned process cleanup was not proven']
        report['verification_errors'] = ['owned process cleanup was not proven']
        report['cleanup'].update({
            'worker_identity': {'pid': 12345, 'state': 'alive'},
            'post_worker_exit_identities': [{'pid': 12346, 'state': 'unknown'}],
            'ok': False,
            'root_removed': False,
        })
        report['cleanup'][location] = identity

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    errors = benchmark.profile_report_errors(report)
    assert any('cleanup' in error and 'identity' in error for error in errors)


def test_failed_profile_accepts_valid_live_cleanup_identity_states(profile_input):
    output, _report = complete_profile_artifacts(profile_input)

    def change(report):
        report['status'] = 'failed'
        report['failed_phase'] = 'cleanup'
        report['profile'] = None
        report['errors'] = ['owned process cleanup was not proven']
        report['verification_errors'] = ['owned process cleanup was not proven']
        report['cleanup'].update({
            'worker_identity': {'pid': 12345, 'state': 'alive'},
            'post_worker_exit_identities': [
                {'pid': 12346, 'state': 'unknown'},
                {'pid': 12347, 'state': 'zombie'},
                {'pid': 12348, 'state': 'reused'},
            ],
            'ok': False,
            'root_removed': False,
        })

    mutate_profile_report(output, change)
    report = json.loads((output / 'profile.json').read_text())
    hashes = json.loads((output / 'sha256.json').read_text())
    assert hashes['profile.json'] == hashlib.sha256((output / 'profile.json').read_bytes()).hexdigest()
    assert benchmark.profile_report_errors(report) == []


def test_offline_verifier_rejects_failed_report_even_when_artifacts_are_rehashed(profile_input):
    output, _report = complete_profile_artifacts(profile_input)

    def mark_failed(report):
        report['status'] = 'failed'
        report['failed_phase'] = 'copy'
        report['profile'] = None
        report['errors'] = ['intentional failed profile report']
        report['verification_errors'] = ['intentional failed profile report']

    mutate_profile_report(output, mark_failed)

    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


def test_offline_verifier_rejects_rehashed_non_string_failed_phase_without_traceback(
    tmp_path, profile_input,
):
    output, _report = complete_profile_artifacts(profile_input)

    def mark_failed_with_invalid_phase(report):
        report['status'] = 'failed'
        report['failed_phase'] = []
        report['profile'] = None
        report['errors'] = ['intentional failed profile report']
        report['verification_errors'] = ['intentional failed profile report']

    mutate_profile_report(output, mark_failed_with_invalid_phase)
    environment = private_cli_environment(tmp_path)
    result = run_cli([
        'profile', '--verify', str(output), '--result-root', str(profile_input[0].result_root),
    ], env=environment)

    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    assert 'failed profile report must name its first failing phase' in result.stderr


@pytest.mark.parametrize('label', ['warmup', 'target'])
def test_offline_verifier_rejects_rehashed_failed_sample_in_complete_report(
    tmp_path, profile_input, label,
):
    output, _report = complete_profile_artifacts(profile_input)
    mutate_profile_report(
        output,
        lambda report: report[label]['sample'].__setitem__('outcome', 'failure'),
    )

    result = run_cli(
        ['profile', '--verify', str(output), '--result-root', str(profile_input[0].result_root)],
        env=private_cli_environment(tmp_path),
    )

    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    assert f'profile report {label} sample outcome must be success' in result.stderr


@pytest.mark.parametrize('failure', ['missing_file', 'wrong_file_hash', 'changed_manifest'])
def test_offline_verifier_rejects_missing_or_tampered_artifacts(profile_input, failure):
    output, _report = complete_profile_artifacts(profile_input)
    if failure == 'missing_file':
        (output / 'worker.stdout.log').unlink()
    elif failure == 'wrong_file_hash':
        hashes = json.loads((output / 'sha256.json').read_text())
        hashes['worker.stdout.log'] = 'f' * 64
        benchmark.atomic_profile_json(output / 'sha256.json', hashes)
    else:
        with (output / 'version-manifest.json').open('ab') as stream:
            stream.write(b'changed')
    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


def test_profile_verify_cli_rejects_rehashed_malformed_url_without_traceback(profile_input, tmp_path):
    output, _report = complete_profile_artifacts(profile_input)
    mutate_profile_report(
        output,
        lambda report: report['warmup']['sample'].__setitem__('url', 'http://[::1'),
    )

    result = run_cli(
        ['profile', '--verify', str(output), '--result-root', str(profile_input[0].result_root)],
        env=private_cli_environment(tmp_path),
    )

    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    assert 'URL' in result.stderr


def test_profile_verify_cli_rejects_nul_hash_name_without_traceback(profile_input, tmp_path):
    output, _report = complete_profile_artifacts(profile_input)
    hashes = json.loads((output / 'sha256.json').read_text())
    hashes['invalid\x00name'] = '0' * 64
    benchmark.atomic_profile_json(output / 'sha256.json', hashes)

    result = run_cli(
        ['profile', '--verify', str(output), '--result-root', str(profile_input[0].result_root)],
        env=private_cli_environment(tmp_path),
    )

    assert result.returncode == 1
    assert 'Traceback' not in result.stderr
    assert 'basename' in result.stderr


@pytest.mark.parametrize('hash_name', ['/tmp/absolute.json', 'nested/file.json', '../traversal.json'])
def test_offline_verifier_rejects_non_basename_hash_entries(profile_input, hash_name):
    output, _report = complete_profile_artifacts(profile_input)
    hashes = json.loads((output / 'sha256.json').read_text())
    hashes[hash_name] = '0' * 64
    benchmark.atomic_profile_json(output / 'sha256.json', hashes)
    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1


def test_offline_verifier_rejects_symlink_and_non_regular_artifacts(profile_input, tmp_path):
    output, _report = complete_profile_artifacts(profile_input)
    log = output / 'worker.stdout.log'
    log.unlink()
    outside = tmp_path / 'outside.log'
    outside.write_text('not the worker log')
    log.symlink_to(outside)
    verify = profile_function('verify_profile')
    assert verify(output, profile_input[0].result_root) == 1

    shutil.rmtree(output)
    output, _report = complete_profile_artifacts(profile_input)
    log = output / 'worker.stderr.log'
    log.unlink()
    log.mkdir()
    assert verify(output, profile_input[0].result_root) == 1


def test_offline_verifier_uses_shared_physical_temp_path_rules(profile_input, tmp_path):
    output, _report = complete_profile_artifacts(profile_input)
    verify = profile_function('verify_profile')
    outside = tmp_path / 'outside'
    outside.mkdir(mode=0o700)
    outside.chmod(0o700)
    assert verify(outside, profile_input[0].result_root) == 1
    assert verify(str(output.parent / '..' / output.name), profile_input[0].result_root) == 1

    alias = output.parent / 'alias'
    alias.symlink_to(output.parent, target_is_directory=True)
    assert verify(alias / output.name, profile_input[0].result_root) == 1


def test_profile_verify_cli_is_offline_and_does_not_modify_artifacts(profile_input, monkeypatch):
    output, _report = complete_profile_artifacts(profile_input)
    profile_function('verify_profile')
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns, stat.S_IMODE(path.stat().st_mode))
        for path in output.iterdir()
    }
    for name in ('NAV_BENCH_CANDIDATE_SOURCE', 'NAV_BENCH_CHROME', 'NAV_BENCH_VERSION_MANIFEST'):
        monkeypatch.delenv(name, raising=False)
    for name in tuple(os.environ):
        if name.startswith('AWS_') and name != 'AWS_EC2_METADATA_DISABLED':
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(benchmark, 'git_value', lambda *_args: pytest.fail('verify read Git state'))
    monkeypatch.setattr(benchmark.subprocess, 'Popen', lambda *_args, **_kwargs: pytest.fail('verify launched a process'))
    monkeypatch.setattr(benchmark.asyncio, 'create_subprocess_exec', lambda *_args, **_kwargs: pytest.fail('verify launched a worker'))
    monkeypatch.setattr(benchmark, 'Arm', lambda *_args, **_kwargs: pytest.fail('verify constructed a browser arm'))
    assert benchmark.dispatch([
        'profile', '--verify', str(output), '--result-root', str(profile_input[0].result_root),
    ]) == 0
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns, stat.S_IMODE(path.stat().st_mode))
        for path in output.iterdir()
    }
    assert after == before


def test_offline_verifier_accepts_one_complete_fast_profile(profile_input):
    request, parent, manifest_path, chrome = profile_input
    fast_request = replace(
        request, workload='fast', output=profile_output_path(request.result_root, workload='fast'),
    )
    output, report = complete_profile_artifacts((fast_request, parent, manifest_path, chrome))
    verify = profile_function('verify_profile')
    assert report['workload'] == 'fast'
    assert verify(output, profile_input[0].result_root) == 0


def test_profile_verify_command_needs_no_source_chrome_manifest_or_aws_variables(profile_input, tmp_path):
    output, _report = complete_profile_artifacts(profile_input)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    env = private_cli_environment(tmp_path)
    result = run_cli([
        'profile', '--verify', str(output), '--result-root', str(profile_input[0].result_root),
    ], env=env)
    assert result.returncode == 0, result.stderr
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in output.iterdir()
    }
    assert after == before


def test_profile_verify_rejects_mixed_modes_and_returns_one_for_bad_directory(profile_input):
    output, _report = complete_profile_artifacts(profile_input)
    profile_function('verify_profile')
    with pytest.raises(SystemExit) as malformed:
        benchmark.dispatch([
            'profile', '--verify', str(output), '--result-root', str(profile_input[0].result_root),
            '--workload', 'fast',
        ])
    assert malformed.value.code == 2
    assert benchmark.dispatch([
        'profile', '--verify', str(output.parent / 'missing'),
        '--result-root', str(profile_input[0].result_root),
    ]) == 1
