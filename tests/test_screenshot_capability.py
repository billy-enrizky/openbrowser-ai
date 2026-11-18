import importlib.util
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import urllib.request
import uuid
from types import SimpleNamespace

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'screenshot_capability.py'
sys.path.insert(0, str(SCRIPT.parent))
import latest_benchmark_preflight as preflight_module
spec = importlib.util.spec_from_file_location('screenshot_capability_tests', SCRIPT)
capability = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capability
spec.loader.exec_module(capability)


def test_screenshot_run_rejects_changed_frozen_smoke_preflight_hashes(
    private_result_root, monkeypatch,
):
    root = private_result_root
    source = SCRIPT.parents[1]

    def write_private_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        path.chmod(0o600)

    executed_files = {
        relative: capability.navigation.sha256(source / relative)
        for relative in (
            'benchmarks/navigation_compare.py',
            'benchmarks/screenshot_capability.py',
            'benchmarks/screenshot_lightpanda.js',
        )
    }
    identity = {'executed_files': executed_files}
    identity_path = root / 'candidate-identity.json'
    write_private_json(identity_path, identity)
    identity_hash = capability.navigation.sha256(identity_path)
    candidate_freeze = root / 'candidate-freeze.txt'
    browser_freeze = root / 'browser-use-freeze.txt'
    tool_manifest = root / 'tool-manifest.json'
    candidate_freeze.write_text('openbrowser-ai==0.1.56\n')
    browser_freeze.write_text('browser-use==1.0.6\n')
    tool_manifest.write_text('{}\n')
    for path in (candidate_freeze, browser_freeze, tool_manifest):
        path.chmod(0o600)
    tool_hash = capability.navigation.sha256(tool_manifest)
    version_path = root / 'version-manifest.json'
    write_private_json(version_path, {
        'schema': 'openbrowser-latest-version-manifest/1',
        'candidate_identity_sha256': identity_hash,
        'tool_manifest_sha256': tool_hash,
    })
    version_hash = capability.navigation.sha256(version_path)
    run_preflight = root / 'preflight-screenshot-run.json'
    write_private_json(run_preflight, {
        'schema': 'openbrowser-latest-preflight/1',
        'phase': 'screenshot-run',
        'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'checks': [], 'outcome': 'passed', 'errors': [],
    })
    smoke_preflight = root / 'preflight-screenshot-smoke.json'
    write_private_json(smoke_preflight, {
        'schema': 'openbrowser-latest-preflight/1',
        'phase': 'screenshot-smoke',
        'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'checks': [{'name': 'fixture', 'expected': True, 'actual': True, 'state': 'passed'}],
        'outcome': 'passed', 'errors': [],
    })
    nav_smoke_preflight = root / 'preflight-navigation-smoke.json'
    write_private_json(nav_smoke_preflight, {
        'schema': 'openbrowser-latest-preflight/1',
        'phase': 'navigation-smoke',
        'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'checks': [{'name': 'fixture', 'expected': True, 'actual': True, 'state': 'passed'}],
        'outcome': 'passed', 'errors': [],
    })
    frozen_path = root / 'frozen-harness.json'
    frozen = {
        'schema': 'openbrowser-frozen-harness/1',
        'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'tool_manifest_sha256': tool_hash,
        'candidate_freeze_sha256': capability.navigation.sha256(candidate_freeze),
        'browser_use_freeze_sha256': capability.navigation.sha256(browser_freeze),
        'smoke_preflight_sha256': {
            'navigation': capability.navigation.sha256(nav_smoke_preflight),
            'screenshot': '0' * 64,
        },
        'smoke_manifest_sha256': {'navigation': '1' * 64, 'screenshot': '2' * 64},
        'executed_file_sha256': executed_files,
    }
    write_private_json(frozen_path, frozen)

    monkeypatch.setattr(capability, 'benchmark_preflight', preflight_module)
    monkeypatch.setattr(preflight_module, '_validate_frozen_harness', lambda value, **_kwargs: value)
    monkeypatch.setattr(
        preflight_module, 'validate_frozen_smoke_preflight_hashes',
        lambda *_args: (_ for _ in ()).throw(ValueError('frozen smoke preflight hash mismatch')),
    )
    args = SimpleNamespace(
        command='run', candidate_identity=str(identity_path),
        version_manifest=str(version_path), preflight=str(run_preflight),
        frozen_harness=str(frozen_path), source=str(source), python=sys.executable,
        chrome=None, lightpanda=None, output=str(root / 'screenshot'),
    )

    with pytest.raises(ValueError, match='frozen smoke preflight hash'):
        capability.validate_screenshot_inputs(args, root)


def test_screenshot_verifier_rejects_changed_frozen_smoke_preflight_hashes(
    private_result_root, monkeypatch,
):
    root = private_result_root
    source = SCRIPT.parents[1]

    def write_private_json(path, value):
        path.write_text(json.dumps(value, sort_keys=True) + '\n')
        path.chmod(0o600)

    executed_files = {
        relative: capability.navigation.sha256(source / relative)
        for relative in (
            'benchmarks/navigation_compare.py',
            'benchmarks/screenshot_capability.py',
            'benchmarks/screenshot_lightpanda.js',
        )
    }
    identity = {
        'schema': 'openbrowser-candidate-identity/1',
        'executed_files': executed_files,
    }
    identity_path = root / 'candidate-identity.json'
    write_private_json(identity_path, identity)
    identity_hash = capability.navigation.sha256(identity_path)
    tool_hash = 'd' * 64
    version_path = root / 'version-manifest.json'
    write_private_json(version_path, {
        'schema': 'openbrowser-latest-version-manifest/1',
        'candidate_identity_sha256': identity_hash,
        'tool_manifest_sha256': tool_hash,
    })
    version_hash = capability.navigation.sha256(version_path)
    preflight_path = root / 'preflight-screenshot-run.json'
    write_private_json(preflight_path, {
        'schema': 'openbrowser-latest-preflight/1', 'phase': 'screenshot-run',
        'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'checks': [{'name': 'fixture', 'expected': True, 'actual': True, 'state': 'passed'}],
        'outcome': 'passed', 'errors': [],
    })
    for name in ('navigation', 'screenshot'):
        smoke_preflight = root / f'preflight-{name}-smoke.json'
        write_private_json(smoke_preflight, {
            'schema': 'openbrowser-latest-preflight/1', 'phase': f'{name}-smoke',
            'candidate_identity_sha256': identity_hash,
            'version_manifest_sha256': version_hash,
            'checks': [{'name': 'fixture', 'expected': True, 'actual': True, 'state': 'passed'}],
            'outcome': 'passed', 'errors': [],
        })
    candidate_freeze = root / 'candidate-freeze.txt'
    browser_freeze = root / 'browser-use-freeze.txt'
    candidate_freeze.write_text('openbrowser-ai==0.1.56\n')
    browser_freeze.write_text('browser-use==1.0.6\n')
    for path in (candidate_freeze, browser_freeze):
        path.chmod(0o600)
    frozen = {
        'schema': 'openbrowser-frozen-harness/1',
        'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'tool_manifest_sha256': tool_hash,
        'candidate_freeze_sha256': capability.navigation.sha256(candidate_freeze),
        'browser_use_freeze_sha256': capability.navigation.sha256(browser_freeze),
        'smoke_preflight_sha256': {
            'navigation': capability.navigation.sha256(root / 'preflight-navigation-smoke.json'),
            'screenshot': '0' * 64,
        },
        'smoke_manifest_sha256': {'navigation': '1' * 64, 'screenshot': '2' * 64},
        'executed_file_sha256': executed_files,
    }
    frozen_path = root / 'frozen-harness.json'
    write_private_json(frozen_path, frozen)
    tool_manifest_path = root / 'tool-manifest.json'
    tool_manifest_path.write_text('{}\n')
    tool_manifest_path.chmod(0o600)
    run_root = root / 'screenshot'
    run_root.mkdir(mode=0o700)
    manifest = {
        'schema': capability.RUN_SCHEMA, 'mode': 'run', 'run_id': 'c' * 32,
        'state': 'complete', 'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': version_hash,
        'preflight_sha256': capability.navigation.sha256(preflight_path),
        'frozen_harness_sha256': capability.navigation.sha256(frozen_path),
        'tool_manifest_sha256': tool_hash,
        'harness_sha256': capability.navigation.sha256(SCRIPT),
        'lightpanda_script_sha256': capability.navigation.sha256(SCRIPT.with_name('screenshot_lightpanda.js')),
    }
    write_private_json(run_root / 'run-manifest.json', manifest)
    verifier_calls = []
    monkeypatch.setattr(
        capability.benchmark_preflight, '_validate_frozen_harness',
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(capability.benchmark_preflight, '_validate_identity', lambda value: value)
    monkeypatch.setattr(capability.benchmark_preflight, '_validate_version_manifest', lambda value: value)
    monkeypatch.setattr(
        capability, 'verify_screenshot_manifest',
        lambda *_args: verifier_calls.append(True) or [],
    )
    args = SimpleNamespace(
        command='verify', candidate_identity=str(identity_path),
        preflight=str(preflight_path), frozen_harness=str(frozen_path),
    )

    assert capability.verify_run_directory(run_root, root, args) == 1
    assert verifier_calls == []


@pytest.fixture
def private_result_root():
    with tempfile.TemporaryDirectory(dir=Path('/tmp').resolve()) as directory:
        root = Path(directory)
        root.chmod(0o700)
        yield root


def attempt(state, signature=None):
    return {
        'state': state,
        'mismatch_signature': signature,
        'parseable_png': state in ('passed', 'pixel_mismatch'),
        'image_request_ok': state in ('passed', 'pixel_mismatch'),
        'cleanup_ok': state != 'incomplete',
    }


def screenshot_png_128(index):
    image = Image.new('RGB', (128, 128), (255, 255, 255))
    proof = Image.open(BytesIO(capability.png_for_index(index))).convert('RGB')
    image.paste(proof, (0, 0))
    output = BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def verification_attempt(result_root, *, worker_state='absent'):
    run_root = result_root / 'screenshot'
    attempt_dir = run_root / 'arms' / 'openbrowser' / 'attempt-1'
    for directory in (run_root, run_root / 'arms', run_root / 'arms' / 'openbrowser', attempt_dir):
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
    run_id = 'c' * 32
    nonce = f'{run_id}-openbrowser-0'
    runtime_root = Path('/tmp').resolve(strict=True) / f'nb-screenshot-verify-{uuid.uuid4().hex}'
    worker_root = runtime_root / 'openbrowser'
    identity = {
        'pid': 1000,
        'created': 1.0,
        'cmdline': ['/tmp/openbrowser-worker'],
        'owner_token': 'verification-owner-token',
        'owner_root': str(worker_root),
    }
    request_proof = {
        'nonce': nonce,
        'count': 1,
        'successful_count': 1,
        'ok': True,
        'image_requests': [{
            'method': 'GET', 'path': '/pixels.png', 'nonce': nonce, 'status': 200,
        }],
    }
    attempt = {
        'attempt_id': f'{run_id}:openbrowser:0',
        'arm': 'openbrowser',
        'index': 0,
        'nonce': nonce,
        'request': capability.screenshot_request(run_id, 'openbrowser', 0, 12345),
        'directory': 'arms/openbrowser/attempt-1',
        'state': 'failed',
        'parseable_png': False,
        'image_request_ok': True,
        'cleanup_ok': True,
        'runtime_root_removed': True,
        'runtime_root': str(runtime_root),
        'worker_identity': identity,
        'worker_cleanup': {
            'ok': True,
            'worker_identity': {**identity, 'state': worker_state},
            'post_worker_exit_identities': [],
            'errors': [],
        },
        'request_proof': request_proof,
        'png': None,
        'pixel': {
            'parseable': False, 'probe_count': 0, 'mismatch_count': 0,
            'mismatch_signature': None,
        },
        'mismatch_signature': None,
        'errors': ['screenshot worker failed after startup'],
    }
    return run_root, attempt


def test_screenshot_smoke_never_classifies_lightpanda():
    assert capability.arms_for_mode('smoke') == ('openbrowser', 'raw-cdp')
    assert 'lightpanda' not in capability.arms_for_mode('smoke')


def test_screenshot_run_and_attempt_states_reject_illegal_transitions():
    run_state = capability.transition_run('preparing', 'running')
    run_state = capability.transition_run(run_state, 'cleaning')
    assert capability.transition_run(run_state, 'complete') == 'complete'

    attempt_state = capability.transition_attempt('pending', 'running')
    assert capability.transition_attempt(attempt_state, 'pixel_mismatch') == 'pixel_mismatch'
    with pytest.raises(ValueError, match='illegal screenshot attempt transition'):
        capability.transition_attempt('pixel_mismatch', 'running')


def test_screenshot_fixture_uses_nonce_indexed_four_color_formula():
    index = 3
    image = Image.open(BytesIO(capability.png_for_index(index)))
    image.load()

    assert image.size == (64, 64)
    assert [image.convert('RGB').getpixel(point) for point in ((16, 16), (48, 16), (16, 48), (48, 48))] == [
        capability.expected_rgb(index, quadrant) for quadrant in range(4)
    ]


def test_screenshot_fixture_serves_one_nonce_scoped_image_request():
    run_id = 'b' * 32
    nonce = f'{run_id}-openbrowser-2'
    fixture = capability.ScreenshotFixture(run_id)
    thread = threading.Thread(target=fixture.serve_forever, daemon=True)
    thread.start()
    try:
        base = f'http://127.0.0.1:{fixture.server_address[1]}'
        with urllib.request.urlopen(f'{base}/screenshot?nonce={nonce}', timeout=2) as response:
            html = response.read().decode()
        assert 'id="proof"' in html
        assert '128px' in html
        assert nonce in html
        with urllib.request.urlopen(f'{base}/pixels.png?nonce={nonce}', timeout=2) as response:
            content = response.read()
            assert response.headers['Content-Type'] == 'image/png'
        image = Image.open(BytesIO(content)).convert('RGB')
        assert image.size == (64, 64)
        assert image.getpixel((16, 16)) == capability.expected_rgb(2, 0)
        assert len(fixture.snapshot(nonce)['image_requests']) == 1
    finally:
        fixture.shutdown()
        fixture.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_parseable_pixel_mismatch_is_completed_work_not_an_execution_error():
    image = Image.new('RGB', (128, 128), (0, 0, 0))
    output = BytesIO()
    image.save(output, format='PNG')
    evidence = capability.inspect_png(output.getvalue(), index=0)

    assert evidence['parseable'] is True
    assert evidence['probe_count'] == 4
    assert evidence['mismatch_count'] == 4
    assert evidence['mismatch_signature'] == 'probes=0,1,2,3'


def test_screenshot_work_counts_separate_pixel_mismatch_from_execution_errors():
    attempts = [
        {'state': 'passed', 'pixel': {'parseable': True, 'probe_count': 4, 'mismatch_count': 0}, 'errors': []},
        {'state': 'pixel_mismatch', 'pixel': {'parseable': True, 'probe_count': 4, 'mismatch_count': 4}, 'errors': []},
        {'state': 'failed', 'pixel': {'parseable': False, 'probe_count': 0, 'mismatch_count': 0}, 'errors': ['PNG decode failed']},
    ]

    assert capability.screenshot_work_counts(attempts) == {
        'attempted_work_count': 3,
        'completed_work_count': 2,
        'successful_work_count': 1,
        'pixel_mismatch_attempt_count': 1,
        'pixel_probe_count': 8,
        'pixel_probe_mismatch_count': 4,
        'error_count': 1,
    }


def test_unsupported_requires_six_matching_pixel_mismatches():
    attempts = [attempt('pixel_mismatch', 'probes:0,1,2,3') for _ in range(6)]
    controls = {'openbrowser': 'passed', 'raw-cdp': 'passed'}

    assert capability.classify_arm_status(
        'lightpanda', attempts, controls, fixture_stopped=True,
    ) == 'unsupported'
    assert capability.classify_arm_status(
        'lightpanda', attempts[:5], controls, fixture_stopped=True,
    ) == 'incomplete'
    mixed = [*attempts[:5], attempt('pixel_mismatch', 'dimensions:64x64')]
    assert capability.classify_arm_status(
        'lightpanda', mixed, controls, fixture_stopped=True,
    ) == 'inconclusive'


def test_fixture_cleanup_precedes_arm_status():
    attempts = [attempt('passed') for _ in range(6)]

    assert capability.classify_arm_status(
        'openbrowser', attempts, {}, fixture_stopped=False,
    ) == 'incomplete'


def test_screenshot_run_exits_zero_for_complete_failed_arm():
    attempts = [
        {'arm': arm, 'index': index, 'state': ('failed' if arm == 'lightpanda' else 'passed'),
         'cleanup_ok': True, 'runtime_root_removed': True}
        for arm in capability.SCREENSHOT_ARMS for index in range(6)
    ]
    manifest = {
        'schema': 'openbrowser-screenshot-capability-run/1',
        'mode': 'run',
        'state': 'complete',
        'cleanup': {'ok': True, 'attempts_ok': True, 'runtime_roots_removed': True},
        'fixture': {
            'started': True, 'stopped': True, 'socket_closed': True,
            'thread_stopped': True,
        },
        'errors': [],
        'attempts': attempts,
        'arm_status': {
            'openbrowser': 'passed', 'raw-cdp': 'passed', 'lightpanda': 'failed',
        },
    }

    assert capability.screenshot_run_exit_code(manifest) == 0


def test_screenshot_run_exits_one_for_incomplete_cleanup():
    manifest = {
        'schema': 'openbrowser-screenshot-capability-run/1',
        'mode': 'run',
        'state': 'incomplete',
        'cleanup': {'ok': False},
        'fixture': {'stopped': False},
        'attempts': [],
        'arm_status': {},
    }

    assert capability.screenshot_run_exit_code(manifest) == 1


def test_screenshot_run_exits_two_for_invalid_path(tmp_path):
    args = [
        'smoke', '--candidate-identity', str(tmp_path / 'identity.json'),
        '--preflight', str(tmp_path / 'preflight.json'),
        '--version-manifest', str(tmp_path / 'version.json'),
        '--source', str(tmp_path), '--python', sys.executable,
        '--chrome', str(tmp_path / 'chrome'), '--output', str(tmp_path / 'outside'),
    ]

    assert capability.dispatch(args) == 2


def test_screenshot_run_exit_code_rejects_non_object_attempt_records():
    manifest = {
        'schema': capability.RUN_SCHEMA,
        'mode': 'run',
        'state': 'complete',
        'cleanup': {'ok': True, 'attempts_ok': True, 'runtime_roots_removed': True},
        'fixture': {
            'started': True, 'stopped': True, 'socket_closed': True,
            'thread_stopped': True,
        },
        'errors': [],
        'attempts': [None] * 18,
    }

    try:
        result = capability.screenshot_run_exit_code(manifest)
    except Exception as error:
        pytest.fail(f'run exit validation raised instead of rejecting the artifact: {type(error).__name__}: {error}')

    assert result == 1


def test_screenshot_verifier_rejects_non_object_attempt_records(private_result_root):
    run_root = private_result_root / 'screenshot'
    manifest = {
        'schema': capability.RUN_SCHEMA,
        'mode': 'run',
        'run_id': 'c' * 32,
        'state': 'complete',
        'candidate_identity_sha256': 'a' * 64,
        'version_manifest_sha256': 'b' * 64,
        'preflight_sha256': 'c' * 64,
        'frozen_harness_sha256': 'd' * 64,
        'tool_manifest_sha256': 'e' * 64,
        'harness_sha256': 'f' * 64,
        'lightpanda_script_sha256': '1' * 64,
        'arms': list(capability.SCREENSHOT_ARMS),
        'attempts': [None],
        'arm_status': {},
        'work_counts': capability.screenshot_work_counts([]),
        'fixture': {
            'started': True, 'stopped': True, 'socket_closed': True,
            'thread_stopped': True,
        },
        'cleanup': {'ok': True, 'attempts_ok': True, 'runtime_roots_removed': True},
        'errors': [],
    }

    try:
        errors = capability.verify_screenshot_manifest(run_root, manifest, 'run')
    except Exception as error:
        pytest.fail(f'offline verification raised instead of rejecting the artifact: {type(error).__name__}: {error}')

    assert errors


def test_screenshot_verifier_rejects_malformed_image_request_records(private_result_root):
    run_root, attempt = verification_attempt(private_result_root)
    attempt['request_proof']['image_requests'] = [None]

    try:
        errors = capability.verify_attempt_artifacts(run_root, attempt)
    except Exception as error:
        pytest.fail(f'offline verification raised instead of rejecting the artifact: {type(error).__name__}: {error}')

    assert any('request proof' in error for error in errors)


def test_screenshot_verifier_rejects_unretired_worker_after_failed_attempt(private_result_root):
    run_root, attempt = verification_attempt(private_result_root, worker_state='alive')

    errors = capability.verify_attempt_artifacts(run_root, attempt)

    assert any('worker is not proven retired' in error for error in errors)


@pytest.mark.asyncio
async def test_lightpanda_attempt_retires_owned_child_after_parent_exit(
    private_result_root, monkeypatch,
):
    attempt_dir = private_result_root / 'screenshot' / 'arms' / 'lightpanda' / 'attempt-1'
    for directory in (
        private_result_root / 'screenshot',
        private_result_root / 'screenshot' / 'arms',
        private_result_root / 'screenshot' / 'arms' / 'lightpanda',
        attempt_dir,
    ):
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.chmod(0o700)
    runtime_root = Path('/tmp').resolve(strict=True) / f'nb-screenshot-orphan-{uuid.uuid4().hex}'
    runtime_root.mkdir(mode=0o700)
    token = uuid.uuid4().hex
    (runtime_root / '.benchmark-owner').write_text(token)
    run_id = 'd' * 32
    request = capability.screenshot_request(run_id, 'lightpanda', 0, 12345)
    attempt = {
        'attempt_id': f'{run_id}:lightpanda:0',
        'arm': 'lightpanda',
        'index': 0,
        'nonce': request['nonce'],
        'request': request,
    }

    class FakeProcessRef:
        def __init__(self, pid):
            self.pid = pid

    class FinishedProcess:
        pid = 52000

        async def wait(self):
            return 0

    child = FakeProcessRef(52001)
    group_members = [child]
    child_is_alive = True
    terminate_calls = []
    process_started_new_session = []
    identities = {
        52000: {
            'pid': 52000, 'created': 1.0, 'cmdline': ['/tmp/lightpanda', 'run'],
            'owner_token': token, 'owner_root': str(runtime_root / 'lightpanda'),
        },
        52001: {
            'pid': 52001, 'created': 2.0, 'cmdline': ['/tmp/lightpanda', 'child'],
            'owner_token': token, 'owner_root': str(runtime_root / 'lightpanda'),
        },
    }

    async def spawn_process(*_args, **kwargs):
        process_started_new_session.append(kwargs.get('start_new_session') is True)
        return FinishedProcess()

    async def terminate_owned(processes, _token, grace=2):
        nonlocal child_is_alive
        terminate_calls.extend(identity['pid'] for identity in processes)
        child_is_alive = False
        group_members.clear()
        return [{**identity, 'state': 'absent'} for identity in processes]

    monkeypatch.setattr(capability, 'new_runtime_root', lambda: (runtime_root, token))
    monkeypatch.setattr(
        capability.navigation,
        'make_worker_launch',
        lambda **_kwargs: SimpleNamespace(lightpanda=Path('/tmp/lightpanda')),
    )
    monkeypatch.setattr(capability.navigation, 'worker_environment', lambda _launch: {})
    monkeypatch.setattr(capability.navigation, 'prepare_worker_directories', lambda _environment: None)
    monkeypatch.setattr(capability.navigation.asyncio, 'create_subprocess_exec', spawn_process)
    monkeypatch.setattr(capability.navigation.psutil, 'Process', FakeProcessRef)
    monkeypatch.setattr(capability.navigation.psutil, 'process_iter', lambda: list(group_members))
    monkeypatch.setattr(capability.os, 'getpgid', lambda _pid: 52000)
    monkeypatch.setattr(
        capability.navigation,
        'record',
        lambda process, _token: identities[process.pid],
    )
    monkeypatch.setattr(capability.navigation, 'profile_identity_errors', lambda *_args, **_kwargs: [])
    monkeypatch.setattr(capability.navigation, 'descendant_identities', lambda *_args: [])
    monkeypatch.setattr(
        capability.navigation,
        'identity_state',
        lambda identity: 'absent' if identity['pid'] == 52000 or not child_is_alive else 'alive',
    )
    monkeypatch.setattr(capability.navigation, 'terminate_owned', terminate_owned)
    monkeypatch.setattr(capability.navigation, 'removal_allowed', lambda *_args: True)
    monkeypatch.setattr(capability, 'image_request_proof', lambda _fixture, nonce: {
        'nonce': nonce, 'image_requests': [{'method': 'GET', 'path': '/pixels.png',
                                            'nonce': nonce, 'status': 200}],
        'count': 1, 'successful_count': 1, 'ok': True,
    })
    monkeypatch.setattr(capability, 'verify_png_file', lambda *_args, **_kwargs: {
        'file': 'result.png', 'byte_count': 4, 'sha256': 'a' * 64,
        'pixel': {'parseable': True, 'mismatch_signature': None},
    })

    try:
        result = await capability.execute_lightpanda_attempt(
            SimpleNamespace(source=str(private_result_root), python=sys.executable,
                            lightpanda='/tmp/lightpanda'),
            object(), attempt_dir, attempt,
        )
    except Exception as error:
        pytest.fail(f'Lightpanda one-shot request did not reach cleanup: {type(error).__name__}: {error}')

    assert process_started_new_session == [True], result
    assert terminate_calls == [52001], result['worker_cleanup']
    assert result['cleanup_ok'] is True
    assert result['runtime_root_removed'] is True
    assert result['worker_cleanup'].get('process_state') == 'verified'


def test_lightpanda_cleanup_state_machine_rejects_illegal_transitions():
    transition = getattr(capability, 'transition_lightpanda_cleanup', None)
    assert callable(transition), 'Lightpanda cleanup lifecycle has no explicit transition model'
    assert transition('inspecting', 'found_live_process') == 'terminating'
    assert transition('terminating', 'retry_inventory') == 'inspecting'
    assert transition('terminating', 'all_processes_retired') == 'verified'
    with pytest.raises(ValueError, match='illegal Lightpanda cleanup transition'):
        transition('verified', 'found_live_process')


def test_screenshot_verify_exit_codes(tmp_path):
    assert capability.dispatch([
        'verify', '--run-root', str(tmp_path / 'outside'),
        '--frozen-harness', str(tmp_path / 'frozen.json'),
        '--candidate-identity', str(tmp_path / 'identity.json'),
        '--preflight', str(tmp_path / 'preflight.json'),
    ]) == 2

    with tempfile.TemporaryDirectory(dir=Path('/tmp').resolve()) as directory:
        result_root = Path(directory)
        result_root.chmod(0o700)
        run_root = result_root / 'screenshot'
        run_root.mkdir(mode=0o700)
        assert capability.dispatch([
            'verify', '--run-root', str(run_root),
            '--frozen-harness', str(result_root / 'frozen.json'),
            '--candidate-identity', str(result_root / 'identity.json'),
            '--preflight', str(result_root / 'preflight.json'),
        ]) == 1


def test_screenshot_verifier_rejects_a_tampered_png(private_result_root):
    smoke_root = private_result_root / 'smoke'
    smoke_root.mkdir(mode=0o700)
    smoke_root.chmod(0o700)
    run_root = smoke_root / 'screenshot'
    run_root.mkdir(mode=0o700)
    run_root.chmod(0o700)
    candidate_identity = private_result_root / 'candidate-identity.json'
    preflight_path = private_result_root / 'preflight-screenshot-smoke.json'
    interpreter = preflight_module._inspect_python(sys.executable, root=private_result_root)
    identity = {
        'schema': 'openbrowser-candidate-identity/1',
        'candidate_commit': 'a' * 40,
        'candidate_tree': 'b' * 40,
        'runtime_tree': 'c' * 40,
        'release_runtime_tree': 'c' * 40,
        'origin_main_commit': 'd' * 40,
        'origin_main_runtime_tree': 'c' * 40,
        'executed_files': {
            relative: 'e' * 64 for relative in preflight_module.EXECUTED_FILE_PATHS
        },
        'candidate_python': interpreter,
        'browser_use_python': interpreter,
    }
    identity_bytes = capability.json.dumps(identity).encode()
    candidate_identity.write_bytes(identity_bytes)
    candidate_identity.chmod(0o600)
    identity_hash = capability.navigation.sha256(candidate_identity)
    preflight = {
        'schema': 'openbrowser-latest-preflight/1', 'phase': 'screenshot-smoke',
        'candidate_identity_sha256': identity_hash, 'version_manifest_sha256': 'b' * 64,
        'checks': [], 'outcome': 'passed', 'errors': [],
    }
    capability.atomic_json(preflight_path, preflight)

    arms_root = run_root / 'arms'
    arms_root.mkdir(mode=0o700)
    arms_root.chmod(0o700)
    attempts = []
    for arm in capability.SMOKE_ARMS:
        arm_root = arms_root / arm
        arm_root.mkdir(mode=0o700)
        arm_root.chmod(0o700)
        attempt_dir = arm_root / 'attempt-1'
        attempt_dir.mkdir(mode=0o700)
        attempt_dir.chmod(0o700)
        content = screenshot_png_128(0)
        png_path = attempt_dir / 'result.png'
        png_path.write_bytes(content)
        png_path.chmod(0o600)
        pixel = capability.inspect_png(content, index=0)
        request = capability.screenshot_request('c' * 32, arm, 0, 12345)
        runtime_root = Path('/tmp').resolve(strict=True) / f'nb-screenshot-manual-{arm}'
        worker_identity = {
            'pid': 1000 + len(attempts),
            'created': 1.0 + len(attempts),
            'cmdline': ['worker'],
            'owner_token': 'owner',
            'owner_root': str(runtime_root / arm),
        }
        reply = {
            'schema': 'openbrowser-screenshot-reply/1',
            'attempt_id': request['attempt_id'], 'arm': arm,
            'nonce': request['nonce'], 'ready_marker': request['nonce'],
            'output': 'result.png', 'byte_count': len(content),
            'sha256': capability.navigation.sha256(png_path),
        }
        attempts.append({
            'attempt_id': f'{"c" * 32}:{arm}:0', 'arm': arm, 'index': 0,
            'nonce': f'{"c" * 32}-{arm}-0',
            'request': request,
            'directory': f'arms/{arm}/attempt-1', 'state': 'passed',
            'parseable_png': True, 'image_request_ok': True, 'cleanup_ok': True,
            'runtime_root_removed': True, 'runtime_root': str(runtime_root), 'errors': [],
            'worker_identity': worker_identity,
            'worker_reply': reply,
            'request_proof': {
                'nonce': f'{"c" * 32}-{arm}-0', 'count': 1, 'successful_count': 1,
                'ok': True,
                'image_requests': [{'method': 'GET', 'path': '/pixels.png',
                                    'nonce': f'{"c" * 32}-{arm}-0', 'status': 200}],
            },
            'png': {'file': 'result.png', 'byte_count': len(content),
                    'sha256': capability.navigation.sha256(png_path), 'pixel': pixel},
            'pixel': pixel,
            'worker_cleanup': {
                'ok': True,
                'worker_identity': {**worker_identity, 'state': 'absent'},
                'post_worker_exit_identities': [],
                'errors': [],
            },
        })
    manifest = {
        'schema': capability.SMOKE_SCHEMA, 'mode': 'smoke', 'run_id': 'c' * 32,
        'state': 'complete', 'candidate_identity_sha256': identity_hash,
        'version_manifest_sha256': 'b' * 64,
        'preflight_sha256': capability.navigation.sha256(preflight_path),
        'frozen_harness_sha256': None, 'tool_manifest_sha256': 'd' * 64,
        'harness_sha256': capability.navigation.sha256(capability.__file__),
        'lightpanda_script_sha256': capability.navigation.sha256(
            Path(capability.__file__).with_name('screenshot_lightpanda.js'),
        ),
        'arms': list(capability.SMOKE_ARMS), 'attempts': attempts, 'arm_status': {},
        'work_counts': capability.screenshot_work_counts(attempts),
        'fixture': {'started': True, 'stopped': True, 'socket_closed': True, 'thread_stopped': True},
        'cleanup': {'ok': True, 'attempts_ok': True, 'runtime_roots_removed': True},
        'errors': [],
    }
    capability.atomic_json(run_root / 'run-manifest.json', manifest)
    capability.write_screenshot_summary(run_root, manifest)

    args = SimpleNamespace(
        command='verify-smoke', candidate_identity=str(candidate_identity),
        preflight=str(preflight_path), frozen_harness=None,
    )
    assert capability.verify_run_directory(run_root, private_result_root, args) == 0
    (run_root / 'arms/openbrowser/attempt-1/result.png').write_bytes(
        (run_root / 'arms/openbrowser/attempt-1/result.png').read_bytes() + b'tampered',
    )
    assert capability.verify_run_directory(run_root, private_result_root, args) == 1


@pytest.mark.asyncio
async def test_screenshot_smoke_runs_only_the_two_pixel_controls(
    private_result_root, monkeypatch,
):
    output = private_result_root / 'smoke' / 'screenshot'
    args = SimpleNamespace(command='smoke', attempts_per_arm=1, output=str(output))
    monkeypatch.setattr(
        capability, 'validate_screenshot_inputs',
        lambda *_args: {
            'candidate_identity_sha256': 'a' * 64,
            'version_manifest_sha256': 'b' * 64,
            'preflight_sha256': 'c' * 64,
            'frozen_harness_sha256': None,
        },
    )
    launched = []

    async def passed_attempt(_args, _fixture, _run_root, record):
        launched.append(record['arm'])
        return {
            'state': 'passed', 'cleanup_ok': True,
            'runtime_root_removed': True,
            'parseable_png': True, 'image_request_ok': True,
            'pixel': {'parseable': True, 'probe_count': 4, 'mismatch_count': 0,
                      'mismatch_signature': None},
            'png': {'byte_count': 8, 'sha256': 'd' * 64},
            'errors': [],
        }

    monkeypatch.setattr(capability, 'execute_attempt', passed_attempt)
    code = await capability.run_screenshot(args, private_result_root, output)

    manifest = json.loads((output / 'run-manifest.json').read_text())
    assert code == 0
    assert launched == ['openbrowser', 'raw-cdp']
    assert manifest['mode'] == 'smoke'
    assert manifest['state'] == 'complete'
    assert manifest['arm_status'] == {}
    assert manifest['fixture']['stopped'] is True


@pytest.mark.asyncio
async def test_screenshot_run_stops_after_cleanup_becomes_uncertain(private_result_root, monkeypatch):
    output = private_result_root / 'screenshot'
    args = SimpleNamespace(command='run', attempts_per_arm=6, output=str(output))
    monkeypatch.setattr(
        capability, 'validate_screenshot_inputs',
        lambda *_args: {
            'candidate_identity_sha256': 'a' * 64,
            'version_manifest_sha256': 'b' * 64,
            'preflight_sha256': 'c' * 64,
            'frozen_harness_sha256': 'd' * 64,
        },
    )
    launched = []

    async def uncertain_attempt(_args, _fixture, _run_root, record):
        launched.append(record['arm'])
        return {
            'state': 'incomplete', 'cleanup_ok': False,
            'runtime_root_removed': False,
            'parseable_png': False, 'image_request_ok': False,
            'pixel': {'parseable': False, 'probe_count': 0, 'mismatch_count': 0,
                      'mismatch_signature': None},
            'png': None, 'errors': ['worker identity is unknown'],
        }

    monkeypatch.setattr(capability, 'execute_attempt', uncertain_attempt)
    code = await capability.run_screenshot(args, private_result_root, output)

    manifest = json.loads((output / 'run-manifest.json').read_text())
    assert code == 1
    assert launched == ['openbrowser']
    assert manifest['state'] == 'incomplete'
    assert manifest['fixture']['stopped'] is True


@pytest.fixture
def sealed_screenshot_inputs():
    from test_latest_benchmark_preflight import prepare_scenario

    scenario = prepare_scenario()
    root = scenario['result_root']
    phase = 'screenshot-smoke'
    preflight_path = root / f'preflight-{phase}.json'
    capability.atomic_json(preflight_path, {
        'schema': preflight_module.PREFLIGHT_SCHEMA,
        'phase': phase,
        'candidate_identity_sha256': capability.navigation.sha256(scenario['identity_path']),
        'version_manifest_sha256': capability.navigation.sha256(scenario['version_path']),
        'checks': [{'name': 'fixture', 'expected': True, 'actual': True, 'state': 'passed'}],
        'outcome': 'passed', 'errors': [],
    })
    tools = json.loads(scenario['tool_manifest_path'].read_text())['tools']
    args = SimpleNamespace(
        command='smoke', candidate_identity=scenario['identity_path'],
        version_manifest=scenario['version_path'], preflight=preflight_path,
        output=root / 'smoke/screenshot', source=scenario['source_root'],
        python=sys.executable, chrome=tools['chrome_for_testing']['path'],
        lightpanda=None, frozen_harness=None,
    )
    try:
        yield scenario, args
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(scenario['source_root'], ignore_errors=True)


def test_screenshot_smoke_accepts_sealed_executable_inputs(sealed_screenshot_inputs):
    scenario, args = sealed_screenshot_inputs
    result = capability.validate_screenshot_inputs(args, scenario['result_root'])
    assert result['candidate_commit'] == scenario['identity']['candidate_commit']


@pytest.mark.parametrize('changed', ['source', 'chrome_path', 'chrome_bytes', 'python'])
def test_screenshot_smoke_rejects_changed_executable_inputs(sealed_screenshot_inputs, changed):
    scenario, args = sealed_screenshot_inputs
    if changed == 'source':
        (scenario['source_root'] / 'src/openbrowser/config.py').write_text('CONFIG = None\n')
    elif changed == 'chrome_path':
        replacement = scenario['result_root'] / 'replacement-chrome'
        shutil.copyfile(args.chrome, replacement)
        replacement.chmod(0o700)
        args.chrome = replacement
    elif changed == 'chrome_bytes':
        Path(args.chrome).write_text('#!/bin/sh\necho changed\n')
    else:
        args.python = Path(sys.executable).resolve()

    with pytest.raises(ValueError):
        capability.validate_screenshot_inputs(args, scenario['result_root'])
