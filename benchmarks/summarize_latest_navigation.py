from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import stat
import uuid
from typing import Any
from urllib.parse import parse_qs, urlsplit

import latest_benchmark_preflight as preflight


CLI_DESCRIPTION = 'Verify local navigation benchmark artifacts and summarize verified pairs.'
LOG = logging.getLogger('summarize-latest-navigation')

RUN_SCHEMA = 'openbrowser-latest-navigation-run/1'
SMOKE_SCHEMA = 'openbrowser-latest-navigation-smoke/1'
ATTEMPT_SCHEMA = 'openbrowser-navigation-pair-attempt/1'
SUMMARY_SCHEMA = 'openbrowser-latest-navigation-summary/1'
RUN_MANIFEST_NAME = 'run-manifest.json'
ATTEMPT_MANIFEST_NAME = 'attempt-manifest.json'
SUMMARY_NAME = 'summary.json'
SMOKE_DELAY_MIN_NS = 100_000_000
OPENBROWSER_RUNTIME_VERSION = '0.1.56'
COMPETITORS = ('raw-cdp', 'lightpanda', 'agent-browser', 'browser-use')
WORKLOADS = ('fast', 'delayed')
ARMS_BY_MODE = {
    'smoke': ('openbrowser', 'raw-cdp'),
    'run': ('openbrowser', *COMPETITORS),
}
RUN_STATES = {'preparing', 'preflighted', 'running', 'verifying', 'complete', 'incomplete'}
PAIR_STATES = {'pending', 'running', 'verified', 'failed', 'contaminated'}
ATTEMPT_STATES = {'pending', 'running', 'verified', 'failed', 'contaminated'}
HOST_STATES = {'clear', 'contaminated', 'unknown'}
SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
RUN_ID_RE = re.compile(r'^[0-9a-f]{32}$')
OWNER_TOKEN_RE = re.compile(r'^[0-9a-f]{32}$')
UTC_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
SAMPLE_NONCE_RE = re.compile(r'^[0-9a-f]{32}$')

RUN_KEYS = {
    'schema', 'mode', 'run_id', 'state', 'created_at_utc',
    'candidate_identity_sha256', 'version_manifest_sha256', 'preflight_sha256',
    'frozen_harness_sha256', 'harness_sha256', 'source_commit', 'source_tree',
    'arms', 'workloads', 'rounds', 'samples_per_workload', 'pair_order_by_round',
    'pairs', 'host_preflight', 'work_counts', 'fixture', 'cleanup', 'failed_phase',
    'errors',
}
PAIR_KEYS = {
    'round', 'competitor', 'arms', 'state', 'attempts', 'retry_host_checks', 'errors',
}
ATTEMPT_REFERENCE_KEYS = {
    'attempt_number', 'state', 'directory', 'manifest_sha256', 'cleanup_ok',
    'runtime_root_removed', 'host_gate', 'work_counts', 'error_count', 'errors',
}
ATTEMPT_KEYS = {
    'schema', 'run_id', 'round', 'competitor', 'attempt_number', 'state',
    'execution_order', 'warmups', 'samples', 'host_samples', 'host_gate',
    'work_counts', 'cleanup', 'errors',
}
SAMPLE_RECORD_KEYS = {
    'arm', 'workload', 'index', 'nonce', 'url', 'warmup', 'state', 'sample',
    'request_proof', 'errors',
}
EXECUTION_ORDER_KEYS = {'workload', 'index', 'arm'}
HOST_GATE_KEYS = {'state', 'reasons'}
HOST_SAMPLE_KEYS = {
    'captured_at_utc', 'load_1m', 'logical_cpu_count', 'processes',
    'classification_errors',
}
HOST_PROCESS_KEYS = {'pid', 'name', 'cpu_percent', 'excluded'}
WORK_COUNT_KEYS = {'attempted_work_count', 'successful_work_count', 'error_count'}
RUN_WORK_COUNT_KEYS = WORK_COUNT_KEYS | {'host_contaminated_attempt_count'}
FIXTURE_KEYS = {'started', 'stopped', 'socket_closed', 'thread_stopped'}
RUN_CLEANUP_KEYS = {'ok', 'workers_ok', 'runtime_roots_removed', 'fixture_stopped'}
ATTEMPT_CLEANUP_KEYS = {
    'ok', 'workers_ok', 'runtime_root', 'runtime_root_removed', 'workers', 'errors',
}
WORKER_CLEANUP_KEYS = {
    'ok', 'fallback_used', 'result', 'worker_identity',
    'post_worker_exit_identities', 'error',
}
WORKER_STOP_RESULT_KEYS = {'identities', 'errors'}
PROCESS_IDENTITY_KEYS = {'pid', 'created', 'cmdline', 'owner_token', 'owner_root', 'state'}
PROCESS_IDENTITY_WITHOUT_STATE_KEYS = PROCESS_IDENTITY_KEYS - {'state'}
SAMPLE_REQUEST_KEYS = {
    'op', 'arm', 'kind', 'index', 'nonce', 'url', 'start_ns',
    'navigation_return_ns', 'end_ns', 'elapsed_ns', 'navigation_ns',
    'post_navigation_ns', 'checked_ns', 'navigation_observation',
    'effect_observation', 'final_observation', 'outcome', 'server',
}
OBSERVATION_KEYS = {
    'url', 'title', 'readyState', 'marker', 'loaded', 'scriptDone', 'field',
}
LEDGER_KEYS = {'requests', 'effects'}
SUCCESS_EVENT_KEYS = {
    'path', 'nonce', 'method', 'received_ns', 'status', 'bytes', 'completed_ns',
}
FAILURE_EVENT_KEYS = SUCCESS_EVENT_KEYS | {'error'}
FROZEN_HARNESS_KEYS = {
    'schema', 'candidate_identity_sha256', 'version_manifest_sha256',
    'tool_manifest_sha256', 'candidate_freeze_sha256', 'browser_use_freeze_sha256',
    'smoke_preflight_sha256', 'smoke_manifest_sha256', 'executed_file_sha256',
}
SMOKE_HASH_KEYS = {'navigation', 'screenshot'}
METRICS = ('navigation_ns', 'elapsed_ns', 'post_navigation_ns')


class VerificationError(ValueError):
    pass


class InvalidPathError(ValueError):
    pass


class UnverifiedAttempt(Exception):
    pass


def _exact_keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VerificationError(f'{where} must be a JSON object with string keys')
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise VerificationError(f'{where} has invalid keys: missing={missing}, unknown={unknown}')
    return value


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise VerificationError(f'{where} must be a lowercase SHA-256 digest')
    return value


def _exact_int(value: Any, where: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise VerificationError(f'{where} must be an exact integer greater than or equal to {minimum}')
    return value


def _choice(value: Any, choices: set[str], where: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise VerificationError(f'{where} has an unsupported value')
    return value


def _finite_number(value: Any, where: str, *, minimum: float = 0.0) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise VerificationError(f'{where} must be a finite number greater than or equal to {minimum}')
    return value


def _string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise VerificationError(f'{where} must be a string list')
    return value


def _timestamp(value: Any, where: str) -> str:
    if not isinstance(value, str) or not UTC_RE.fullmatch(value):
        raise VerificationError(f'{where} must be a UTC timestamp ending in Z')
    try:
        datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except ValueError as error:
        raise VerificationError(f'{where} is not a valid UTC timestamp') from error
    return value


def _strict_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f'duplicate JSON object key: {key}')
        result[key] = value
    return result


def _reject_json_constant(value):
    raise VerificationError(f'non-finite JSON constant is forbidden: {value}')


def _read_private_bytes(path: Path, where: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f'{where} must be a regular file, not a symlink')
    status = path.stat()
    if stat.S_IMODE(status.st_mode) != 0o600 or status.st_uid != os.getuid():
        raise VerificationError(f'{where} must be owned by this user with mode 0600')
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VerificationError(f'{where} is unavailable') from error
    return raw


def _decode_json(raw: bytes, where: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode('utf-8'),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f'{where} is not strict UTF-8 JSON') from error
    if not isinstance(value, dict):
        raise VerificationError(f'{where} must contain a JSON object')
    return value


def _read_json(path: Path, where: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_private_bytes(path, where)
    return _decode_json(raw, where), raw


def _resolve_existing_directory(value: str | os.PathLike[str], where: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or '..' in raw.parts:
        raise InvalidPathError(f'{where} must be an absolute path without parent traversal')
    try:
        resolved = preflight._canonical_existing(raw, where, directory=True)
    except preflight.PreflightError as error:
        raise InvalidPathError(str(error)) from error
    return resolved


def _validate_private_directory(path: Path, where: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise VerificationError(f'{where} must be a real directory')
    status = path.stat()
    if stat.S_IMODE(status.st_mode) != 0o700 or status.st_uid != os.getuid():
        raise VerificationError(f'{where} must be owned by this user with mode 0700')


def _resolve_existing_file(value: str | os.PathLike[str], expected: Path, root: Path, where: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or '..' in raw.parts:
        raise InvalidPathError(f'{where} must be an absolute path without parent traversal')
    try:
        resolved = preflight._artifact_file(raw, root, where)
    except preflight.PreflightError as error:
        raise InvalidPathError(str(error)) from error
    if resolved != expected:
        raise InvalidPathError(f'{where} must be {expected.name} directly below RESULT_ROOT')
    return resolved


def _run_paths(run_root_value: str | os.PathLike[str], expected_mode: str):
    run_root = _resolve_existing_directory(run_root_value, 'run root')
    if expected_mode == 'smoke':
        if run_root.name != 'navigation' or run_root.parent.name != 'smoke':
            raise InvalidPathError('smoke run root must be RESULT_ROOT/smoke/navigation')
        result_root = run_root.parent.parent
        phase = 'navigation-smoke'
    elif run_root.name in ('navigation', 'navigation-final'):
        result_root = run_root.parent
        phase = 'navigation-run' if run_root.name == 'navigation' else 'navigation-final'
    else:
        raise InvalidPathError('full run root must be RESULT_ROOT/navigation or navigation-final')
    try:
        result_root = preflight._private_result_root(result_root)
    except preflight.PreflightError as error:
        raise InvalidPathError(str(error)) from error
    if run_root.parent != result_root and expected_mode != 'smoke':
        raise InvalidPathError('full run root must be a direct child of RESULT_ROOT')
    if expected_mode == 'smoke' and run_root.parent.parent != result_root:
        raise InvalidPathError('smoke run root must use RESULT_ROOT/smoke/navigation')
    _validate_private_directory(run_root, 'run root')
    if expected_mode == 'smoke':
        smoke_root = run_root.parent
        _validate_private_directory(smoke_root, 'smoke directory')
    return run_root, result_root, phase


def _validate_host_sample(value: Any, where: str) -> dict[str, Any]:
    sample = _exact_keys(value, HOST_SAMPLE_KEYS, where)
    _timestamp(sample['captured_at_utc'], f'{where}.captured_at_utc')
    load = _finite_number(sample['load_1m'], f'{where}.load_1m')
    logical_cpus = _exact_int(sample['logical_cpu_count'], f'{where}.logical_cpu_count', minimum=1)
    errors = _string_list(sample['classification_errors'], f'{where}.classification_errors')
    processes = sample['processes']
    if not isinstance(processes, list):
        raise VerificationError(f'{where}.processes must be a list')
    for index, raw_process in enumerate(processes):
        process = _exact_keys(raw_process, HOST_PROCESS_KEYS, f'{where}.processes[{index}]')
        _exact_int(process['pid'], f'{where}.processes[{index}].pid', minimum=1)
        if not isinstance(process['name'], str):
            raise VerificationError(f'{where}.processes[{index}].name must be a string')
        _finite_number(process['cpu_percent'], f'{where}.processes[{index}].cpu_percent')
        if type(process['excluded']) is not bool:
            raise VerificationError(f'{where}.processes[{index}].excluded must be a Boolean')
    return {'load': load, 'logical_cpus': logical_cpus, 'errors': errors, 'processes': processes}


def _classify_host_sample(sample: dict[str, Any], where: str) -> tuple[str, list[str]]:
    parsed = _validate_host_sample(sample, where)
    if parsed['errors']:
        return 'unknown', list(parsed['errors'])
    if parsed['load'] >= parsed['logical_cpus']:
        return 'contaminated', ['one-minute load reached logical CPU count']
    for process in parsed['processes']:
        if process['excluded'] is not True and process['cpu_percent'] >= 100:
            return 'contaminated', [f"unrelated process {process['pid']} used at least one CPU"]
    return 'clear', []


def _validate_host_gate(value: Any, samples: Any, where: str) -> dict[str, Any]:
    gate = _exact_keys(value, HOST_GATE_KEYS, where)
    _choice(gate['state'], HOST_STATES, f'{where}.state')
    reasons = _string_list(gate['reasons'], f'{where}.reasons')
    if not isinstance(samples, list):
        raise VerificationError(f'{where} host samples must be a list')
    checks = [_classify_host_sample(sample, f'{where}.samples[{index}]')
              for index, sample in enumerate(samples)]
    if not checks:
        expected_state, expected_reasons = 'unknown', ['host monitor collected no samples']
    else:
        expected_reasons = list(dict.fromkeys(reason for _state, values in checks for reason in values))
        if any(state == 'unknown' for state, _values in checks):
            expected_state = 'unknown'
        elif any(state == 'contaminated' for state, _values in checks):
            expected_state = 'contaminated'
        else:
            expected_state = 'clear'
    if gate['state'] != expected_state or reasons != expected_reasons:
        raise VerificationError(f'{where} does not match its host samples')
    return gate


def _validate_host_preflight(value: Any, where: str) -> dict[str, Any]:
    record = _exact_keys(value, {'state', 'reasons', 'logical_cpu_count', 'uptime', 'samples'}, where)
    _choice(record['state'], HOST_STATES, f'{where}.state')
    _string_list(record['reasons'], f'{where}.reasons')
    if record['logical_cpu_count'] is not None:
        _exact_int(record['logical_cpu_count'], f'{where}.logical_cpu_count', minimum=1)
    if record['uptime'] is not None and not isinstance(record['uptime'], str):
        raise VerificationError(f'{where}.uptime must be a string or null')
    if not isinstance(record['samples'], list):
        raise VerificationError(f'{where}.samples must be a list')
    if record['samples']:
        derived = _validate_host_gate(
            {'state': record['state'], 'reasons': record['reasons']},
            record['samples'], where,
        )
        first = record['samples'][0]
        if record['logical_cpu_count'] != first['logical_cpu_count']:
            raise VerificationError(f'{where}.logical_cpu_count differs from its samples')
        if any(sample['logical_cpu_count'] != record['logical_cpu_count'] for sample in record['samples']):
            raise VerificationError(f'{where}.samples use inconsistent logical CPU counts')
        return {**record, 'derived': derived}
    if record['state'] == 'clear':
        raise VerificationError(f'{where} cannot be clear without samples')
    return record


def _validate_frozen_harness(value: Any, *, candidate_sha: str, version_sha: str,
                             executed_files: dict[str, str]) -> dict[str, Any]:
    harness = _exact_keys(value, FROZEN_HARNESS_KEYS, 'frozen harness')
    if harness['schema'] != 'openbrowser-frozen-harness/1':
        raise VerificationError('frozen harness schema is unsupported')
    if _sha256(harness['candidate_identity_sha256'], 'frozen harness.candidate_identity_sha256') != candidate_sha:
        raise VerificationError('frozen harness does not match the candidate identity')
    if _sha256(harness['version_manifest_sha256'], 'frozen harness.version_manifest_sha256') != version_sha:
        raise VerificationError('frozen harness does not match the version manifest')
    for name in ('tool_manifest_sha256', 'candidate_freeze_sha256', 'browser_use_freeze_sha256'):
        _sha256(harness[name], f'frozen harness.{name}')
    for name in ('smoke_preflight_sha256', 'smoke_manifest_sha256'):
        records = _exact_keys(harness[name], SMOKE_HASH_KEYS, f'frozen harness.{name}')
        for phase, digest in records.items():
            _sha256(digest, f'frozen harness.{name}.{phase}')
    files = _exact_keys(harness['executed_file_sha256'], set(preflight.EXECUTED_FILE_PATHS),
                        'frozen harness.executed_file_sha256')
    if files != executed_files:
        raise VerificationError('frozen harness executed-file hashes differ from candidate identity')
    return harness


def _preflight_versions(checks: list[dict[str, Any]]) -> dict[str, str]:
    by_name = {check['name']: check for check in checks}
    if len(by_name) != len(checks):
        raise VerificationError('navigation preflight contains duplicate check names')
    required_checks = {'latest_versions', 'installed_tool_versions', 'installed_package_versions'}
    if not required_checks.issubset(by_name):
        raise VerificationError('navigation preflight is missing sealed version checks')
    for name in required_checks:
        check = by_name[name]
        if check['state'] != 'passed' or check['expected'] != check['actual']:
            raise VerificationError(f'navigation preflight version check did not match: {name}')
    latest = _exact_keys(
        by_name['latest_versions']['actual'],
        {'lightpanda', 'agent-browser', 'browser-use', 'browser-harness', 'chrome_for_testing'},
        'navigation preflight latest_versions',
    )
    installed_tools = _exact_keys(
        by_name['installed_tool_versions']['actual'],
        {'lightpanda', 'agent-browser', 'chrome_for_testing'},
        'navigation preflight installed_tool_versions',
    )
    installed_packages = _exact_keys(
        by_name['installed_package_versions']['actual'],
        {
            'lightpanda', 'agent-browser', 'chrome_for_testing', 'browser-use',
            'browser-harness', 'candidate:psutil', 'browser_use:psutil',
            'candidate:websockets', 'browser_use:websockets',
        },
        'navigation preflight installed_package_versions',
    )
    version_pattern = re.compile(r'^[0-9]+(?:\.[0-9]+){1,3}[A-Za-z0-9._+-]*$')
    for name, version in latest.items():
        if not isinstance(version, str) or not version_pattern.fullmatch(version):
            raise VerificationError(f'navigation preflight has an invalid version for {name}')
    for name, version in installed_tools.items():
        if version != latest[name]:
            raise VerificationError(f'navigation preflight installed {name} differs from latest metadata')
    for name, version in installed_packages.items():
        if not isinstance(version, str) or not version_pattern.fullmatch(version):
            raise VerificationError(f'navigation preflight has an invalid installed version for {name}')
    if installed_packages['browser-use'] != latest['browser-use'] \
            or installed_packages['browser-harness'] != latest['browser-harness']:
        raise VerificationError('navigation preflight installed browser packages differ from latest metadata')
    return {
        'openbrowser': OPENBROWSER_RUNTIME_VERSION,
        'raw-cdp': latest['chrome_for_testing'],
        'lightpanda': latest['lightpanda'],
        'agent-browser': latest['agent-browser'],
        'browser-use': latest['browser-use'],
        'browser-harness': latest['browser-harness'],
    }


def _validate_provenance(args, result_root: Path, phase: str,
                         manifest: dict[str, Any], mode: str) -> dict[str, Any]:
    identity_path = _resolve_existing_file(
        args.candidate_identity, result_root / 'candidate-identity.json', result_root,
        'candidate identity',
    )
    identity_value, identity_raw = _read_json(identity_path, 'candidate identity')
    try:
        identity = preflight._validate_identity(identity_value)
    except preflight.PreflightError as error:
        raise VerificationError(f'candidate identity is invalid: {error}') from error
    identity_sha = hashlib.sha256(identity_raw).hexdigest()

    preflight_path = _resolve_existing_file(
        args.preflight, result_root / f'preflight-{phase}.json', result_root,
        'navigation preflight',
    )
    preflight_value, preflight_raw = _read_json(preflight_path, 'navigation preflight')
    try:
        validated_preflight = preflight._validate_preflight_result(preflight_value)
    except preflight.PreflightError as error:
        raise VerificationError(f'navigation preflight is invalid: {error}') from error
    version_sha = _sha256(manifest['version_manifest_sha256'], 'run manifest.version_manifest_sha256')
    if validated_preflight['phase'] != phase or validated_preflight['outcome'] != 'passed' \
            or validated_preflight['errors'] != []:
        raise VerificationError(f'{phase} preflight did not pass')
    if any(check['state'] != 'passed' for check in validated_preflight['checks']):
        raise VerificationError(f'{phase} preflight contains a failed check')
    versions = _preflight_versions(validated_preflight['checks'])
    if validated_preflight['candidate_identity_sha256'] != identity_sha:
        raise VerificationError('navigation preflight does not match candidate identity')
    if validated_preflight['version_manifest_sha256'] != version_sha:
        raise VerificationError('navigation preflight does not match the version manifest hash')
    if manifest['candidate_identity_sha256'] != identity_sha:
        raise VerificationError('run manifest does not match the candidate identity hash')
    if manifest['preflight_sha256'] != hashlib.sha256(preflight_raw).hexdigest():
        raise VerificationError('run manifest does not match the navigation preflight hash')
    if manifest['source_commit'] != identity['candidate_commit'] \
            or manifest['source_tree'] != identity['candidate_tree']:
        raise VerificationError('run manifest source revision differs from candidate identity')
    if manifest['harness_sha256'] != identity['executed_files']['benchmarks/latest_navigation_baseline.py']:
        raise VerificationError('run manifest harness hash differs from candidate identity')

    frozen = None
    frozen_sha = None
    if mode == 'run':
        frozen_path = _resolve_existing_file(
            args.frozen_harness, result_root / 'frozen-harness.json', result_root,
            'frozen harness',
        )
        frozen_value, frozen_raw = _read_json(frozen_path, 'frozen harness')
        frozen = _validate_frozen_harness(
            frozen_value,
            candidate_sha=identity_sha,
            version_sha=version_sha,
            executed_files=identity['executed_files'],
        )
        frozen_sha = hashlib.sha256(frozen_raw).hexdigest()
        if manifest['frozen_harness_sha256'] != frozen_sha:
            raise VerificationError('run manifest does not match the frozen harness hash')
    elif manifest['frozen_harness_sha256'] is not None:
        raise VerificationError('smoke manifest frozen_harness_sha256 must be null')

    return {
        'identity': identity,
        'identity_sha256': identity_sha,
        'preflight': validated_preflight,
        'preflight_sha256': hashlib.sha256(preflight_raw).hexdigest(),
        'frozen_harness': frozen,
        'frozen_harness_sha256': frozen_sha,
        'version_manifest_sha256': version_sha,
        'versions': versions,
    }


def _verify_current_runner_files(identity: dict[str, Any]) -> None:
    source_root = Path(__file__).resolve().parents[1]
    try:
        current_files = preflight._hash_executed_files(source_root)
    except preflight.PreflightError as error:
        raise VerificationError(f'candidate runner or verifier files are unavailable: {error}') from error
    if current_files != identity['executed_files']:
        raise VerificationError('candidate runner or verifier files changed after candidate identity creation')


def _validate_url(url: Any, nonce: str, workload: str, where: str) -> None:
    if not isinstance(url, str):
        raise VerificationError(f'{where}.url must be a string')
    try:
        parsed = urlsplit(url)
        port = parsed.port
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as error:
        raise VerificationError(f'{where}.url is malformed') from error
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or type(port) is not int \
            or not 1 <= port <= 65535 or parsed.username is not None or parsed.password is not None \
            or parsed.path != '/fixture' or parsed.fragment or set(query) != {'nonce', 'delayed'} \
            or query['nonce'] != [nonce] or query['delayed'] != [str(int(workload == 'delayed'))]:
        raise VerificationError(f'{where}.url does not name the requested loopback fixture task')


def _validate_ledger_shape(value: Any, where: str) -> dict[str, Any]:
    ledger = _exact_keys(value, LEDGER_KEYS, where)
    _exact_int(ledger['effects'], f'{where}.effects')
    requests = ledger['requests']
    if not isinstance(requests, list):
        raise VerificationError(f'{where}.requests must be a list')
    for index, raw_event in enumerate(requests):
        event_where = f'{where}.requests[{index}]'
        if not isinstance(raw_event, dict):
            raise VerificationError(f'{event_where} must be an object')
        if set(raw_event) not in (SUCCESS_EVENT_KEYS, FAILURE_EVENT_KEYS):
            raise VerificationError(f'{event_where} has unknown or missing keys')
        event = raw_event
        if not isinstance(event['path'], str) or not event['path'].startswith('/'):
            raise VerificationError(f'{event_where}.path is invalid')
        if not isinstance(event['nonce'], str):
            raise VerificationError(f'{event_where}.nonce must be a string')
        if not isinstance(event['method'], str):
            raise VerificationError(f'{event_where}.method must be a string')
        _exact_int(event['received_ns'], f'{event_where}.received_ns')
        _exact_int(event['status'], f'{event_where}.status')
        _exact_int(event['bytes'], f'{event_where}.bytes')
        if 'completed_ns' in event:
            _exact_int(event['completed_ns'], f'{event_where}.completed_ns')
        if 'error' in event and not isinstance(event['error'], str):
            raise VerificationError(f'{event_where}.error must be a string')
    return ledger


def _verify_timing(sample: dict[str, Any], where: str, *, require_complete: bool) -> dict[str, int] | None:
    fields = ('start_ns', 'navigation_return_ns', 'end_ns', 'elapsed_ns', 'navigation_ns', 'post_navigation_ns')
    present = {name for name in fields if name in sample}
    if require_complete and present != set(fields):
        missing = sorted(set(fields) - present)
        raise VerificationError(f'{where} is missing timing fields: {missing}')
    if not require_complete and not present:
        return None
    for name in present:
        _exact_int(sample[name], f'{where}.{name}')
    if 'start_ns' not in sample or 'end_ns' not in sample or 'elapsed_ns' not in sample:
        if require_complete:
            raise VerificationError(f'{where} is missing timer bounds')
        return None
    start, end = sample['start_ns'], sample['end_ns']
    if end < start or sample['elapsed_ns'] != end - start:
        raise VerificationError(f'{where} has invalid elapsed_ns bounds')
    if 'navigation_return_ns' in sample:
        navigation_return = sample['navigation_return_ns']
        if not start <= navigation_return <= end:
            raise VerificationError(f'{where} navigation timestamps are out of interval')
        if 'navigation_ns' in sample and sample['navigation_ns'] != navigation_return - start:
            raise VerificationError(f'{where}.navigation_ns does not match its timestamps')
        if 'post_navigation_ns' in sample and sample['post_navigation_ns'] != end - navigation_return:
            raise VerificationError(f'{where}.post_navigation_ns does not match its timestamps')
    return {name: sample[name] for name in fields} if present == set(fields) else None


def _verify_successful_worker_sample(sample: Any, record: dict[str, Any], run_id: str,
                                     round_number: int, competitor: str, attempt_number: int,
                                     workload: str, arm: str, index: int, where: str) -> dict[str, int]:
    if not isinstance(sample, dict):
        raise VerificationError(f'{where}.sample must be an object for a successful trial')
    if set(sample) != SAMPLE_REQUEST_KEYS:
        raise VerificationError(f'{where}.sample has unknown or missing keys')
    if sample['op'] != 'trial' or sample['outcome'] != 'success':
        raise VerificationError(f'{where}.sample is not a successful navigation trial')
    if sample['arm'] != arm or sample['kind'] != workload or sample['index'] != index \
            or type(sample['index']) is not int:
        raise VerificationError(f'{where}.sample arm, workload, or index differs from schedule')
    nonce = record['nonce']
    nonce_prefix = f'{run_id}-r{round_number}-{competitor}-a{attempt_number}-{arm}-{workload}-{index}-'
    if not isinstance(nonce, str) or not nonce.startswith(nonce_prefix) \
            or not SAMPLE_NONCE_RE.fullmatch(nonce[len(nonce_prefix):]):
        raise VerificationError(f'{where}.nonce does not match its run, pair, attempt, and sample')
    if sample['nonce'] != nonce or sample['url'] != record['url']:
        raise VerificationError(f'{where}.sample request differs from its saved request')
    _validate_url(record['url'], nonce, workload, where)
    timings = _verify_timing(sample, f'{where}.sample', require_complete=True)
    if timings is None:
        raise VerificationError(f'{where}.sample has incomplete timing')
    start, navigation_return, end, elapsed, navigation_ns, post_navigation = (
        timings[name] for name in (
            'start_ns', 'navigation_return_ns', 'end_ns', 'elapsed_ns',
            'navigation_ns', 'post_navigation_ns',
        )
    )
    if type(sample['checked_ns']) is not int or not start <= sample['checked_ns'] <= end:
        raise VerificationError(f'{where}.sample.checked_ns is outside the measured interval')
    for field, value in (
        ('arm', arm), ('kind', workload), ('index', index), ('nonce', nonce),
        ('url', record['url']),
    ):
        if sample[field] != value:
            raise VerificationError(f'{where}.sample.{field} differs from its request')
    if sample['elapsed_ns'] != end - start or sample['navigation_ns'] != navigation_return - start \
            or sample['post_navigation_ns'] != end - navigation_return:
        raise VerificationError(f'{where}.sample timing values do not match timer bounds')

    navigation_observation = _exact_keys(
        sample['navigation_observation'], OBSERVATION_KEYS,
        f'{where}.sample.navigation_observation',
    )
    final_observation = _exact_keys(
        sample['final_observation'], OBSERVATION_KEYS,
        f'{where}.sample.final_observation',
    )
    expected_observation = {
        'url': record['url'],
        'title': f'fixture-{nonce}',
        'readyState': 'complete',
        'marker': nonce,
        'loaded': nonce,
        'scriptDone': nonce,
    }
    for name, expected in expected_observation.items():
        if navigation_observation[name] != expected or final_observation[name] != expected:
            raise VerificationError(f'{where} navigation observation failed {name}')
    if navigation_observation['field'] != '' or final_observation['field'] != nonce:
        raise VerificationError(f'{where} DOM field mutation proof is invalid')
    effect = _exact_keys(sample['effect_observation'], {'field', 'response'},
                         f'{where}.sample.effect_observation')
    response = _exact_keys(effect['response'], {'count', 'nonce', 'payload'},
                           f'{where}.sample.effect_observation.response')
    if effect['field'] != nonce or response != {'count': 1, 'nonce': nonce, 'payload': nonce}:
        raise VerificationError(f'{where} DOM side-effect response proof is invalid')

    ledger = _validate_ledger_shape(record['request_proof'], f'{where}.request_proof')
    worker_ledger = _validate_ledger_shape(sample['server'], f'{where}.sample.server')
    if ledger != worker_ledger:
        raise VerificationError(f'{where} independent fixture ledger differs from worker audit')
    if type(ledger['effects']) is not int or ledger['effects'] != 1:
        raise VerificationError(f'{where} fixture ledger must prove exactly one side effect')
    needed = {'/fixture': 'GET', '/effect': 'POST'}
    if workload == 'delayed':
        needed['/slow.js'] = 'GET'
    requests = ledger['requests']
    if len(requests) != len(needed):
        raise VerificationError(f'{where} fixture request ledger has unexpected work')
    seen = {}
    for event in requests:
        if set(event) != SUCCESS_EVENT_KEYS:
            raise VerificationError(f'{where} fixture request is not a successful response record')
        key = (event['path'], event['method'])
        seen[key] = seen.get(key, 0) + 1
        if event['nonce'] != nonce or event['status'] != 200 or 'completed_ns' not in event:
            raise VerificationError(f'{where} fixture event is not a successful nonce-bound response')
        received, completed = event['received_ns'], event['completed_ns']
        if not start <= received <= completed <= end:
            raise VerificationError(f'{where} fixture response is outside the timer')
    if set(seen) != {(path, method) for path, method in needed.items()} or any(count != 1 for count in seen.values()):
        raise VerificationError(f'{where} fixture ledger does not prove required request counts')
    if elapsed != end - start or navigation_ns + post_navigation != elapsed:
        raise VerificationError(f'{where} measured timing phases are inconsistent')
    return timings


def _verify_trial_record(value: Any, *, run_id: str, round_number: int, competitor: str,
                         attempt_number: int, arm: str, workload: str, index: int,
                         warmup: bool, nonce_set: set[str], where: str) -> dict[str, Any]:
    record = _exact_keys(value, SAMPLE_RECORD_KEYS, where)
    if record['arm'] != arm or record['workload'] != workload or record['index'] != index:
        raise VerificationError(f'{where} arm, workload, or index differs from schedule')
    if type(record['index']) is not int:
        raise VerificationError(f'{where}.index must be an exact integer')
    if type(record['warmup']) is not bool or record['warmup'] is not warmup:
        raise VerificationError(f'{where}.warmup does not match its collection')
    _choice(record['state'], {'verified', 'failed'}, f'{where}.state')
    errors = _string_list(record['errors'], f'{where}.errors')
    nonce = record['nonce']
    nonce_prefix = f'{run_id}-r{round_number}-{competitor}-a{attempt_number}-{arm}-{workload}-{index}-'
    if not isinstance(nonce, str) or not nonce.startswith(nonce_prefix) \
            or not SAMPLE_NONCE_RE.fullmatch(nonce[len(nonce_prefix):]):
        raise VerificationError(f'{where}.nonce does not match its run, pair, attempt, and sample')
    if nonce in nonce_set:
        raise VerificationError(f'{where}.nonce is not unique within the run')
    nonce_set.add(nonce)
    _validate_url(record['url'], nonce, workload, where)
    ledger = _validate_ledger_shape(record['request_proof'], f'{where}.request_proof')
    worker_sample = record['sample']
    successful_response = isinstance(worker_sample, dict) and worker_sample.get('outcome') == 'success'
    timing = None
    if successful_response:
        timing = _verify_successful_worker_sample(
            worker_sample, record, run_id, round_number, competitor, attempt_number,
            workload, arm, index, where,
        )
    elif worker_sample is None:
        if record['state'] != 'failed':
            raise VerificationError(f'{where} has no worker response but is not failed')
    elif isinstance(worker_sample, dict):
        if worker_sample.get('outcome') != 'failure':
            raise VerificationError(f'{where}.sample outcome is unsupported')
        for name, expected in (
            ('arm', arm), ('kind', workload), ('index', index), ('nonce', nonce), ('url', record['url']),
        ):
            if worker_sample.get(name) != expected:
                raise VerificationError(f'{where}.sample.{name} differs from its request')
        if 'server' in worker_sample:
            _validate_ledger_shape(worker_sample['server'], f'{where}.sample.server')
        _verify_timing(worker_sample, f'{where}.sample', require_complete=False)
    else:
        raise VerificationError(f'{where}.sample must be an object or null')
    if record['state'] == 'verified':
        if timing is None or errors:
            raise VerificationError(f'{where} is marked verified without successful proof and timing')
    elif not errors:
        raise VerificationError(f'{where} is marked failed without an error record')
    return {'record': record, 'timing': timing, 'ledger': ledger}


def _validate_process_identity(value: Any, where: str, *, state_required: bool,
                               owner_token: str | None = None, owner_root: str | None = None) -> dict[str, Any]:
    expected = PROCESS_IDENTITY_KEYS if state_required else PROCESS_IDENTITY_WITHOUT_STATE_KEYS
    identity = _exact_keys(value, expected, where)
    _exact_int(identity['pid'], f'{where}.pid', minimum=1)
    _finite_number(identity['created'], f'{where}.created')
    if not isinstance(identity['cmdline'], list) or not identity['cmdline'] \
            or any(not isinstance(argument, str) for argument in identity['cmdline']):
        raise VerificationError(f'{where}.cmdline must be a non-empty string list')
    if not isinstance(identity['owner_token'], str) or not OWNER_TOKEN_RE.fullmatch(identity['owner_token']):
        raise VerificationError(f'{where}.owner_token is invalid')
    if owner_token is not None and identity['owner_token'] != owner_token:
        raise VerificationError(f'{where}.owner_token differs within the attempt')
    if not isinstance(identity['owner_root'], str) or not Path(identity['owner_root']).is_absolute() \
            or '..' in Path(identity['owner_root']).parts:
        raise VerificationError(f'{where}.owner_root must be an absolute private arm path')
    if owner_root is not None and identity['owner_root'] != owner_root:
        raise VerificationError(f'{where}.owner_root differs from the expected arm root')
    if state_required:
        _choice(identity['state'], {'absent', 'reused', 'alive', 'zombie', 'unknown'}, f'{where}.state')
    return identity


def _validate_worker_cleanup(value: Any, arm: str, runtime_root: Path, where: str,
                             owner_tokens: set[str], *, require_ok: bool) -> bool:
    worker = _exact_keys(value, set(value) if isinstance(value, dict) else set(), where)
    if not set(worker).issubset(WORKER_CLEANUP_KEYS) or not {'ok', 'fallback_used', 'worker_identity',
                                                             'post_worker_exit_identities'}.issubset(worker):
        raise VerificationError(f'{where} has unknown or missing cleanup keys')
    if 'error' in worker and not isinstance(worker['error'], str):
        raise VerificationError(f'{where}.error must be a string')
    if type(worker["ok"]) is not bool:
        raise VerificationError(f"{where} cleanup flags must be Booleans")
    if type(worker["fallback_used"]) is not bool and not (
        worker["fallback_used"] is None
        and worker["ok"] is False
        and worker.get("error", "").strip()
    ):
        raise VerificationError(
            f"{where}.fallback_used may be unknown only for failed cleanup "
            "with a non-empty coordinator error; otherwise it must be a Boolean"
        )
    worker_identity = _validate_process_identity(
        worker['worker_identity'], f'{where}.worker_identity', state_required=True,
        owner_root=str(runtime_root / arm),
    )
    token = worker_identity['owner_token']
    owner_tokens.add(token)
    post = worker['post_worker_exit_identities']
    if not isinstance(post, list):
        raise VerificationError(f'{where}.post_worker_exit_identities must be a list')
    post_states = []
    post_identity_keys = set()
    for index, item in enumerate(post):
        identity = _validate_process_identity(
            item,
            f"{where}.post_worker_exit_identities[{index}]",
            state_required=True,
            owner_token=token,
            owner_root=str(runtime_root / arm),
        )
        post_states.append(identity["state"])
        post_identity_keys.add(
            (
                identity["pid"],
                identity["created"],
                identity["owner_token"],
                tuple(identity["cmdline"]),
            )
        )
    result_identity_keys = set()
    result_errors = []
    if "result" in worker:
        result = _exact_keys(
            worker["result"], WORKER_STOP_RESULT_KEYS, f"{where}.result"
        )
        if not isinstance(result["identities"], list):
            raise VerificationError(f"{where}.result.identities must be a list")
        result_errors = _string_list(result["errors"], f"{where}.result.errors")
        if worker["ok"] and result_errors:
            raise VerificationError(
                f"{where} successful cleanup has worker stop errors"
            )
        for index, item in enumerate(result["identities"]):
            identity = _validate_process_identity(
                item,
                f"{where}.result.identities[{index}]",
                state_required=True,
                owner_token=token,
                owner_root=str(runtime_root / arm),
            )
            result_identity_keys.add(
                (
                    identity["pid"],
                    identity["created"],
                    identity["owner_token"],
                    tuple(identity["cmdline"]),
                )
            )
    elif worker["ok"]:
        raise VerificationError(
            f"{where} successful cleanup lacks a worker stop result"
        )
    retired = (
        worker_identity["state"] in {"absent", "reused"}
        and all(state in {"absent", "reused"} for state in post_states)
        and result_identity_keys.issubset(post_identity_keys)
    )
    cleanup_ok = retired and not result_errors and not worker.get("error")
    if worker["ok"] is not cleanup_ok:
        raise VerificationError(f"{where}.ok does not match worker retirement proof")
    if require_ok and worker["ok"] is not True:
        raise VerificationError(f"{where} cleanup is not verified")
    return worker["ok"]


def _validate_attempt_cleanup(value: Any, expected_arms: tuple[str, ...], where: str,
                              *, terminal_state: str) -> dict[str, Any]:
    cleanup = _exact_keys(value, ATTEMPT_CLEANUP_KEYS, where)
    if type(cleanup['ok']) is not bool or type(cleanup['workers_ok']) is not bool \
            or type(cleanup['runtime_root_removed']) is not bool:
        raise VerificationError(f'{where} cleanup flags must be Booleans')
    if not isinstance(cleanup['errors'], list) or any(not isinstance(error, str) for error in cleanup['errors']):
        raise VerificationError(f'{where}.errors must be a string list')
    raw_root = cleanup['runtime_root']
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute() or '..' in Path(raw_root).parts:
        raise VerificationError(f'{where}.runtime_root must be an absolute private path')
    runtime_root = Path(raw_root)
    physical_tmp = Path('/tmp').resolve(strict=True)
    if runtime_root.parent != physical_tmp or not runtime_root.name.startswith('nb-navigation-'):
        raise VerificationError(f'{where}.runtime_root is outside the documented private temp root')
    if cleanup['runtime_root_removed'] and runtime_root.exists():
        raise VerificationError(f'{where}.runtime_root_removed is contradicted by a live runtime root')
    workers = cleanup['workers']
    if not isinstance(workers, dict) or any(not isinstance(name, str) for name in workers):
        raise VerificationError(f'{where}.workers must be an arm-keyed object')
    if set(workers) - set(expected_arms):
        raise VerificationError(f'{where}.workers contains an unexpected arm')
    owner_tokens: set[str] = set()
    worker_results = {
        arm: _validate_worker_cleanup(
            worker, arm, runtime_root, f'{where}.workers.{arm}', owner_tokens,
            require_ok=terminal_state in {'verified', 'contaminated'},
        )
        for arm, worker in workers.items()
    }
    workers_ok = len(workers) == len(expected_arms) and all(worker_results.values())
    if cleanup['workers_ok'] is not workers_ok:
        raise VerificationError(f'{where}.workers_ok does not match worker cleanup records')
    if len(owner_tokens) > 1:
        raise VerificationError(f'{where} worker owner tokens differ within the attempt')
    expected_ok = workers_ok and cleanup['runtime_root_removed']
    if cleanup['ok'] is not expected_ok:
        raise VerificationError(f'{where}.ok does not match its cleanup proofs')
    if cleanup['ok'] and cleanup['errors']:
        raise VerificationError(f'{where} successful cleanup has errors')
    if terminal_state in {'verified', 'contaminated'} and not cleanup['ok']:
        raise VerificationError(f'{where} terminal attempt lacks verified cleanup')
    return cleanup


def _expected_execution(round_number: int, competitor: str, samples_per_workload: int):
    expected = []
    pair = ('openbrowser', competitor)
    for workload in WORKLOADS:
        for index in range(samples_per_workload):
            arms = pair if index % 2 == 0 else tuple(reversed(pair))
            expected.extend({'workload': workload, 'index': index, 'arm': arm} for arm in arms)
    return expected


def _validate_retry_gate(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VerificationError(f'{where} must be an object')
    keys = set(value)
    if keys == HOST_GATE_KEYS:
        gate = _exact_keys(value, HOST_GATE_KEYS, where)
        _choice(gate['state'], HOST_STATES, f'{where}.state')
        _string_list(gate['reasons'], f'{where}.reasons')
        return gate
    record = _validate_host_preflight(value, where)
    return {'state': record['state'], 'reasons': record['reasons']}


def _verify_attempt(run_root: Path, run_manifest: dict[str, Any], pair: dict[str, Any],
                    reference: dict[str, Any], nonce_set: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    where = f"round-{pair['round']}/{pair['competitor']}/attempt-{reference['attempt_number']}"
    attempt_number = _exact_int(reference['attempt_number'], f'{where}.attempt_number', minimum=1)
    if attempt_number not in (1, 2):
        raise VerificationError(f'{where} attempt number or state is unsupported')
    _choice(reference['state'], ATTEMPT_STATES, f'{where}.state')
    directory = reference['directory']
    expected_directory = f"round-{pair['round']}/pair-{pair['competitor']}/attempt-{attempt_number}"
    path = PurePosixPath(directory) if isinstance(directory, str) else None
    if path is None or path.is_absolute() or '..' in path.parts or path.as_posix() != expected_directory:
        raise VerificationError(f'{where} directory does not match its pair and attempt')
    attempt_dir = run_root.joinpath(*path.parts)
    for directory, name in (
        (run_root / path.parts[0], f'{where} round directory'),
        (run_root / path.parts[0] / path.parts[1], f'{where} pair directory'),
        (attempt_dir, f'{where} attempt directory'),
    ):
        if not directory.exists() or directory.is_symlink():
            raise UnverifiedAttempt(f'{where} has no verified attempt directory')
        try:
            _validate_private_directory(directory, name)
        except VerificationError as error:
            raise UnverifiedAttempt(f'{where} attempt directory is not privately verifiable') from error
    raw_manifest_path = attempt_dir / ATTEMPT_MANIFEST_NAME
    if raw_manifest_path.is_symlink() or not raw_manifest_path.is_file():
        raise UnverifiedAttempt(f'{where} attempt manifest is missing or unsafe')
    try:
        manifest_path = preflight._artifact_file(
            raw_manifest_path, run_root, f'{where} manifest',
        )
        raw = _read_private_bytes(manifest_path, f'{where} manifest')
    except (preflight.PreflightError, VerificationError) as error:
        raise UnverifiedAttempt(f'{where} attempt manifest is not privately verifiable') from error
    if _sha256(reference['manifest_sha256'], f'{where}.manifest_sha256') != hashlib.sha256(raw).hexdigest():
        raise UnverifiedAttempt(f'{where} attempt manifest hash does not match its reference')
    attempt = _decode_json(raw, f'{where} manifest')
    attempt = _exact_keys(attempt, ATTEMPT_KEYS, f'{where} manifest')
    if attempt['schema'] != ATTEMPT_SCHEMA:
        raise VerificationError(f'{where} manifest schema is unsupported')
    if attempt['run_id'] != run_manifest['run_id'] or attempt['round'] != pair['round'] \
            or attempt['competitor'] != pair['competitor'] or attempt['attempt_number'] != attempt_number:
        raise VerificationError(f'{where} manifest identity differs from its pair reference')
    if attempt['state'] != reference['state']:
        raise VerificationError(f'{where} manifest state differs from its pair reference')
    _choice(attempt['state'], ATTEMPT_STATES, f'{where} manifest state')
    expected_arms = ('openbrowser', pair['competitor'])

    host_samples = attempt['host_samples']
    if not isinstance(host_samples, list):
        raise VerificationError(f'{where}.host_samples must be a list')
    if attempt['state'] in {'pending', 'running'} and not host_samples:
        host_gate = _exact_keys(attempt['host_gate'], HOST_GATE_KEYS, f'{where}.host_gate')
        if host_gate['state'] != 'unknown':
            raise VerificationError(f'{where} has no host samples but does not have an unknown host gate')
        _string_list(host_gate['reasons'], f'{where}.host_gate.reasons')
    else:
        host_gate = _validate_host_gate(attempt['host_gate'], host_samples, f'{where}.host_gate')
    if host_gate != reference['host_gate']:
        raise VerificationError(f'{where} host gate differs from its pair reference')
    host_reference = _exact_keys(reference['host_gate'], HOST_GATE_KEYS, f'{where} reference host_gate')
    if host_reference != host_gate:
        raise VerificationError(f'{where} pair reference contains a different host gate')

    if not isinstance(attempt['execution_order'], list) or not isinstance(attempt['samples'], list):
        raise VerificationError(f'{where} execution order and samples must be lists')
    expected_execution = _expected_execution(
        pair['round'], pair['competitor'], run_manifest['samples_per_workload'],
    )
    order = []
    for index, item in enumerate(attempt['execution_order']):
        entry = _exact_keys(item, EXECUTION_ORDER_KEYS, f'{where}.execution_order[{index}]')
        if entry['workload'] not in WORKLOADS or entry['arm'] not in expected_arms:
            raise VerificationError(f'{where}.execution_order[{index}] uses an unsupported workload or arm')
        _exact_int(entry['index'], f'{where}.execution_order[{index}].index')
        order.append(entry)
    if len(order) > len(expected_execution) or order != expected_execution[:len(order)]:
        raise VerificationError(f'{where}.execution_order does not follow the counterbalanced schedule')
    if len(attempt['samples']) != len(order):
        if attempt['state'] != 'running' or len(order) != len(attempt['samples']) + 1:
            raise VerificationError(f'{where}.samples do not match its execution order')
    sample_results = []
    for index, (entry, raw_record) in enumerate(zip(order, attempt['samples'])):
        sample_results.append(_verify_trial_record(
            raw_record,
            run_id=run_manifest['run_id'], round_number=pair['round'],
            competitor=pair['competitor'], attempt_number=attempt_number,
            arm=entry['arm'], workload=entry['workload'], index=entry['index'],
            warmup=False, nonce_set=nonce_set,
            where=f'{where}.samples[{index}]',
        ))

    warmups = attempt['warmups']
    if not isinstance(warmups, list) or len(warmups) > len(expected_arms):
        raise VerificationError(f'{where}.warmups must be an arm-ordered prefix')
    warmup_results = []
    for index, (arm, raw_record) in enumerate(zip(expected_arms, warmups)):
        warmup_results.append(_verify_trial_record(
            raw_record,
            run_id=run_manifest['run_id'], round_number=pair['round'],
            competitor=pair['competitor'], attempt_number=attempt_number,
            arm=arm, workload='fast', index=0, warmup=True, nonce_set=nonce_set,
            where=f'{where}.warmups[{index}]',
        ))
    attempt_errors = _string_list(attempt['errors'], f'{where}.errors')
    cleanup = _validate_attempt_cleanup(
        attempt['cleanup'], expected_arms, f'{where}.cleanup', terminal_state=attempt['state'],
    )
    counts = _exact_keys(attempt['work_counts'], WORK_COUNT_KEYS, f'{where}.work_counts')
    for name in WORK_COUNT_KEYS:
        _exact_int(counts[name], f'{where}.work_counts.{name}')
    in_flight_count = len(order) - len(sample_results)
    attempted = len(sample_results) + in_flight_count
    successful = sum(result['record']['state'] == 'verified' for result in sample_results)
    nested_errors = [
        error
        for result in sample_results + warmup_results
        for error in result['record']['errors']
    ]
    unbound_errors = list(attempt_errors)
    for nested_error in nested_errors:
        match = next(
            (index for index, error in enumerate(unbound_errors)
             if error.endswith(f': {nested_error}')),
            None,
        )
        if match is None:
            raise VerificationError(f'{where} attempt errors do not preserve a sample or warmup error')
        del unbound_errors[match]
    error_count = len(attempt_errors)
    expected_counts = {
        'attempted_work_count': attempted,
        'successful_work_count': successful,
        'error_count': error_count,
    }
    if counts != expected_counts:
        raise VerificationError(f'{where}.work_counts do not match its samples and errors')
    if reference['work_counts'] != counts or reference['error_count'] != error_count \
            or reference['errors'] != attempt_errors:
        raise VerificationError(f'{where} work counts or errors differ from the pair reference')
    if type(reference['cleanup_ok']) is not bool or reference['cleanup_ok'] is not cleanup['ok']:
        raise VerificationError(f'{where}.cleanup_ok differs from the attempt cleanup proof')
    if type(reference['runtime_root_removed']) is not bool \
            or reference['runtime_root_removed'] is not cleanup['runtime_root_removed']:
        raise VerificationError(f'{where}.runtime_root_removed differs from the attempt cleanup proof')
    if attempt['state'] in {'verified', 'contaminated'}:
        if len(sample_results) != len(expected_execution) or len(warmup_results) != len(expected_arms):
            raise VerificationError(f'{where} terminal successful attempt is missing work or warmups')
        if any(result['record']['state'] != 'verified' for result in sample_results + warmup_results):
            raise VerificationError(f'{where} terminal successful attempt contains an unverified trial')
        if attempt_errors or error_count:
            raise VerificationError(f'{where} terminal successful attempt contains errors')
    if attempt['state'] == 'pending' and (
            order or sample_results or warmup_results or host_samples or attempt_errors
            or counts != {'attempted_work_count': 0, 'successful_work_count': 0, 'error_count': 0}
            or cleanup['workers']):
        raise VerificationError(f'{where} pending attempt contains work, host samples, errors, or workers')
    if attempt['state'] == 'verified' and host_gate['state'] != 'clear':
        raise VerificationError(f'{where} verified attempt has a non-clear host gate')
    if attempt['state'] == 'contaminated' and host_gate['state'] == 'clear':
        raise VerificationError(f'{where} contaminated attempt has a clear host gate')
    if attempt['state'] == 'failed' and not (error_count or not cleanup['ok']):
        raise VerificationError(f'{where} failed attempt has no recorded failure')
    return attempt, {
        'attempted_work_count': attempted,
        'successful_work_count': successful,
        'error_count': error_count,
        'host_contaminated_attempt_count': int(host_gate['state'] != 'clear'),
        'sample_results': sample_results,
        'cleanup': cleanup,
        'host_gate': host_gate,
    }


def _validate_unwritten_reference(pair: dict[str, Any], reference: dict[str, Any],
                                  expected_number: int) -> dict[str, Any]:
    where = f"round-{pair['round']}/{pair['competitor']}/attempt-{expected_number}"
    if type(reference['attempt_number']) is not int or reference['attempt_number'] != expected_number \
            or not isinstance(reference['state'], str) or reference['state'] not in {'pending', 'failed'} \
            or reference['manifest_sha256'] is not None:
        raise VerificationError(f'{where} may omit its manifest hash only while pending or failed')
    expected_directory = f"round-{pair['round']}/pair-{pair['competitor']}/attempt-{expected_number}"
    directory = reference['directory']
    path = PurePosixPath(directory) if isinstance(directory, str) else None
    if path is None or path.is_absolute() or '..' in path.parts or path.as_posix() != expected_directory:
        raise VerificationError(f'{where} directory does not match its pair and attempt')
    if type(reference['cleanup_ok']) is not bool or reference['cleanup_ok'] \
            or type(reference['runtime_root_removed']) is not bool or reference['runtime_root_removed']:
        raise VerificationError(f'{where} unverified reference cannot claim cleanup')
    gate = _exact_keys(reference['host_gate'], HOST_GATE_KEYS, f'{where}.host_gate')
    if gate['state'] != 'unknown':
        raise VerificationError(f'{where} unverified reference requires an unknown host gate')
    _string_list(gate['reasons'], f'{where}.host_gate.reasons')
    counts = _exact_keys(reference['work_counts'], WORK_COUNT_KEYS, f'{where}.work_counts')
    for name in WORK_COUNT_KEYS:
        _exact_int(counts[name], f'{where}.work_counts.{name}')
    error_count = _exact_int(reference['error_count'], f'{where}.error_count')
    errors = _string_list(reference['errors'], f'{where}.errors')
    if counts['successful_work_count'] != 0:
        raise VerificationError(f'{where} unverified reference cannot claim successful work')
    if reference['state'] == 'pending':
        if any(counts[name] != 0 for name in WORK_COUNT_KEYS) or error_count or errors:
            raise VerificationError(f'{where} pending reference cannot claim work or errors')
    elif counts['attempted_work_count'] != 0 or error_count < 1 or not errors \
            or counts['error_count'] != error_count:
        raise VerificationError(f'{where} failed reference must preserve its pre-manifest error only')
    return {
        'attempted_work_count': counts['attempted_work_count'],
        'successful_work_count': counts['successful_work_count'],
        'error_count': error_count,
        'host_contaminated_attempt_count': 1,
        'unverified': True,
    }


def _validate_unverified_reference(pair: dict[str, Any], reference: dict[str, Any],
                                   expected_number: int) -> dict[str, Any]:
    where = f"round-{pair['round']}/{pair['competitor']}/attempt-{expected_number}"
    _choice(reference['state'], ATTEMPT_STATES, f'{where}.state')
    expected_directory = f"round-{pair['round']}/pair-{pair['competitor']}/attempt-{expected_number}"
    directory = reference['directory']
    path = PurePosixPath(directory) if isinstance(directory, str) else None
    if path is None or path.is_absolute() or '..' in path.parts or path.as_posix() != expected_directory:
        raise VerificationError(f'{where} directory does not match its pair and attempt')
    if type(reference['cleanup_ok']) is not bool or type(reference['runtime_root_removed']) is not bool:
        raise VerificationError(f'{where} cleanup flags must be Booleans')
    gate = _exact_keys(reference['host_gate'], HOST_GATE_KEYS, f'{where}.host_gate')
    _choice(gate['state'], HOST_STATES, f'{where}.host_gate.state')
    _string_list(gate['reasons'], f'{where}.host_gate.reasons')
    counts = _exact_keys(reference['work_counts'], WORK_COUNT_KEYS, f'{where}.work_counts')
    for name in WORK_COUNT_KEYS:
        _exact_int(counts[name], f'{where}.work_counts.{name}')
    if counts['successful_work_count'] > counts['attempted_work_count']:
        raise VerificationError(f'{where} successful work exceeds attempted work')
    error_count = _exact_int(reference['error_count'], f'{where}.error_count')
    errors = _string_list(reference['errors'], f'{where}.errors')
    if error_count != counts['error_count']:
        raise VerificationError(f'{where} reference error count differs from its work counts')
    return {
        'attempted_work_count': counts['attempted_work_count'],
        'successful_work_count': counts['successful_work_count'],
        'error_count': error_count,
        'host_contaminated_attempt_count': int(gate['state'] != 'clear'),
        'unverified': True,
    }


def _verify_pair(run_root: Path, run_manifest: dict[str, Any], value: Any,
                 nonce_set: set[str]) -> tuple[dict[str, Any], dict[str, int]]:
    pair = _exact_keys(value, PAIR_KEYS, 'pair record')
    round_number = _exact_int(pair['round'], 'pair.round', minimum=1)
    if round_number not in (1, 2):
        raise VerificationError('pair.round must be 1 or 2')
    competitor = pair['competitor']
    if competitor not in COMPETITORS:
        raise VerificationError('pair.competitor is unsupported')
    if pair['arms'] != ['openbrowser', competitor]:
        raise VerificationError('pair.arms must contain OpenBrowser followed by its competitor')
    _choice(pair['state'], PAIR_STATES, 'pair.state')
    pair_errors = _string_list(pair['errors'], 'pair.errors')
    retry_checks = pair['retry_host_checks']
    if not isinstance(retry_checks, list) or len(retry_checks) > 2:
        raise VerificationError('pair.retry_host_checks must contain at most two host checks')
    retry_gates = [_validate_retry_gate(item, f'pair.retry_host_checks[{index}]')
                   for index, item in enumerate(retry_checks)]
    attempts = pair['attempts']
    if not isinstance(attempts, list) or len(attempts) > 2:
        raise VerificationError('pair.attempts must contain at most two attempts')
    attempt_records = []
    declared_counts = {'attempted_work_count': 0, 'successful_work_count': 0, 'error_count': 0,
                       'host_contaminated_attempt_count': 0}
    validated_counts = dict.fromkeys(declared_counts, 0)
    unverified_attempt_numbers = []
    for index, reference in enumerate(attempts, start=1):
        reference = _exact_keys(reference, ATTEMPT_REFERENCE_KEYS, f'pair.attempts[{index - 1}]')
        if _exact_int(reference['attempt_number'], f'pair.attempts[{index - 1}].attempt_number', minimum=1) != index:
            raise VerificationError('pair attempt numbers must be ordered and consecutive')
        if reference['manifest_sha256'] is None:
            if not isinstance(reference['state'], str) or reference['state'] not in {'pending', 'failed'}:
                raise VerificationError('attempt manifest hash may be null only while the reference is pending or failed')
            evidence = _validate_unwritten_reference(pair, reference, index)
            attempt_records.append((None, evidence))
            unverified_attempt_numbers.append(index)
            for name in declared_counts:
                declared_counts[name] += evidence[name]
                validated_counts[name] += evidence[name]
        else:
            _sha256(reference['manifest_sha256'], f'pair.attempts[{index - 1}].manifest_sha256')
            try:
                attempt, evidence = _verify_attempt(run_root, run_manifest, pair, reference, nonce_set)
            except UnverifiedAttempt:
                evidence = _validate_unverified_reference(pair, reference, index)
                attempt_records.append((None, evidence))
                unverified_attempt_numbers.append(index)
                for name in declared_counts:
                    declared_counts[name] += evidence[name]
            else:
                if attempt['state'] in {'pending', 'running'}:
                    evidence['unverified'] = True
                    unverified_attempt_numbers.append(index)
                attempt_records.append((attempt, evidence))
                for name in declared_counts:
                    declared_counts[name] += evidence[name]
                    validated_counts[name] += evidence[name]
    terminal_run = run_manifest['state'] in {'verifying', 'complete', 'incomplete'}
    if len(attempts) == 2:
        if attempts[0]['state'] != 'contaminated' or not retry_gates \
                or retry_gates[0]['state'] != 'clear':
            raise VerificationError('pair retry requires a contaminated first attempt and a fresh clear host check')
        if pair['state'] == 'contaminated' and terminal_run and len(retry_gates) != 2:
            raise VerificationError('terminal contaminated pair is missing its post-pair host check')
        if pair['state'] in {'verified', 'failed', 'running'} and len(retry_gates) != 1:
            raise VerificationError('non-contaminated terminal pair has an unexpected post-pair host check')
    elif attempts and attempts[0]['state'] == 'contaminated':
        if pair['state'] == 'contaminated':
            if terminal_run and len(retry_gates) != 2:
                raise VerificationError('terminal contaminated pair is missing its post-pair host check')
            if not terminal_run and not retry_gates:
                raise VerificationError('contaminated pair is missing its retry host check')
            if retry_gates and retry_gates[0]['state'] == 'clear':
                raise VerificationError('single contaminated attempt should have retried after a clear host check')
        elif pair['state'] == 'running' and len(retry_gates) > 1:
            raise VerificationError('running pair has too many retry host checks')
        elif pair['state'] not in {'running', 'contaminated'}:
            if retry_gates:
                raise VerificationError('pair host check is present without a contaminated attempt')
    elif retry_gates:
        raise VerificationError('pair host check is present without a contaminated attempt')
    if len(retry_gates) == 2 and attempts:
        if attempts[0]['state'] == 'contaminated' and len(attempts) == 1 \
                and retry_gates[0]['state'] == 'clear':
            raise VerificationError('single contaminated attempt cannot have a successful pre-retry gate')
        if len(attempts) == 2 and retry_gates[0]['state'] != 'clear':
            raise VerificationError('second attempt requires a clear first host gate')
    if pair['state'] == 'verified':
        if not attempts or attempts[-1]['state'] != 'verified' or pair_errors:
            raise VerificationError('verified pair must end with one verified attempt and no pair errors')
    elif attempts:
        terminal_attempt_state = attempts[-1]['state']
        expected_pair_state = 'running' if terminal_attempt_state == 'pending' else terminal_attempt_state
        if pair['state'] != expected_pair_state:
            raise VerificationError('pair state differs from its final attempt state')
    elif pair['state'] not in {'pending', 'running'}:
        raise VerificationError('terminal pair has no attempt record')
    if pair['state'] == 'failed' and not pair_errors and attempt_records \
            and (attempts[-1]['errors'] == []):
        raise VerificationError('failed pair has no recorded errors')

    pair_unverified = bool(unverified_attempt_numbers)
    pair_verification_state = 'unverified' if pair_unverified else (
        'verified' if pair['state'] == 'verified' else 'incomplete'
    )
    summary_pair = {
        'round': round_number,
        'competitor': competitor,
        'state': 'unverified' if pair_unverified else pair['state'],
        'manifest_state': pair['state'],
        'verification_state': pair_verification_state,
        'unverified_attempts': unverified_attempt_numbers,
        'unverified_attempt_count': len(unverified_attempt_numbers),
        'attempt_count': len(attempts),
        'attempted_work_count': validated_counts['attempted_work_count'],
        'successful_work_count': validated_counts['successful_work_count'],
        'error_count': validated_counts['error_count'],
        'metrics': None,
    }
    if pair['state'] == 'verified' and not pair_unverified:
        verified_attempt = attempt_records[-1][1]
        summary_pair['metrics'] = _pair_metrics(verified_attempt['sample_results'], competitor)
    return summary_pair, validated_counts, declared_counts


def _pair_metrics(sample_results: list[dict[str, Any]], competitor: str) -> dict[str, Any]:
    result = {}
    for arm in ('openbrowser', competitor):
        result[arm] = {}
        for workload in WORKLOADS:
            matching = [
                sample['timing'] for sample in sample_results
                if sample['record']['arm'] == arm and sample['record']['workload'] == workload
            ]
            if not matching or any(value is None for value in matching):
                raise VerificationError(f'verified pair is missing {arm}/{workload} timing samples')
            cell = {'samples': len(matching), 'errors': 0}
            for metric in METRICS:
                values = [sample[metric] for sample in matching]
                cell[metric] = {
                    'p50': statistics.median(values),
                    'min': min(values),
                    'max': max(values),
                }
            result[arm][workload] = cell
    return result


def _verify_run_manifest(run_root: Path, manifest: dict[str, Any], expected_mode: str,
                         nonce_set: set[str]) -> dict[str, Any]:
    _exact_keys(manifest, RUN_KEYS, 'run manifest')
    if manifest['mode'] != expected_mode:
        raise VerificationError('run manifest mode differs from the requested verifier')
    expected_schema = SMOKE_SCHEMA if expected_mode == 'smoke' else RUN_SCHEMA
    if manifest['schema'] != expected_schema:
        raise VerificationError('run manifest schema does not match its mode')
    _choice(manifest['state'], RUN_STATES, 'run manifest.state')
    run_id = manifest['run_id']
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise VerificationError('run manifest run_id must be 32 lowercase hexadecimal characters')
    _timestamp(manifest['created_at_utc'], 'run manifest.created_at_utc')
    for name in ('candidate_identity_sha256', 'version_manifest_sha256', 'preflight_sha256', 'harness_sha256'):
        _sha256(manifest[name], f'run manifest.{name}')
    if expected_mode == 'run':
        _sha256(manifest['frozen_harness_sha256'], 'run manifest.frozen_harness_sha256')
    elif manifest['frozen_harness_sha256'] is not None:
        raise VerificationError('smoke run frozen_harness_sha256 must be null')
    if not isinstance(manifest['source_commit'], str) or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', manifest['source_commit']):
        raise VerificationError('run manifest source_commit is invalid')
    if not isinstance(manifest['source_tree'], str) or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', manifest['source_tree']):
        raise VerificationError('run manifest source_tree is invalid')

    arms = list(ARMS_BY_MODE[expected_mode])
    if manifest['arms'] != arms or manifest['workloads'] != list(WORKLOADS):
        raise VerificationError('run manifest arms or workloads differ from the fixed benchmark contract')
    rounds = 1 if expected_mode == 'smoke' else 2
    samples = 1 if expected_mode == 'smoke' else 6
    if type(manifest['rounds']) is not int or manifest['rounds'] != rounds \
            or type(manifest['samples_per_workload']) is not int or manifest['samples_per_workload'] != samples:
        raise VerificationError('run manifest rounds or sample count differ from the fixed benchmark contract')
    expected_order = {'1': ['raw-cdp']} if expected_mode == 'smoke' else {
        '1': list(COMPETITORS), '2': list(reversed(COMPETITORS)),
    }
    if manifest['pair_order_by_round'] != expected_order:
        raise VerificationError('run manifest pair order differs from the counterbalanced schedule')

    host_preflight = _validate_host_preflight(manifest['host_preflight'], 'run manifest.host_preflight')
    _string_list(manifest['errors'], 'run manifest.errors')
    if manifest['failed_phase'] is not None and not isinstance(manifest['failed_phase'], str):
        raise VerificationError('run manifest.failed_phase must be a string or null')
    fixture = _exact_keys(manifest['fixture'], FIXTURE_KEYS, 'run manifest.fixture')
    cleanup = _exact_keys(manifest['cleanup'], RUN_CLEANUP_KEYS, 'run manifest.cleanup')
    for name in FIXTURE_KEYS:
        if type(fixture[name]) is not bool:
            raise VerificationError(f'run manifest.fixture.{name} must be a Boolean')
    for name in RUN_CLEANUP_KEYS:
        if type(cleanup[name]) is not bool:
            raise VerificationError(f'run manifest.cleanup.{name} must be a Boolean')

    pairs = manifest['pairs']
    if not isinstance(pairs, list):
        raise VerificationError('run manifest.pairs must be a list')
    expected_pairs = [(1, competitor) for competitor in COMPETITORS] \
        + [(2, competitor) for competitor in reversed(COMPETITORS)]
    if expected_mode == 'smoke':
        expected_pairs = [(1, 'raw-cdp')]
    if len(pairs) > len(expected_pairs):
        raise VerificationError('run manifest contains more pair blocks than its schedule')
    pair_results = {}
    totals = {'attempted_work_count': 0, 'successful_work_count': 0, 'error_count': 0,
              'host_contaminated_attempt_count': 0}
    declared_totals = dict.fromkeys(totals, 0)
    unverified_attempt_count = 0
    actual_pairs = []
    cleanup_flags = []
    for raw_pair in pairs:
        pair_result, pair_counts, pair_declared_counts = _verify_pair(
            run_root, manifest, raw_pair, nonce_set,
        )
        key = (pair_result['round'], pair_result['competitor'])
        actual_pairs.append(key)
        pair_results[f'round-{key[0]}/{key[1]}'] = pair_result
        for name in totals:
            totals[name] += pair_counts[name]
            declared_totals[name] += pair_declared_counts[name]
        unverified_attempt_count += pair_result['unverified_attempt_count']
        for reference in raw_pair['attempts']:
            cleanup_flags.append((reference['cleanup_ok'], reference['runtime_root_removed']))
    if actual_pairs != expected_pairs[:len(actual_pairs)]:
        raise VerificationError('run manifest pair blocks do not follow the recorded pair order')
    counts = _exact_keys(manifest['work_counts'], RUN_WORK_COUNT_KEYS, 'run manifest.work_counts')
    for name in RUN_WORK_COUNT_KEYS:
        _exact_int(counts[name], f'run manifest.work_counts.{name}')
    if counts != declared_totals:
        raise VerificationError('run manifest work counts do not match its pair attempts')
    expected_workers_ok = all(workers_ok for workers_ok, _removed in cleanup_flags)
    expected_roots_removed = all(removed for _workers_ok, removed in cleanup_flags)
    if cleanup['fixture_stopped'] is not fixture['stopped']:
        raise VerificationError('run cleanup summary differs from its pair and fixture proofs')
    if manifest['state'] in {'complete', 'incomplete'}:
        if cleanup['workers_ok'] is not expected_workers_ok \
                or cleanup['runtime_roots_removed'] is not expected_roots_removed:
            raise VerificationError('terminal run cleanup summary differs from its pair proofs')
    elif (cleanup['workers_ok'] and not expected_workers_ok) \
            or (cleanup['runtime_roots_removed'] and not expected_roots_removed):
        raise VerificationError('in-progress run claims cleanup not proved by its attempt references')
    if cleanup['ok'] is not (expected_workers_ok and expected_roots_removed and fixture['stopped']):
        raise VerificationError('run cleanup ok flag does not match its component proofs')
    if fixture['stopped'] is not (fixture['socket_closed'] and fixture['thread_stopped']):
        raise VerificationError('fixture stopped flag differs from socket and thread cleanup')

    all_pairs_verified = actual_pairs == expected_pairs and all(
        pair_results[f'round-{round_number}/{competitor}']['verification_state'] == 'verified'
        for round_number, competitor in expected_pairs
    )
    run_shape_complete = manifest['state'] == 'complete' and actual_pairs == expected_pairs and all(
        pair_results[f'round-{round_number}/{competitor}']['manifest_state'] == 'verified'
        for round_number, competitor in expected_pairs
    ) and cleanup['ok'] is True and host_preflight['state'] == 'clear' \
        and manifest['errors'] == [] and fixture['started'] is True
    fully_complete = run_shape_complete and all_pairs_verified and unverified_attempt_count == 0 \
        and cleanup['ok'] is True and host_preflight['state'] == 'clear' \
        and manifest['errors'] == [] and fixture['started'] is True
    if manifest['state'] == 'complete' and not run_shape_complete:
        raise VerificationError('complete run manifest lacks all pair, host, fixture, or cleanup proofs')
    if manifest['state'] == 'complete' and len(manifest['host_preflight']['samples']) != 6:
        raise VerificationError('complete run preflight must contain six host-gate samples')

    if expected_mode == 'smoke':
        if not fully_complete or len(pairs) != 1:
            raise VerificationError('navigation smoke is incomplete or contains non-control work')
        smoke_pair = pair_results['round-1/raw-cdp']
        for arm in ('openbrowser', 'raw-cdp'):
            fast = smoke_pair['metrics'][arm]['fast']['navigation_ns']['p50']
            delayed = smoke_pair['metrics'][arm]['delayed']['navigation_ns']['p50']
            if delayed - fast < SMOKE_DELAY_MIN_NS:
                raise VerificationError(f'{arm} delayed control is less than 100000000 ns slower than fast')

    return {
        'complete': fully_complete,
        'all_pairs_verified': all_pairs_verified,
        'pair_results': pair_results,
        'work_counts': totals,
        'unverified_attempt_count': unverified_attempt_count,
        'host_preflight_state': host_preflight['state'],
        'cleanup_ok': cleanup['ok'],
    }


def _comparison_status(first: dict[str, Any], second: dict[str, Any], workload: str) -> str:
    openbrowser = first['metrics']['openbrowser'][workload]['navigation_ns']
    competitor_first = first['metrics'][first['competitor']][workload]['navigation_ns']
    openbrowser_second = second['metrics']['openbrowser'][workload]['navigation_ns']
    competitor_second = second['metrics'][second['competitor']][workload]['navigation_ns']

    def round_result(ob, competitor):
        if ob['p50'] < competitor['p50'] and ob['max'] < competitor['min']:
            return 'OPENBROWSER_FASTER'
        if competitor['p50'] < ob['p50'] and competitor['max'] < ob['min']:
            return 'COMPETITOR_FASTER'
        return 'INCONCLUSIVE'

    outcomes = (round_result(openbrowser, competitor_first),
                round_result(openbrowser_second, competitor_second))
    if outcomes[0] != outcomes[1] or 'INCONCLUSIVE' in outcomes:
        return 'INCONCLUSIVE'
    return outcomes[0]


def _build_ranking(pair_results: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for competitor in COMPETITORS:
        comparisons[competitor] = {}
        first = pair_results[f'round-1/{competitor}']
        second = pair_results[f'round-2/{competitor}']
        for workload in WORKLOADS:
            comparisons[competitor][workload] = _comparison_status(first, second, workload)
    common_win = all(
        comparisons[competitor][workload] == 'OPENBROWSER_FASTER'
        for competitor in COMPETITORS for workload in WORKLOADS
    )
    return {
        'scope': 'pairwise adapter navigation on the named local workloads only',
        'comparisons': comparisons,
        'common_navigation_win': common_win,
        'claim_limit': 'This is not a universal product-performance claim.',
    }


def _summary_text(summary: dict[str, Any]) -> str:
    lines = [
        '# Local navigation benchmark',
        '',
        'Scope: adapter-level navigation on two local fixture workloads, with task proofs and cleanup checks.',
        'These measurements do not establish a universal product-performance result.',
        '',
        '## Measured versions',
        '',
        '| Arm | Version |',
        '| --- | --- |',
    ]
    for arm, version in summary['versions'].items():
        lines.append(f'| {arm} | {version} |')
    lines.extend([
        '',
        'Browser-harness is a browser-use dependency, not a measured arm.',
        '',
        '| Dependency | Version |',
        '| --- | --- |',
    ])
    for name, version in summary['dependencies'].items():
        lines.append(f'| {name} | {version} |')
    lines.extend([
        '',
        '| Pair | Round | Arm | Workload | Samples | Errors | navigation p50 / min / max (ns) | elapsed p50 / min / max (ns) | post-navigation p50 / min / max (ns) |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ])
    for pair in summary['pair_results'].values():
        if pair['metrics'] is None:
            continue
        for arm, workloads in pair['metrics'].items():
            for workload, cell in workloads.items():
                rows = []
                for metric in METRICS:
                    values = cell[metric]
                    rows.append(f"{values['p50']} / {values['min']} / {values['max']}")
                lines.append(
                    f"| {pair['competitor']} | {pair['round']} | {arm} | {workload} | "
                    f"{cell['samples']} | {cell['errors']} | {rows[0]} | {rows[1]} | {rows[2]} |"
                )
    if 'ranking' in summary:
        lines.extend(['', '## Pairwise navigation comparison', '',
                      '| Competitor pair | Workload | Result |', '| --- | --- | --- |'])
        for competitor, workloads in summary['ranking']['comparisons'].items():
            for workload, result in workloads.items():
                lines.append(f'| {competitor} | {workload} | {result} |')
        if summary['ranking']['common_navigation_win']:
            lines.extend(['', 'OpenBrowser has a clear pairwise navigation win against all four arms on both measured workloads.'])
        else:
            lines.extend(['', 'The results do not satisfy the common-navigation win rule against every arm and workload.'])
        lines.extend(['', 'Claim limit: this is not a universal product-performance claim.'])
    lines.append('')
    return '\n'.join(lines)


def _safe_output_path(value: str | os.PathLike[str], root: Path, name: str, where: str,
                      *, allow_existing: bool) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or '..' in raw.parts or raw.name != name or raw.parent != root:
        raise InvalidPathError(f'{where} must be {name} directly below its allowed directory')
    try:
        if raw.is_symlink():
            raise InvalidPathError(f'{where} must not be a symlink')
        parent = raw.parent.resolve(strict=True)
    except OSError as error:
        raise InvalidPathError(f'{where} parent directory does not exist') from error
    if parent != root or Path(os.path.abspath(raw.parent)) != parent:
        raise InvalidPathError(f'{where} parent must use its physical path')
    if raw.exists() and not allow_existing:
        raise InvalidPathError(f'{where} already exists')
    if raw.exists() and not raw.is_file():
        raise InvalidPathError(f'{where} must be a regular file')
    return raw


def _validate_project_root(value: str | os.PathLike[str]) -> Path:
    return _resolve_existing_directory(value, 'project root')


def _sanitized_destination(project_root_value: str | os.PathLike[str],
                           output_value: str | os.PathLike[str]) -> Path:
    project_root = _validate_project_root(project_root_value)
    output = Path(output_value).expanduser()
    allowed_names = {'candidate-baseline-summary.md', 'current-benchmark-summary.md'}
    if not output.is_absolute() or '..' in output.parts or output.name not in allowed_names:
        raise InvalidPathError('sanitized output must use an approved navigation summary filename')
    expected_directory = project_root / 'local_docs' / 'specs' / '2026-11-17-latest-navigation-benchmark'
    try:
        resolved_directory = expected_directory.resolve(strict=True)
    except OSError as error:
        raise InvalidPathError('sanitized summary directory must already exist') from error
    if Path(os.path.abspath(expected_directory)) != resolved_directory or output.parent != resolved_directory:
        raise InvalidPathError('sanitized output must be directly below the approved project summary directory')
    if output.is_symlink():
        raise InvalidPathError('sanitized output must not be a symlink')
    if output.exists() and not output.is_file():
        raise InvalidPathError('sanitized output must be a regular file')
    if output.name == 'candidate-baseline-summary.md' and output.exists():
        raise InvalidPathError('candidate baseline summary is immutable and cannot be replaced')
    return output


def _atomic_write(path: Path, content: bytes, *, replace: bool) -> None:
    if path.is_symlink() or (path.exists() and not replace):
        raise InvalidPathError(f'refusing to replace existing output: {path.name}')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    descriptor = None
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            descriptor = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise InvalidPathError(f'refusing to replace symlinked output: {path.name}')
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


def _summary(run_manifest: dict[str, Any], verified: dict[str, Any], versions: dict[str, str]) -> dict[str, Any]:
    result = {
        'schema': SUMMARY_SCHEMA,
        'mode': run_manifest['mode'],
        'state': run_manifest['state'],
        'verification_state': 'complete' if verified['complete'] else 'partial',
        'source_commit': run_manifest['source_commit'],
        'source_tree': run_manifest['source_tree'],
        'candidate_identity_sha256': run_manifest['candidate_identity_sha256'],
        'version_manifest_sha256': run_manifest['version_manifest_sha256'],
        'versions': {name: versions[name] for name in ARMS_BY_MODE['run']},
        'dependencies': {'browser-harness': versions['browser-harness']},
        'pair_results': verified['pair_results'],
        'work_counts': verified['work_counts'],
        'unverified_attempt_count': verified['unverified_attempt_count'],
    }
    if verified['complete'] and run_manifest['mode'] == 'run':
        result['ranking'] = _build_ranking(verified['pair_results'])
    return result


def verify_smoke(args) -> int:
    preflight._require_no_aws()
    run_root, result_root, phase = _run_paths(args.run_root, 'smoke')
    try:
        manifest_path = preflight._artifact_file(
            run_root / RUN_MANIFEST_NAME, run_root, 'navigation smoke manifest',
        )
    except preflight.PreflightError as error:
        raise VerificationError(str(error)) from error
    manifest, _raw = _read_json(manifest_path, 'navigation smoke manifest')
    _exact_keys(manifest, RUN_KEYS, 'navigation smoke manifest')
    provenance = _validate_provenance(args, result_root, phase, manifest, 'smoke')
    _verify_current_runner_files(provenance['identity'])
    _verify_run_manifest(run_root, manifest, 'smoke', set())
    LOG.info('verified the OpenBrowser and raw-CDP navigation control smoke')
    return 0


def verify_run(args) -> tuple[dict[str, Any], Path, Path | None]:
    preflight._require_no_aws()
    run_root, result_root, phase = _run_paths(args.run_root, 'run')
    output_path = _safe_output_path(args.output, run_root, SUMMARY_NAME, 'raw summary output', allow_existing=True)
    project_root = _validate_project_root(args.project_root)
    sanitized_path = None
    if args.sanitized_output is not None:
        sanitized_path = _sanitized_destination(project_root, args.sanitized_output)
    try:
        manifest_path = preflight._artifact_file(
            run_root / RUN_MANIFEST_NAME, run_root, 'navigation run manifest',
        )
    except preflight.PreflightError as error:
        raise VerificationError(str(error)) from error
    manifest, _raw = _read_json(manifest_path, 'navigation run manifest')
    _exact_keys(manifest, RUN_KEYS, 'navigation run manifest')
    provenance = _validate_provenance(args, result_root, phase, manifest, 'run')
    _verify_current_runner_files(provenance['identity'])
    try:
        preflight.validate_frozen_smoke_preflight_hashes(
            result_root, provenance['frozen_harness'],
        )
    except preflight.PreflightError as error:
        raise VerificationError(f'frozen smoke preflight evidence is invalid: {error}') from error
    verified = _verify_run_manifest(run_root, manifest, 'run', set())
    summary = _summary(manifest, verified, provenance['versions'])
    return summary, output_path, sanitized_path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    commands = root.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('verify-smoke', help='verify only the OpenBrowser and raw-CDP controls')
    smoke.add_argument('--run-root', required=True, type=Path)
    smoke.add_argument('--candidate-identity', required=True, type=Path)
    smoke.add_argument('--preflight', required=True, type=Path)

    full = commands.add_parser('verify', help='verify and summarize a full navigation run')
    full.add_argument('--run-root', required=True, type=Path)
    full.add_argument('--frozen-harness', required=True, type=Path)
    full.add_argument('--candidate-identity', required=True, type=Path)
    full.add_argument('--preflight', required=True, type=Path)
    full.add_argument('--output', required=True, type=Path)
    full.add_argument('--project-root', required=True, type=Path)
    full.add_argument('--sanitized-output', type=Path)
    return root


def dispatch(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parser().parse_args(argv)
    try:
        if args.command == 'verify-smoke':
            return verify_smoke(args)
        summary, output_path, sanitized_path = verify_run(args)
        body = json.dumps(summary, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
        _atomic_write(output_path, body, replace=True)
        if sanitized_path is not None:
            if 'ranking' not in summary:
                LOG.error('sanitized navigation summary requires a complete verified run')
                return 1
            sanitized_body = _summary_text(summary).encode('utf-8')
            _atomic_write(sanitized_path, sanitized_body,
                          replace=sanitized_path.name == 'current-benchmark-summary.md')
        if 'ranking' in summary:
            LOG.info('verified all eight navigation pair blocks and wrote a pair-keyed summary')
        else:
            LOG.info('verified partial navigation data without producing an overall ranking')
        return 0
    except InvalidPathError as error:
        LOG.error('%s', error)
        return 2
    except (VerificationError, preflight.PreflightError, OSError, ValueError, TypeError, KeyError) as error:
        LOG.error('%s', error)
        return 1


if __name__ == '__main__':
    raise SystemExit(dispatch())
