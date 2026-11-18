from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from enum import StrEnum
import uuid
from typing import Any, Awaitable, Callable
from urllib.parse import urlencode

import navigation_compare as navigation
import latest_benchmark_preflight as preflight


CLI_DESCRIPTION = 'Run a sealed, local current-version navigation benchmark.'
LOG = logging.getLogger('latest-navigation-baseline')
RUN_SCHEMA = 'openbrowser-latest-navigation-run/1'
SMOKE_SCHEMA = 'openbrowser-latest-navigation-smoke/1'
ATTEMPT_SCHEMA = 'openbrowser-navigation-pair-attempt/1'
RUN_MANIFEST_NAME = 'run-manifest.json'
ATTEMPT_MANIFEST_NAME = 'attempt-manifest.json'
RUN_ID_LENGTH = 32
HOST_CPU_THRESHOLD = 100.0
HOST_PREFLIGHT_SECONDS = 60
HOST_PREFLIGHT_INTERVAL_SECONDS = 10
HOST_MONITOR_INTERVAL_SECONDS = 1


COMPETITORS = ('raw-cdp', 'lightpanda', 'agent-browser', 'browser-use')
WORKLOADS = ('fast', 'delayed')


class RunState(StrEnum):
    PREPARING = 'preparing'
    PREFLIGHTED = 'preflighted'
    RUNNING = 'running'
    VERIFYING = 'verifying'
    COMPLETE = 'complete'
    INCOMPLETE = 'incomplete'


class PairState(StrEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    VERIFIED = 'verified'
    FAILED = 'failed'
    CONTAMINATED = 'contaminated'


class AttemptState(StrEnum):
    PENDING = 'pending'
    RUNNING = 'running'
    VERIFIED = 'verified'
    FAILED = 'failed'
    CONTAMINATED = 'contaminated'


class HostGateState(StrEnum):
    CLEAR = 'clear'
    CONTAMINATED = 'contaminated'
    UNKNOWN = 'unknown'


_RUN_TRANSITIONS = {
    (RunState.PREPARING, 'preflight_passed'): RunState.PREFLIGHTED,
    (RunState.PREPARING, 'preflight_failed'): RunState.INCOMPLETE,
    (RunState.PREFLIGHTED, 'start'): RunState.RUNNING,
    (RunState.PREFLIGHTED, 'uncertain'): RunState.INCOMPLETE,
    (RunState.RUNNING, 'pairs_finished'): RunState.VERIFYING,
    (RunState.RUNNING, 'uncertain'): RunState.INCOMPLETE,
    (RunState.VERIFYING, 'all_pairs_verified'): RunState.COMPLETE,
    (RunState.VERIFYING, 'pair_incomplete'): RunState.INCOMPLETE,
}
_PAIR_TRANSITIONS = {
    (PairState.PENDING, 'start'): PairState.RUNNING,
    (PairState.RUNNING, 'verified'): PairState.VERIFIED,
    (PairState.RUNNING, 'failed'): PairState.FAILED,
    (PairState.RUNNING, 'contaminated'): PairState.CONTAMINATED,
}
_ATTEMPT_TRANSITIONS = {
    (AttemptState.PENDING, 'start'): AttemptState.RUNNING,
    (AttemptState.RUNNING, 'verified'): AttemptState.VERIFIED,
    (AttemptState.RUNNING, 'failed'): AttemptState.FAILED,
    (AttemptState.RUNNING, 'contaminated'): AttemptState.CONTAMINATED,
}


def _transition(table, state, event, label):
    try:
        return table[(type(state)(state), event)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'illegal {label} transition: {state!r} + {event!r}') from error


def transition_run(state, event):
    return _transition(_RUN_TRANSITIONS, state, event, 'navigation run')


def transition_pair(state, event):
    return _transition(_PAIR_TRANSITIONS, state, event, 'navigation pair')


def transition_attempt(state, event):
    return _transition(_ATTEMPT_TRANSITIONS, state, event, 'navigation attempt')


def pair_order(round_number: int) -> tuple[str, ...]:
    if type(round_number) is not int or round_number not in (1, 2):
        raise ValueError('round must be the exact integer 1 or 2')
    return COMPETITORS if round_number == 1 else tuple(reversed(COMPETITORS))


def experiment_source_order(replicate_number: int, round_number: int) -> tuple[str, str]:
    if type(replicate_number) is not int or replicate_number not in (1, 2):
        raise ValueError('replicate must be the exact integer 1 or 2')
    if type(round_number) is not int or round_number not in (1, 2):
        raise ValueError('round must be the exact integer 1 or 2')
    return ('baseline', 'treatment') if replicate_number == round_number \
        else ('treatment', 'baseline')


def sample_arm_order(sample_index: int, competitor: str) -> tuple[str, str]:
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError('sample index must be a non-negative exact integer')
    if competitor not in COMPETITORS:
        raise ValueError(f'unsupported competitor: {competitor!r}')
    pair = ('openbrowser', competitor)
    return pair if sample_index % 2 == 0 else tuple(reversed(pair))


def navigation_phase_errors(sample: dict[str, Any], label: str) -> list[str]:
    if not isinstance(sample, dict) or sample.get('outcome') != 'success':
        return []
    fields = (
        'start_ns', 'navigation_return_ns', 'end_ns', 'elapsed_ns',
        'navigation_ns', 'post_navigation_ns',
    )
    errors = []
    for field in fields:
        value = sample.get(field)
        if type(value) is not int or value < 0:
            errors.append(f'{label}: {field} must be a non-negative exact integer')
    if errors:
        return errors
    start = sample['start_ns']
    navigation_return = sample['navigation_return_ns']
    end = sample['end_ns']
    if not start <= navigation_return <= end:
        errors.append(f'{label}: navigation timestamps are out of order')
        return errors
    expected = {
        'elapsed_ns': end - start,
        'navigation_ns': navigation_return - start,
        'post_navigation_ns': end - navigation_return,
    }
    errors.extend(
        f'{label}: invalid {field}' for field, value in expected.items()
        if sample[field] != value
    )
    return errors


def classify_host_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return {'state': HostGateState.UNKNOWN.value, 'reasons': ['host sample is not an object']}
    load = sample.get('load_1m')
    logical_cpu_count = sample.get('logical_cpu_count')
    if type(logical_cpu_count) is not int or logical_cpu_count <= 0 \
            or type(load) not in (int, float) or not math.isfinite(load) or load < 0:
        return {'state': HostGateState.UNKNOWN.value, 'reasons': ['load or CPU count is invalid']}
    errors = sample.get('classification_errors')
    if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
        return {'state': HostGateState.UNKNOWN.value, 'reasons': ['process classification record is invalid']}
    if errors:
        return {'state': HostGateState.UNKNOWN.value, 'reasons': list(errors)}
    if load >= logical_cpu_count:
        return {'state': HostGateState.CONTAMINATED.value, 'reasons': ['one-minute load reached logical CPU count']}
    processes = sample.get('processes')
    if not isinstance(processes, list):
        return {'state': HostGateState.UNKNOWN.value, 'reasons': ['process samples are not a list']}
    for process in processes:
        if not isinstance(process, dict) or type(process.get('cpu_percent')) not in (int, float) \
                or not math.isfinite(process['cpu_percent']) or process['cpu_percent'] < 0:
            return {'state': HostGateState.UNKNOWN.value, 'reasons': ['process CPU sample is invalid']}
        if process.get('excluded') is not True and process['cpu_percent'] >= 100:
            return {'state': HostGateState.CONTAMINATED.value,
                    'reasons': [f'unrelated process {process.get("pid")} used at least one CPU']}
    return {'state': HostGateState.CLEAR.value, 'reasons': []}


def aggregate_host_gate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(samples, list) or not samples:
        return {'state': HostGateState.UNKNOWN.value, 'reasons': ['host monitor collected no samples']}
    checks = [classify_host_sample(sample) for sample in samples]
    reasons = [reason for check in checks for reason in check['reasons']]
    if any(check['state'] == HostGateState.UNKNOWN.value for check in checks):
        state = HostGateState.UNKNOWN.value
    elif any(check['state'] == HostGateState.CONTAMINATED.value for check in checks):
        state = HostGateState.CONTAMINATED.value
    else:
        state = HostGateState.CLEAR.value
    return {'state': state, 'reasons': list(dict.fromkeys(reasons))}


def _host_command_environment() -> dict[str, str]:
    return {'HOME': '/private/tmp', 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin',
            'TMPDIR': '/private/tmp', 'LANG': 'C'}


def logical_cpu_count() -> int:
    result = subprocess.run(
        ['/usr/sbin/sysctl', '-n', 'hw.logicalcpu'],
        check=False, capture_output=True, text=True, timeout=5,
        cwd='/private/tmp', env=_host_command_environment(),
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value.isdigit() or int(value) <= 0:
        raise ValueError('sysctl did not return a positive logical CPU count')
    return int(value)


def _owned_process_keys() -> set[tuple[int, float]]:
    root = navigation.psutil.Process(os.getpid())
    processes = [root, *root.children(recursive=True)]
    return {(process.pid, process.create_time()) for process in processes}


def one_minute_load() -> float | None:
    values = navigation.load_average()
    if not isinstance(values, (tuple, list)) or not values:
        return None
    value = values[0]
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _process_cpu_snapshot() -> dict[str, Any]:
    captured_wall_time = time.time()
    processes = {}
    errors = []
    try:
        owned = _owned_process_keys()
    except navigation.psutil.Error as error:
        owned = set()
        errors.append(f'cannot identify benchmark process tree: {type(error).__name__}')
    inventory = []
    for process in navigation.psutil.process_iter():
        try:
            pid = process.pid
            if pid == 0:
                continue
            created = process.create_time()
            inventory.append((process, pid, created, process.name()))
        except navigation.psutil.NoSuchProcess:
            continue
        except navigation.psutil.Error as error:
            errors.append(f'cannot identify process: {type(error).__name__}')
    try:
        counters = process_cpu_seconds()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        counters = {}
        errors.append(f'cannot read process CPU table: {type(error).__name__}: {error}')
    for process, pid, created, name in inventory:
        try:
            if not process.is_running():
                continue
            if pid not in counters:
                errors.append(f'process {pid} has no CPU counter')
                continue
            processes[(pid, created)] = {
                'pid': pid, 'created': created, 'name': name,
                'cpu_seconds': counters[pid],
                'excluded': (pid, created) in owned,
            }
        except navigation.psutil.NoSuchProcess:
            continue
        except navigation.psutil.Error as error:
            errors.append(f'cannot classify process CPU: {type(error).__name__}')
    return {
        'captured_monotonic': time.monotonic(),
        'captured_wall_time': captured_wall_time,
        'load_1m': one_minute_load(),
        'processes': processes,
        'classification_errors': errors,
    }


def process_cpu_seconds() -> dict[int, float]:
    result = subprocess.run(
        ['/bin/ps', '-A', '-o', 'pid=,time='], check=False, capture_output=True,
        text=True, timeout=5, cwd='/private/tmp', env=_host_command_environment(),
    )
    if result.returncode != 0:
        raise ValueError('macOS process CPU table is unavailable')
    counters = {}
    for row in result.stdout.splitlines():
        fields = row.split()
        if len(fields) != 2 or not fields[0].isdigit():
            raise ValueError('macOS process CPU table has an invalid row')
        pid = int(fields[0])
        parts = fields[1].split(':')
        if pid in counters or len(parts) not in (2, 3) or any(not part.isdigit() for part in parts[:-1]):
            raise ValueError('macOS process CPU table has invalid or duplicate counters')
        seconds = float(parts[-1])
        if not math.isfinite(seconds) or not 0 <= seconds < 60:
            raise ValueError('macOS process CPU seconds are invalid')
        total = seconds + int(parts[-2]) * 60
        if len(parts) == 3:
            if not 0 <= int(parts[-2]) < 60:
                raise ValueError('macOS process CPU minutes are invalid')
            total += int(parts[0]) * 3600
        counters[pid] = total
    if not counters:
        raise ValueError('macOS process CPU table is empty')
    return counters


def _host_sample_since(previous: dict[str, Any], logical_cpus: int) -> tuple[dict[str, Any], dict[str, Any]]:
    current = _process_cpu_snapshot()
    elapsed = current['captured_monotonic'] - previous['captured_monotonic']
    errors = [*previous['classification_errors'], *current['classification_errors']]
    process_samples = []
    if elapsed <= 0 or not math.isfinite(elapsed):
        errors.append('host sample interval is invalid')
    else:
        for key, process in current['processes'].items():
            if process['excluded']:
                cpu_percent = 0.0
            else:
                old = previous['processes'].get(key)
                if old is None:
                    started = key[1]
                    previous_wall = previous.get('captured_wall_time')
                    if previous_wall is None or started < previous_wall:
                        errors.append(f'process {process["pid"]} has no baseline CPU sample')
                        continue
                    delta = process['cpu_seconds']
                else:
                    delta = process['cpu_seconds'] - old['cpu_seconds']
                if delta < 0 or not math.isfinite(delta):
                    errors.append(f'process {process["pid"]} CPU delta is invalid')
                    continue
                cpu_percent = 100.0 * delta / elapsed
            process_samples.append({
                'pid': process['pid'], 'name': process['name'],
                'cpu_percent': cpu_percent, 'excluded': process['excluded'],
            })
    return {
        'captured_at_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        'load_1m': current['load_1m'],
        'logical_cpu_count': logical_cpus,
        'processes': process_samples,
        'classification_errors': list(dict.fromkeys(errors)),
    }, current


def _initial_host_sample(logical_cpus: int) -> dict[str, Any]:
    return {
        'captured_at_utc': datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
        'load_1m': one_minute_load(),
        'logical_cpu_count': logical_cpus,
        'processes': [],
        'classification_errors': [],
    }


async def collect_host_preflight(
    *, duration_seconds: int = HOST_PREFLIGHT_SECONDS,
    interval_seconds: int = HOST_PREFLIGHT_INTERVAL_SECONDS,
) -> dict[str, Any]:
    if type(duration_seconds) is not int or duration_seconds <= 0 \
            or type(interval_seconds) is not int or interval_seconds <= 0 \
            or duration_seconds % interval_seconds != 0:
        raise ValueError('host preflight durations must be positive whole intervals')
    cores = logical_cpu_count()
    uptime = subprocess.run(
        ['/usr/bin/uptime'], check=False, capture_output=True, text=True,
        timeout=5, cwd='/private/tmp', env=_host_command_environment(),
    )
    if uptime.returncode != 0:
        raise ValueError('uptime failed before benchmark start')
    samples = []
    previous = _process_cpu_snapshot()
    for _ in range(duration_seconds // interval_seconds):
        await asyncio.sleep(interval_seconds)
        sample, previous = _host_sample_since(previous, cores)
        samples.append(sample)
    gate = aggregate_host_gate(samples)
    return {
        'state': gate['state'], 'reasons': gate['reasons'],
        'logical_cpu_count': cores, 'uptime': uptime.stdout.strip(), 'samples': samples,
    }


async def monitor_pair_host(stop: asyncio.Event, samples: list[dict[str, Any]]) -> None:
    cores = logical_cpu_count()
    previous = _process_cpu_snapshot()
    samples.append(_initial_host_sample(cores))
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=HOST_MONITOR_INTERVAL_SECONDS)
        except TimeoutError:
            sample, previous = _host_sample_since(previous, cores)
            samples.append(sample)


def _upsert_attempt_reference(pair: dict[str, Any], reference: dict[str, Any]) -> None:
    references = pair.setdefault('attempts', [])
    attempt_number = reference.get('attempt_number')
    for index, existing in enumerate(references):
        if isinstance(existing, dict) and existing.get('attempt_number') == attempt_number:
            references[index] = {**existing, **reference}
            return
    references.append(reference)



class AttemptCheckpointFailure(RuntimeError):
    def __init__(self, error: Exception, reference: dict[str, Any]):
        super().__init__(f'attempt checkpoint failed: {type(error).__name__}: {error}')
        self.reference = reference


async def run_pair_attempts(
    competitor: str,
    execute_attempt: Callable[[int], Awaitable[dict[str, Any]]],
    wait_for_quiet: Callable[[], Awaitable[bool]],
    *,
    pair_record: dict[str, Any] | None = None,
    on_update: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    if competitor not in COMPETITORS:
        raise ValueError(f'unsupported competitor: {competitor!r}')
    pair = pair_record if pair_record is not None else {
        'competitor': competitor,
        'arms': ['openbrowser', competitor],
        'state': PairState.PENDING.value,
        'attempts': [],
        'retry_host_checks': [],
        'errors': [],
    }
    pair.setdefault('retry_host_checks', [])
    pair['state'] = transition_pair(pair['state'], 'start').value
    if on_update is not None:
        update = on_update(pair)
        if asyncio.iscoroutine(update):
            await update
    for attempt_number in (1, 2):
        if attempt_number == 2:
            if pair['attempts'][-1]['state'] != AttemptState.CONTAMINATED.value:
                break
            try:
                quiet_result = await wait_for_quiet()
            except Exception as error:
                quiet_result = {
                    'state': HostGateState.UNKNOWN.value,
                    'reasons': [f'{type(error).__name__}: {error}'],
                }
            if type(quiet_result) is bool:
                quiet_record = {
                    'state': HostGateState.CLEAR.value if quiet_result
                    else HostGateState.CONTAMINATED.value,
                    'reasons': [],
                }
            elif isinstance(quiet_result, dict):
                quiet_record = quiet_result
            else:
                quiet_record = {
                    'state': HostGateState.UNKNOWN.value,
                    'reasons': ['host quiet gate returned an invalid result'],
                }
            pair['retry_host_checks'].append(quiet_record)
            quiet = quiet_record.get('state') == HostGateState.CLEAR.value
            if quiet is not True:
                pair['errors'].extend(
                    f'host quiet gate: {reason}' for reason in quiet_record.get('reasons', [])
                )
                if on_update is not None:
                    update = on_update(pair)
                    if asyncio.iscoroutine(update):
                        await update
                break
        try:
            record = await execute_attempt(attempt_number)
        except AttemptCheckpointFailure as error:
            _upsert_attempt_reference(pair, error.reference)
            child_state = error.reference['state']
            pair['state'] = (
                PairState.RUNNING.value if child_state in {
                    AttemptState.PENDING.value, AttemptState.CONTAMINATED.value,
                }
                else child_state
            )
            raise
        except Exception as error:
            record = {'state': AttemptState.FAILED.value, 'cleanup_ok': False,
                      'errors': [f'{type(error).__name__}: {error}']}
        except BaseException as interruption:
            checkpoint_failure = interruption.__cause__
            if isinstance(checkpoint_failure, AttemptCheckpointFailure):
                _upsert_attempt_reference(pair, checkpoint_failure.reference)
            if pair['attempts']:
                child_state = pair['attempts'][-1]['state']
                pair['state'] = (
                    PairState.RUNNING.value if child_state in {
                        AttemptState.PENDING.value, AttemptState.CONTAMINATED.value,
                    }
                    else child_state
                )
            raise
        if not isinstance(record, dict):
            record = {'state': AttemptState.FAILED.value, 'cleanup_ok': False,
                      'errors': ['pair attempt returned no record']}
        record = {**record, 'attempt_number': attempt_number}
        if record.get('state') not in {state.value for state in AttemptState} - {AttemptState.PENDING.value,
                                                                                 AttemptState.RUNNING.value}:
            record['state'] = AttemptState.FAILED.value
            if not isinstance(record.get('errors'), list):
                record['errors'] = []
            record['errors'].append('pair attempt returned an invalid terminal state')
        if not isinstance(record.get('errors'), list):
            record['errors'] = ['pair attempt errors are not a string list']
            record['state'] = AttemptState.FAILED.value
        counts = record.get('work_counts')
        if not isinstance(counts, dict) or any(
            type(counts.get(field)) is not int or counts[field] < 0
            for field in ('attempted_work_count', 'successful_work_count', 'error_count')
        ):
            counts = zero_work_counts()
            record['work_counts'] = counts
        if type(record.get('error_count')) is not int or record['error_count'] < 0:
            record['error_count'] = counts['error_count']
        if record['state'] == AttemptState.FAILED.value and record['errors']:
            record['error_count'] = max(record['error_count'], len(record['errors']))
            counts['error_count'] = max(counts['error_count'], record['error_count'])
        if record.get('cleanup_ok') is not True:
            if record['state'] != AttemptState.FAILED.value or not record['errors']:
                record['errors'].append('worker cleanup is uncertain')
                record['error_count'] += 1
                counts['error_count'] += 1
            record['state'] = AttemptState.FAILED.value
        _upsert_attempt_reference(pair, record)
        if on_update is not None:
            update = on_update(pair)
            if asyncio.iscoroutine(update):
                await update
        if record.get('cleanup_ok') is not True:
            break
        if record['state'] != AttemptState.CONTAMINATED.value:
            break
    final_state = pair['attempts'][-1]['state'] if pair['attempts'] else AttemptState.FAILED.value
    pair['state'] = transition_pair(pair['state'], final_state).value
    if final_state == AttemptState.FAILED.value:
        pair['errors'].extend(pair['attempts'][-1].get('errors', []))
    if on_update is not None:
        update = on_update(pair)
        if asyncio.iscoroutine(update):
            await update
    return pair


def run_can_rank(manifest: dict[str, Any]) -> bool:
    if not isinstance(manifest, dict) or manifest.get('state') != RunState.COMPLETE.value:
        return False
    if not isinstance(manifest.get('cleanup'), dict) or manifest['cleanup'].get('ok') is not True:
        return False
    if not isinstance(manifest.get('host_preflight'), dict) \
            or manifest['host_preflight'].get('state') != HostGateState.CLEAR.value:
        return False
    if manifest.get('errors') != []:
        return False
    pairs = manifest.get('pairs')
    if not isinstance(pairs, list) or len(pairs) != len(COMPETITORS) * 2:
        return False
    expected = {(round_number, competitor) for round_number in (1, 2) for competitor in COMPETITORS}
    actual = {
        (pair.get('round'), pair.get('competitor'))
        for pair in pairs if isinstance(pair, dict)
    }
    return actual == expected and all(
        isinstance(pair, dict) and pair.get('state') == PairState.VERIFIED.value
        for pair in pairs
    )


def arms_for_mode(mode: str) -> tuple[str, ...]:
    if mode == 'smoke':
        return ('openbrowser', 'raw-cdp')
    if mode == 'run':
        return ('openbrowser', *COMPETITORS)
    raise ValueError(f'unsupported navigation mode: {mode!r}')


def navigation_run_exit_code(manifest: dict[str, Any]) -> int:
    if not isinstance(manifest, dict) or manifest.get('state') != RunState.COMPLETE.value \
            or manifest.get('errors') != []:
        return 1
    cleanup = manifest.get('cleanup')
    fixture = manifest.get('fixture')
    if not isinstance(cleanup, dict) or cleanup.get('ok') is not True \
            or not isinstance(fixture, dict) or fixture.get('stopped') is not True:
        return 1
    mode = manifest.get('mode')
    if mode == 'run':
        return 0 if run_can_rank(manifest) else 1
    if mode == 'smoke':
        pairs = manifest.get('pairs')
        if not isinstance(pairs, list) or len(pairs) != 1:
            return 1
        pair = pairs[0]
        if not isinstance(pair, dict) or pair.get('round') != 1 \
                or pair.get('competitor') != 'raw-cdp' \
                or pair.get('arms') != ['openbrowser', 'raw-cdp'] \
                or pair.get('state') != PairState.VERIFIED.value:
            return 1
        return 0
    return 1


def result_root_for_output(value: str | os.PathLike[str], mode: str) -> tuple[Path, Path]:
    raw = Path(value).expanduser()
    if not raw.is_absolute() or '..' in raw.parts:
        raise ValueError('navigation output must be an absolute path without parent traversal')
    physical_tmp, output = navigation.physical_profile_path(raw)
    if mode == 'smoke':
        if output.name != 'navigation' or output.parent.name != 'smoke':
            raise ValueError('smoke output must be RESULT_ROOT/smoke/navigation')
        result_root = output.parent.parent
    elif mode == 'run':
        if output.name not in ('navigation', 'navigation-final'):
            raise ValueError('run output must be RESULT_ROOT/navigation or navigation-final')
        result_root = output.parent
    else:
        raise ValueError(f'unsupported navigation mode: {mode!r}')
    if result_root.parent != physical_tmp:
        raise ValueError('RESULT_ROOT must be a direct child of physical /tmp')
    result_root = navigation.validate_profile_result_root(result_root)
    if output.exists() or output.is_symlink():
        raise ValueError('navigation output must be a new directory')
    return result_root, output


def create_private_directory(path: str | os.PathLike[str], *, parents: bool = False) -> Path:
    directory = Path(path)
    if directory.exists() or directory.is_symlink():
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f'private output path is not a directory: {directory}')
        navigation.validate_private_directory(directory)
        return directory
    directory.mkdir(mode=0o700, parents=parents, exist_ok=False)
    directory.chmod(0o700)
    navigation.validate_private_directory(directory)
    return directory


def create_private_child(parent: str | os.PathLike[str], name: str) -> Path:
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('private child name must be a basename')
    return create_private_directory(Path(parent) / name)


def atomic_json(path: str | os.PathLike[str], value: dict[str, Any]) -> None:
    destination = Path(path)
    body = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.tmp')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.is_symlink():
            raise ValueError(f'refusing to replace symlinked manifest: {destination}')
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_new_json(path: str | os.PathLike[str], value: dict[str, Any]) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'refusing to overwrite navigation artifact: {destination}')
    atomic_json(destination, value)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def read_json_artifact(result_root: Path, value: str | os.PathLike[str], where: str):
    path = preflight._artifact_file(value, result_root, where)
    parsed, raw = preflight._strict_json(path, where)
    return path, parsed, raw


def validate_navigation_inputs(args, result_root: Path, mode: str) -> dict[str, Any]:
    preflight._require_no_aws()
    expected_phase = (
        'navigation-smoke' if mode == 'smoke'
        else 'navigation-final' if Path(args.output).name == 'navigation-final'
        else 'navigation-run'
    )
    identity_path, identity_value, identity_raw = read_json_artifact(
        result_root, args.candidate_identity, 'candidate identity',
    )
    identity = preflight._validate_identity(identity_value)
    identity_sha = preflight.sha256_bytes(identity_raw)
    version_path, version_value, version_raw = read_json_artifact(
        result_root, args.version_manifest, 'version manifest',
    )
    version = preflight._validate_version_manifest(version_value)
    version_sha = preflight.sha256_bytes(version_raw)
    preflight_path, preflight_value, _preflight_raw = read_json_artifact(
        result_root, args.preflight, 'navigation preflight',
    )
    preflight_value = preflight._validate_preflight_result(preflight_value)
    if any(path.parent != result_root for path in (identity_path, version_path, preflight_path)):
        raise ValueError('identity, version, and preflight artifacts must be direct children of RESULT_ROOT')
    if preflight_path != result_root / f'preflight-{expected_phase}.json':
        raise ValueError(f'navigation preflight must use preflight-{expected_phase}.json')
    if preflight_value.get('phase') != expected_phase or preflight_value.get('outcome') != 'passed' \
            or preflight_value.get('errors') != []:
        raise ValueError(f'{expected_phase} preflight did not pass')
    if not isinstance(preflight_value.get('checks'), list) \
            or any(not isinstance(check, dict) or check.get('state') != 'passed'
                   for check in preflight_value['checks']):
        raise ValueError(f'{expected_phase} preflight contains a failed or invalid check')
    if preflight_value.get('candidate_identity_sha256') != identity_sha \
            or preflight_value.get('version_manifest_sha256') != version_sha:
        raise ValueError('navigation preflight does not match the candidate and version manifests')
    if version.get('candidate_identity_sha256') != identity_sha:
        raise ValueError('version manifest does not match the candidate identity')
    source_root = preflight._source_root(args.source, result_root=result_root)
    source_snapshot = preflight._git_snapshot(source_root, result_root=result_root)
    if any(identity.get(key) != value for key, value in source_snapshot.items()):
        raise ValueError('navigation source does not match the sealed candidate identity')
    if preflight._hash_executed_files(source_root) != identity['executed_files']:
        raise ValueError('navigation runner or verifier files changed after identity creation')
    candidate_python = preflight._inspect_python(args.python, root=result_root, require_current=True)
    if candidate_python != identity['candidate_python']:
        raise ValueError('navigation runner Python differs from the candidate identity')

    tool_manifest_path = preflight._artifact_file(
        result_root / 'tool-manifest.json', result_root, 'tool manifest',
    )
    tool_value, tool_raw = preflight._strict_json(tool_manifest_path, 'tool manifest')
    tools = preflight._validate_tool_manifest(
        tool_value,
        result_root=result_root,
        tools_root=result_root / 'tools',
        verify_installation=True,
    )
    tool_sha = preflight.sha256_bytes(tool_raw)
    if version.get('tool_manifest_sha256') != tool_sha:
        raise ValueError('tool manifest does not match the sealed version manifest')
    tool_paths = {
        'chrome': tools['tools']['chrome_for_testing']['path'],
        'lightpanda': tools['tools']['lightpanda']['path'],
        'agent_browser': tools['tools']['agent-browser']['path'],
    }
    for name, expected in tool_paths.items():
        supplied = getattr(args, name, None)
        if supplied is not None and Path(supplied).resolve(strict=True) != Path(expected):
            raise ValueError(f'navigation {name} path differs from the verified tool manifest')
    if mode == 'run':
        browser_python = preflight._inspect_python(args.browser_use_python, root=result_root)
        if browser_python != identity['browser_use_python']:
            raise ValueError('browser-use Python differs from the candidate identity')

    frozen_path = None
    frozen = None
    frozen_raw = None
    if mode == 'run':
        frozen_path, frozen, frozen_raw = read_json_artifact(
            result_root, args.frozen_harness, 'frozen harness',
        )
        if frozen_path != result_root / 'frozen-harness.json':
            raise ValueError('frozen harness must be RESULT_ROOT/frozen-harness.json')
        frozen = preflight._validate_frozen_harness(
            frozen,
            identity_sha=identity_sha,
            candidate_freeze_sha=preflight.sha256_file(result_root / 'candidate-freeze.txt'),
            browser_use_freeze_sha=preflight.sha256_file(result_root / 'browser-use-freeze.txt'),
            tool_manifest_sha=tool_sha,
        )
        preflight.validate_frozen_smoke_preflight_hashes(result_root, frozen)
        if frozen.get('version_manifest_sha256') != version_sha \
                or frozen.get('executed_file_sha256') != identity['executed_files']:
            raise ValueError('frozen harness does not match the sealed version and runner files')

    return {
        'identity': identity,
        'identity_path': identity_path,
        'identity_sha256': identity_sha,
        'version': version,
        'version_path': version_path,
        'version_sha256': version_sha,
        'preflight': preflight_value,
        'preflight_path': preflight_path,
        'preflight_sha256': preflight.sha256_file(preflight_path),
        'tool_manifest': tools,
        'tool_manifest_path': tool_manifest_path,
        'tool_manifest_sha256': tool_sha,
        'frozen_harness': frozen,
        'frozen_harness_path': frozen_path,
        'frozen_harness_raw': frozen_raw,
        'frozen_harness_sha256': preflight.sha256_bytes(frozen_raw) if frozen_raw else None,
        'source_root': source_root,
        'source_snapshot': source_snapshot,
        'harness_sha256': navigation.sha256(navigation.SCRIPT),
    }


def _initial_run_manifest(mode: str, run_id: str, inputs: dict[str, Any], *, rounds: int,
                          samples_per_workload: int) -> dict[str, Any]:
    schedule = (
        {'1': ['raw-cdp']}
        if mode == 'smoke'
        else {str(round_number): list(pair_order(round_number))
              for round_number in range(1, rounds + 1)}
    )
    return {
        'schema': SMOKE_SCHEMA if mode == 'smoke' else RUN_SCHEMA,
        'mode': mode,
        'run_id': run_id,
        'state': RunState.PREPARING.value,
        'created_at_utc': utc_now(),
        'candidate_identity_sha256': inputs['identity_sha256'],
        'version_manifest_sha256': inputs['version_sha256'],
        'preflight_sha256': inputs['preflight_sha256'],
        'frozen_harness_sha256': inputs['frozen_harness_sha256'],
        'harness_sha256': navigation.sha256(__file__),
        'source_commit': inputs['identity']['candidate_commit'],
        'source_tree': inputs['identity']['candidate_tree'],
        'arms': list(arms_for_mode(mode)),
        'workloads': list(WORKLOADS),
        'rounds': rounds,
        'samples_per_workload': samples_per_workload,
        'pair_order_by_round': schedule,
        'pairs': [],
        'host_preflight': {'state': HostGateState.UNKNOWN.value, 'samples': [], 'reasons': []},
        'work_counts': {
            'attempted_work_count': 0,
            'successful_work_count': 0,
            'error_count': 0,
            'host_contaminated_attempt_count': 0,
        },
        'fixture': {
            'started': False, 'stopped': False,
            'socket_closed': False, 'thread_stopped': False,
        },
        'cleanup': {
            'ok': False, 'workers_ok': False,
            'runtime_roots_removed': True, 'fixture_stopped': False,
        },
        'failed_phase': None,
        'errors': [],
    }


def _update_run_work_counts(manifest: dict[str, Any]) -> None:
    totals = {
        'attempted_work_count': 0,
        'successful_work_count': 0,
        'error_count': 0,
        'host_contaminated_attempt_count': 0,
    }
    for pair in manifest.get('pairs', []):
        if not isinstance(pair, dict):
            continue
        for attempt in pair.get('attempts', []):
            if not isinstance(attempt, dict):
                continue
            counts = attempt.get('work_counts', {})
            for field in ('attempted_work_count', 'successful_work_count', 'error_count'):
                value = counts.get(field) if isinstance(counts, dict) else None
                if type(value) is int and value >= 0:
                    totals[field] += value
            if attempt.get('host_gate', {}).get('state') != HostGateState.CLEAR.value:
                totals['host_contaminated_attempt_count'] += 1
    manifest['work_counts'] = totals


def _pair_attempt_reference(run_root: Path, manifest_path: Path, attempt_manifest: dict[str, Any]) -> dict[str, Any]:
    relative = manifest_path.parent.relative_to(run_root).as_posix()
    return deepcopy({
        'attempt_number': attempt_manifest['attempt_number'],
        'state': attempt_manifest['state'],
        'directory': relative,
        'manifest_sha256': preflight.sha256_file(manifest_path),
        'cleanup_ok': attempt_manifest['cleanup']['ok'],
        'runtime_root_removed': attempt_manifest['cleanup']['runtime_root_removed'],
        'host_gate': attempt_manifest['host_gate'],
        'work_counts': attempt_manifest['work_counts'],
        'error_count': attempt_manifest['work_counts']['error_count'],
        'errors': attempt_manifest['errors'],
    })


async def _stop_fixture(fixture, thread, record):
    if fixture is None:
        record['stopped'] = True
        record['socket_closed'] = True
        record['thread_stopped'] = True
        return []
    errors = []
    try:
        await asyncio.to_thread(fixture.shutdown)
    except Exception as error:
        errors.append(f'fixture shutdown failed: {type(error).__name__}: {error}')
    try:
        fixture.server_close()
        record['socket_closed'] = True
    except Exception as error:
        errors.append(f'fixture socket close failed: {type(error).__name__}: {error}')
    if thread is not None:
        try:
            await asyncio.to_thread(thread.join, 5)
            record['thread_stopped'] = not thread.is_alive()
        except Exception as error:
            errors.append(f'fixture thread join failed: {type(error).__name__}: {error}')
    record['stopped'] = record['socket_closed'] and record['thread_stopped']
    if not record['stopped']:
        errors.append('fixture thread or socket cleanup is unverified')
    return errors


def _run_pair_expected(mode: str, rounds: int) -> list[tuple[int, str]]:
    if mode == 'smoke':
        return [(1, 'raw-cdp')]
    return [(round_number, competitor)
            for round_number in range(1, rounds + 1)
            for competitor in pair_order(round_number)]


def _run_attempt_work_counts(manifest: dict[str, Any]) -> dict[str, int]:
    _update_run_work_counts(manifest)
    return manifest['work_counts']


async def run_navigation(args) -> int:
    mode = args.command
    result_root, output = result_root_for_output(args.output, mode)
    inputs = validate_navigation_inputs(args, result_root, mode)
    if mode == 'run' and (type(args.rounds) is not int or args.rounds != 2
                          or type(args.samples) is not int or args.samples != 6):
        raise ValueError('full baseline uses exactly two rounds and six samples')
    rounds = 1 if mode == 'smoke' else args.rounds
    samples_per_workload = 1 if mode == 'smoke' else args.samples
    if mode == 'smoke':
        create_private_child(result_root, 'smoke')
    create_private_child(output.parent, output.name)

    run_id = uuid.uuid4().hex
    manifest = _initial_run_manifest(
        mode, run_id, inputs, rounds=rounds,
        samples_per_workload=samples_per_workload,
    )
    manifest_path = output / RUN_MANIFEST_NAME
    write_new_json(manifest_path, manifest)

    try:
        host_preflight = await collect_host_preflight()
    except Exception as error:
        host_preflight = {
            'state': HostGateState.UNKNOWN.value,
            'reasons': [f'{type(error).__name__}: {error}'],
            'logical_cpu_count': None,
            'uptime': None,
            'samples': [],
        }
    manifest['host_preflight'] = host_preflight
    if host_preflight.get('state') != HostGateState.CLEAR.value:
        manifest['failed_phase'] = 'host_preflight'
        manifest['errors'].extend(host_preflight.get('reasons', []))
        manifest['state'] = transition_run(RunState.PREPARING, 'preflight_failed').value
        manifest['cleanup'] = {
            'ok': False, 'workers_ok': True,
            'runtime_roots_removed': True, 'fixture_stopped': True,
        }
        atomic_json(manifest_path, manifest)
        return 1

    manifest['state'] = transition_run(RunState.PREPARING, 'preflight_passed').value
    atomic_json(manifest_path, manifest)
    fixture = None
    fixture_thread = None
    stop_new_work = False
    runtime_roots_removed = True
    workers_ok = True
    fixture_errors = []
    interruption: BaseException | None = None
    try:
        fixture = navigation.Fixture(.15)
        fixture_thread = threading.Thread(target=fixture.serve_forever, daemon=True)
        fixture_thread.start()
        manifest['fixture']['started'] = True
        manifest['state'] = transition_run(RunState.PREFLIGHTED, 'start').value
        atomic_json(manifest_path, manifest)

        for round_number, competitors in (
            [(1, ('raw-cdp',))] if mode == 'smoke'
            else [(number, pair_order(number)) for number in range(1, rounds + 1)]
        ):
            for competitor in competitors:
                pair = {
                    'round': round_number,
                    'competitor': competitor,
                    'arms': ['openbrowser', competitor],
                    'state': PairState.PENDING.value,
                    'attempts': [],
                    'retry_host_checks': [],
                    'errors': [],
                }
                manifest['pairs'].append(pair)
                _run_attempt_work_counts(manifest)
                atomic_json(manifest_path, manifest)

                async def save_pair(_pair):
                    _run_attempt_work_counts(manifest)
                    atomic_json(manifest_path, manifest)

                async def save_attempt(reference):
                    _upsert_attempt_reference(pair, reference)
                    await save_pair(pair)

                async def execute_attempt(attempt_number):
                    _upsert_attempt_reference(pair, {
                        'attempt_number': attempt_number,
                        'state': AttemptState.PENDING.value,
                        'directory': (
                            f'round-{round_number}/pair-{competitor}/attempt-{attempt_number}'
                        ),
                        'manifest_sha256': None,
                        'cleanup_ok': False,
                        'runtime_root_removed': False,
                        'host_gate': {
                            'state': HostGateState.UNKNOWN.value,
                            'reasons': ['attempt manifest has not been created'],
                        },
                        'work_counts': zero_work_counts(),
                        'error_count': 0,
                        'errors': [],
                    })
                    await save_pair(pair)
                    return await run_pair_attempt(
                        args,
                        result_root=result_root,
                        run_root=output,
                        fixture=fixture,
                        run_id=run_id,
                        round_number=round_number,
                        competitor=competitor,
                        attempt_number=attempt_number,
                        samples_per_workload=samples_per_workload,
                        on_update=save_attempt,
                    )

                async def wait_for_quiet():
                    return await collect_host_preflight(
                        duration_seconds=HOST_PREFLIGHT_INTERVAL_SECONDS,
                        interval_seconds=HOST_PREFLIGHT_INTERVAL_SECONDS,
                    )

                pair_result = await run_pair_attempts(
                    competitor, execute_attempt, wait_for_quiet,
                    pair_record=pair, on_update=save_pair,
                )
                if any(attempt.get('cleanup_ok') is not True for attempt in pair_result['attempts']):
                    workers_ok = False
                    runtime_roots_removed = False
                    manifest['failed_phase'] = 'cleanup'
                    stop_new_work = True
                if pair_result['state'] == PairState.CONTAMINATED.value:
                    quiet = await wait_for_quiet()
                    pair_result['retry_host_checks'].append(quiet)
                    if not isinstance(quiet, dict) or quiet.get('state') != HostGateState.CLEAR.value:
                        manifest['failed_phase'] = 'host_gate'
                        manifest['errors'].append('host remained contaminated after a pair attempt')
                        stop_new_work = True
                _run_attempt_work_counts(manifest)
                atomic_json(manifest_path, manifest)
                if stop_new_work:
                    break
            if stop_new_work:
                break
    except BaseException as error:
        manifest['failed_phase'] = manifest.get('failed_phase') or 'run'
        manifest['errors'].append(f'{type(error).__name__}: {error}')
        if isinstance(error.__cause__, AttemptCheckpointFailure):
            manifest['errors'].append(str(error.__cause__))
        if not isinstance(error, Exception):
            interruption = error
        stop_new_work = True
    finally:
        fixture_task = asyncio.create_task(
            _stop_fixture(fixture, fixture_thread, manifest['fixture'])
        )
        while True:
            try:
                fixture_errors = await asyncio.shield(fixture_task)
                break
            except asyncio.CancelledError as error:
                if fixture_task.cancelled():
                    raise
                if interruption is None:
                    interruption = error
                    manifest['failed_phase'] = manifest.get('failed_phase') or 'run'
                    manifest['errors'].append(f'{type(error).__name__}: {error}')
        if fixture_errors:
            manifest['failed_phase'] = manifest.get('failed_phase') or 'cleanup'
            manifest['errors'].extend(fixture_errors)

    try:
        final_snapshot = preflight._git_snapshot(inputs['source_root'], result_root=result_root)
        if final_snapshot != inputs['source_snapshot'] \
                or preflight._hash_executed_files(inputs['source_root']) != inputs['identity']['executed_files']:
            manifest['failed_phase'] = manifest.get('failed_phase') or 'source'
            manifest['errors'].append('candidate source or benchmark scripts changed during the run')
    except Exception as error:
        manifest['failed_phase'] = manifest.get('failed_phase') or 'source'
        manifest['errors'].append(f'source stability check failed: {type(error).__name__}: {error}')

    attempts = [attempt for pair in manifest['pairs'] for attempt in pair['attempts']]
    workers_ok = workers_ok and all(attempt.get('cleanup_ok') is True for attempt in attempts)
    runtime_roots_removed = runtime_roots_removed and all(
        attempt.get('runtime_root_removed') is True for attempt in attempts
    )
    fixture_stopped = manifest['fixture'].get('stopped') is True
    manifest['cleanup'] = {
        'ok': workers_ok and runtime_roots_removed and fixture_stopped,
        'workers_ok': workers_ok,
        'runtime_roots_removed': runtime_roots_removed,
        'fixture_stopped': fixture_stopped,
    }
    if manifest['state'] == RunState.RUNNING.value:
        manifest['state'] = transition_run(RunState.RUNNING, 'pairs_finished').value
    elif manifest['state'] == RunState.PREFLIGHTED.value:
        manifest['state'] = transition_run(RunState.PREFLIGHTED, 'uncertain').value
    if manifest['state'] == RunState.VERIFYING.value:
        expected_pairs = _run_pair_expected(mode, rounds)
        actual_pairs = [
            (pair.get('round'), pair.get('competitor'))
            for pair in manifest['pairs'] if isinstance(pair, dict)
        ]
        pairs_verified = actual_pairs == expected_pairs and all(
            pair.get('state') == PairState.VERIFIED.value for pair in manifest['pairs']
        )
        complete = pairs_verified and manifest['cleanup']['ok'] \
            and manifest['host_preflight'].get('state') == HostGateState.CLEAR.value \
            and not manifest['errors']
        terminal_event = 'all_pairs_verified' if complete else 'pair_incomplete'
        manifest['state'] = transition_run(RunState.VERIFYING, terminal_event).value
    _run_attempt_work_counts(manifest)
    try:
        atomic_json(manifest_path, manifest)
    except Exception:
        if interruption is None:
            raise
        LOG.exception('cannot persist interrupted run finalization')
    if interruption is not None:
        raise interruption
    return navigation_run_exit_code(manifest)


def zero_work_counts() -> dict[str, int]:
    return {'attempted_work_count': 0, 'successful_work_count': 0, 'error_count': 0}


def work_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    records = [sample for sample in samples if isinstance(sample, dict)]
    return {
        'attempted_work_count': len(records),
        'successful_work_count': sum(sample.get('state') == AttemptState.VERIFIED.value for sample in records),
        'error_count': sum(
            len(sample.get('errors', [])) for sample in records
            if isinstance(sample.get('errors'), list)
        ),
    }


async def run_pair_attempt(
    args,
    *,
    result_root: Path,
    run_root: Path,
    fixture: navigation.Fixture,
    run_id: str,
    round_number: int,
    competitor: str,
    attempt_number: int,
    samples_per_workload: int,
    on_update: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    result_root = navigation.validate_profile_result_root(result_root)
    run_root = navigation.validate_private_directory(run_root)
    if not run_root.is_relative_to(result_root):
        raise ValueError('navigation run root must be below RESULT_ROOT')
    if competitor not in COMPETITORS or type(attempt_number) is not int or attempt_number not in (1, 2):
        raise ValueError('pair competitor or attempt number is invalid')
    attempt_dir = create_pair_directories(run_root, round_number, competitor, attempt_number)
    log_directory = create_private_child(attempt_dir, 'logs')
    arms = ('openbrowser', competitor)
    attempt_manifest = {
        'schema': ATTEMPT_SCHEMA,
        'run_id': run_id,
        'round': round_number,
        'competitor': competitor,
        'attempt_number': attempt_number,
        'state': AttemptState.PENDING.value,
        'execution_order': [],
        'warmups': [],
        'samples': [],
        'host_samples': [],
        'host_gate': {'state': HostGateState.UNKNOWN.value, 'reasons': ['not started']},
        'work_counts': zero_work_counts(),
        'cleanup': {
            'ok': False,
            'workers_ok': False,
            'runtime_root': None,
            'runtime_root_removed': False,
            'workers': {},
            'errors': [],
        },
        'errors': [],
    }
    manifest_path = attempt_dir / ATTEMPT_MANIFEST_NAME

    async def publish_attempt_reference() -> None:
        if on_update is None:
            return
        reference = _pair_attempt_reference(run_root, manifest_path, attempt_manifest)
        update = on_update(reference)
        if asyncio.iscoroutine(update):
            await update

    async def persist_attempt_manifest() -> None:
        atomic_json(manifest_path, attempt_manifest)
        await publish_attempt_reference()

    handles: dict[str, navigation.WorkerHandle] = {}
    warmups: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    host_samples: list[dict[str, Any]] = []
    host_stop = asyncio.Event()
    host_task = None
    interruption: BaseException | None = None
    pending_trial: tuple[dict[str, Any], bool, str] | None = None
    runtime_root, owner_token = new_runtime_root()
    try:
        attempt_manifest['cleanup']['runtime_root'] = str(runtime_root)
        write_new_json(manifest_path, attempt_manifest)
        await publish_attempt_reference()
        attempt_manifest['state'] = transition_attempt(
            AttemptState.PENDING, 'start',
        ).value
        await persist_attempt_manifest()
        source_root = Path(args.source).expanduser().resolve(strict=True)
        for arm in arms:
            launch = navigation.make_worker_launch(
                arm=arm,
                mode=navigation.WorkerMode.NAVIGATION,
                source_root=source_root,
                candidate_python=Path(args.python).expanduser().absolute(),
                browser_use_python=(Path(args.browser_use_python).expanduser().absolute()
                                    if getattr(args, 'browser_use_python', None) else None),
                runtime_root=runtime_root,
                worker_root=runtime_root / arm,
                working_directory=runtime_root / arm,
                log_directory=log_directory,
                log_prefix=arm,
                owner_token=owner_token,
                chrome=Path(args.chrome).expanduser().absolute(),
                lightpanda=(Path(args.lightpanda).expanduser().absolute()
                            if getattr(args, 'lightpanda', None) else None),
                agent_browser=(Path(args.agent_browser).expanduser().absolute()
                               if getattr(args, 'agent_browser', None) else None),
                node=(Path(args.node).expanduser().absolute() if getattr(args, 'node', None) else None),
            )
            handle = await navigation.launch_worker(launch)
            handles[arm] = handle
            response, start_errors = await navigation.start_worker(handle)
            if start_errors or not isinstance(response, dict) or response.get('ok') is not True:
                errors.extend(f'{arm} start: {error}' for error in start_errors)
                if not start_errors:
                    errors.append(f'{arm} worker did not start successfully')
                break

        async def execute(arm: str, workload: str, index: int, *, is_warmup: bool):
            nonlocal pending_trial
            request = _trial_request(
                run_id, round_number, competitor, attempt_number,
                arm, workload, index, fixture.server_address[1],
            )
            label = f'{round_number}/{competitor}/{attempt_number}/{arm}/{workload}/{index}'
            if not is_warmup:
                await persist_attempt_manifest()
                attempt_manifest['execution_order'].append({
                    'workload': workload, 'index': index, 'arm': arm,
                })
            pending_trial = (request, is_warmup, label)
            record = await execute_trial(handles[arm], fixture, request, is_warmup=is_warmup)
            pending_trial = None
            return record, label

        if len(handles) == len(arms) and not errors:
            for arm in arms:
                warmup, label = await execute(arm, 'fast', 0, is_warmup=True)
                warmups.append(warmup)
                if warmup['state'] != AttemptState.VERIFIED.value:
                    errors.extend(f'{label}: {error}' for error in warmup['errors'])
                attempt_manifest['warmups'] = warmups
                attempt_manifest['errors'] = errors
                attempt_manifest['work_counts'] = {
                    **work_counts(samples), 'error_count': len(errors),
                }
                await persist_attempt_manifest()
            if not errors:
                host_task = asyncio.create_task(monitor_pair_host(host_stop, host_samples))
                for workload in WORKLOADS:
                    for index in range(samples_per_workload):
                        for arm in sample_arm_order(index, competitor):
                            sample, label = await execute(arm, workload, index, is_warmup=False)
                            samples.append(sample)
                            if sample['state'] != AttemptState.VERIFIED.value:
                                sample_errors = sample.get('errors', [])
                                errors.extend(
                                    f'{label}: {error}' for error in sample_errors
                                )
                                if not sample_errors:
                                    errors.append(f'{label}: failed sample has no error detail')
                            attempt_manifest['samples'] = samples
                            attempt_manifest['work_counts'] = {
                                **work_counts(samples), 'error_count': len(errors),
                            }
                            attempt_manifest['errors'] = errors
                            await persist_attempt_manifest()
                            if sample['state'] != AttemptState.VERIFIED.value:
                                if handles[arm].state is not navigation.WorkerState.READY:
                                    break
                        if errors:
                            break
                    if errors:
                        break
    except BaseException as error:
        LOG.debug("attempt execution failed", exc_info=True)
        failure = f"{type(error).__name__}: {error}"
        if pending_trial is None:
            errors.append(failure)
        else:
            request, is_warmup, label = pending_trial
            trial_errors = [failure]
            try:
                request_proof = fixture.snapshot(request["nonce"])
            except Exception as proof_error:
                LOG.debug("interrupted trial proof failed", exc_info=True)
                request_proof = {"requests": [], "effects": 0}
                trial_errors.append(
                    f"fixture proof failed: {type(proof_error).__name__}: {proof_error}"
                )
            record = {
                "arm": request["arm"],
                "workload": request["kind"],
                "index": request["index"],
                "nonce": request["nonce"],
                "url": request["url"],
                "warmup": is_warmup,
                "state": AttemptState.FAILED.value,
                "sample": None,
                "request_proof": request_proof,
                "errors": trial_errors,
            }
            (warmups if is_warmup else samples).append(record)
            errors.extend(f"{label}: {trial_error}" for trial_error in trial_errors)
        if not isinstance(error, Exception):
            interruption = error
    finally:

        async def cleanup_owned_resources() -> dict[str, Any]:
            nonlocal interruption
            if host_task is not None:
                host_stop.set()
                try:
                    await host_task
                except BaseException as error:
                    LOG.debug("host monitor failed", exc_info=True)
                    errors.append(
                        f"host monitor failed: {type(error).__name__}: {error}"
                    )
                    if not isinstance(error, Exception) and interruption is None:
                        interruption = error
            return await cleanup_pair_workers(handles, arms, runtime_root, owner_token)

        async def complete_owned(awaitable):
            nonlocal interruption
            task = asyncio.create_task(awaitable)
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    if task.cancelled():
                        raise
                    if interruption is None:
                        errors.append(f"{type(error).__name__}: {error}")
                        interruption = error

        try:
            cleanup = await complete_owned(cleanup_owned_resources())
            errors.extend(cleanup["errors"])
            host_gate = aggregate_host_gate(host_samples)
            counts = work_counts(samples)
            counts["error_count"] = len(errors)
            if not cleanup["ok"] or errors:
                terminal_state = AttemptState.FAILED.value
            elif host_gate["state"] != HostGateState.CLEAR.value:
                terminal_state = AttemptState.CONTAMINATED.value
            elif any(
                sample.get("state") != AttemptState.VERIFIED.value for sample in samples
            ) or any(
                warmup.get("state") != AttemptState.VERIFIED.value for warmup in warmups
            ):
                terminal_state = AttemptState.FAILED.value
            else:
                terminal_state = AttemptState.VERIFIED.value
            attempt_manifest.update(
                {
                    "warmups": warmups,
                    "samples": samples,
                    "host_samples": host_samples,
                    "host_gate": host_gate,
                    "work_counts": counts,
                    "cleanup": cleanup,
                    "errors": errors,
                }
            )
            attempt_manifest["state"] = transition_attempt(
                AttemptState.RUNNING,
                terminal_state,
            ).value
            await complete_owned(persist_attempt_manifest())
        except Exception as error:
            try:
                _, published, raw = read_json_artifact(
                    result_root, manifest_path, 'last attempt checkpoint',
                )
                for field in ('schema', 'run_id', 'round', 'competitor', 'attempt_number'):
                    if published[field] != attempt_manifest[field]:
                        raise ValueError('last attempt checkpoint identity changed')
                reference = _pair_attempt_reference(run_root, manifest_path, published)
                reference['manifest_sha256'] = preflight.sha256_bytes(raw)
            except (OSError, ValueError, KeyError, TypeError) as read_error:
                failure = (
                    f'{type(error).__name__}: {error}; last checkpoint unavailable: '
                    f'{type(read_error).__name__}: {read_error}'
                )
                reference = {
                    'attempt_number': attempt_number,
                    'state': AttemptState.FAILED.value,
                    'directory': attempt_dir.relative_to(run_root).as_posix(),
                    'manifest_sha256': None,
                    'cleanup_ok': False,
                    'runtime_root_removed': False,
                    'host_gate': {
                        'state': HostGateState.UNKNOWN.value,
                        'reasons': ['last attempt checkpoint is unavailable'],
                    },
                    'work_counts': {**zero_work_counts(), 'error_count': 1},
                    'error_count': 1,
                    'errors': [failure],
                }
            checkpoint_failure = AttemptCheckpointFailure(error, reference)
            if interruption is not None:
                raise interruption from checkpoint_failure
            raise checkpoint_failure from error
    if interruption is not None:
        raise interruption
    return _pair_attempt_reference(run_root, manifest_path, attempt_manifest)


def new_runtime_root() -> tuple[Path, str]:
    physical_tmp = Path('/tmp').resolve(strict=True)
    runtime_root = Path(tempfile.mkdtemp(prefix='nb-navigation-', dir=str(physical_tmp)))
    runtime_root.chmod(0o700)
    owner_token = uuid.uuid4().hex
    navigation.atomic_profile_bytes(runtime_root / '.benchmark-owner', owner_token.encode('ascii'))
    return runtime_root, owner_token


def create_pair_directories(run_root: Path, round_number: int, competitor: str, attempt_number: int) -> Path:
    round_root = create_private_directory(run_root / f'round-{round_number}')
    pair_root = create_private_directory(round_root / f'pair-{competitor}')
    attempt_root = pair_root / f'attempt-{attempt_number}'
    attempt_root.mkdir(mode=0o700, exist_ok=False)
    attempt_root.chmod(0o700)
    return navigation.validate_private_directory(attempt_root)


def _trial_request(run_id: str, round_number: int, competitor: str, attempt_number: int,
                   arm: str, workload: str, index: int, port: int) -> dict[str, Any]:
    nonce = f'{run_id}-r{round_number}-{competitor}-a{attempt_number}-{arm}-{workload}-{index}-{uuid.uuid4().hex}'
    query = urlencode({'nonce': nonce, 'delayed': int(workload == 'delayed')})
    return {
        'op': 'trial',
        'arm': arm,
        'kind': workload,
        'index': index,
        'nonce': nonce,
        'url': f'http://127.0.0.1:{port}/fixture?{query}',
    }


async def execute_trial(
    handle: navigation.WorkerHandle,
    fixture: navigation.Fixture,
    request: dict[str, Any],
    *,
    is_warmup: bool,
) -> dict[str, Any]:
    label = f'{request["arm"]}/{request["kind"]}/{request["index"]}'
    errors: list[str] = []
    raw_sample = None
    try:
        response = await navigation.send_operation(handle, request, timeout=40)
        if not isinstance(response, dict) or response.get('ok') is not True \
                or not isinstance(response.get('result'), dict):
            raise RuntimeError(f'worker returned an invalid trial reply: {response!r}')
        raw_sample = response['result']
    except Exception as error:
        errors.append(f'{label}: {type(error).__name__}: {error}')

    try:
        server = fixture.snapshot(request['nonce'])
    except Exception as error:
        server = {'requests': [], 'effects': 0}
        errors.append(f'{label}: fixture proof failed: {type(error).__name__}: {error}')

    if isinstance(raw_sample, dict):
        sample = {**raw_sample, 'server': server}
        for field in ('arm', 'kind', 'index', 'nonce', 'url'):
            if sample.get(field) != request.get(field):
                errors.append(f'{label}: worker sample {field} differs from its request')
        if sample.get('outcome') != 'success':
            errors.append(f'{label}: task outcome is {sample.get("outcome")!r}')
        errors.extend(navigation_phase_errors(sample, label))
        try:
            errors.extend(navigation.trial_proof_errors(sample, label))
        except Exception as error:
            errors.append(f'{label}: task proof raised {type(error).__name__}: {error}')
    else:
        sample = None

    return {
        'arm': request['arm'],
        'workload': request['kind'],
        'index': request['index'],
        'nonce': request['nonce'],
        'url': request['url'],
        'warmup': is_warmup,
        'state': AttemptState.VERIFIED.value if not errors else AttemptState.FAILED.value,
        'sample': sample,
        'request_proof': server,
        'errors': errors,
    }


async def cleanup_pair_workers(
    handles: dict[str, navigation.WorkerHandle],
    expected_arms: tuple[str, ...],
    runtime_root: Path,
    owner_token: str,
) -> dict[str, Any]:
    workers = {}
    errors = []
    owned_workers_clean = set(handles).issubset(expected_arms)
    workers_ok = len(handles) == len(expected_arms) and owned_workers_clean
    for arm, handle in reversed(list(handles.items())):
        try:
            cleanup = await navigation.cleanup_worker(handle)
        except Exception as error:
            cleanup = {
                "ok": False,
                "fallback_used": None,
                "worker_identity": {**handle.worker_identity, "state": "unknown"},
                "post_worker_exit_identities": [],
                "error": f"{type(error).__name__}: {error}",
            }
        workers[arm] = cleanup
        if cleanup.get('ok') is not True:
            owned_workers_clean = False
            workers_ok = False
            errors.append(f'{arm} worker cleanup is incomplete')

    states = []
    for cleanup in workers.values():
        states.append(cleanup.get('worker_identity', {}).get('state', 'unknown'))
        states.extend(
            identity.get('state', 'unknown')
            for identity in cleanup.get('post_worker_exit_identities', [])
            if isinstance(identity, dict)
        )
    runtime_root_removed = False
    if owned_workers_clean and navigation.removal_allowed(runtime_root, owner_token, states):
        try:
            shutil.rmtree(runtime_root)
            runtime_root_removed = not runtime_root.exists()
        except OSError as error:
            errors.append(f'cannot remove owned runtime root: {type(error).__name__}: {error}')
    if not runtime_root_removed:
        errors.append('owned runtime root was retained because cleanup could not be proved')
    return {
        'ok': workers_ok and runtime_root_removed,
        'workers_ok': workers_ok,
        'runtime_root': str(runtime_root),
        'runtime_root_removed': runtime_root_removed,
        'workers': workers,
        'errors': errors,
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    commands = root.add_subparsers(dest='command', required=True)
    smoke = commands.add_parser('smoke', help='run the two fixed navigation controls')
    run = commands.add_parser('run', help='run all four competitor pairs in two rounds')
    for command in (smoke, run):
        command.add_argument('--candidate-identity', required=True, type=Path)
        command.add_argument('--preflight', required=True, type=Path)
        command.add_argument('--version-manifest', required=True, type=Path)
        command.add_argument('--source', required=True, type=Path)
        command.add_argument('--python', required=True, type=Path)
        command.add_argument('--chrome', required=True, type=Path)
        command.add_argument('--output', required=True, type=Path)
    run.add_argument('--frozen-harness', required=True, type=Path)
    run.add_argument('--lightpanda', required=True, type=Path)
    run.add_argument('--agent-browser', required=True, type=Path)
    run.add_argument('--browser-use-python', required=True, type=Path)
    run.add_argument('--node', type=Path)
    run.add_argument('--rounds', type=int, default=2)
    run.add_argument('--samples', type=int, default=6)
    return root


class NavigationPreflightFailure(ValueError):
    pass


def dispatch(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parser().parse_args(argv)
    try:
        return asyncio.run(run_navigation(args))
    except NavigationPreflightFailure as error:
        LOG.error('%s', error)
        return 1
    except (preflight.PreflightError, ValueError, OSError) as error:
        LOG.error('%s', error)
        return 2


if __name__ == '__main__':
    raise SystemExit(dispatch())
