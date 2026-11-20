"""Behavior tests for explicit storage-state load outcomes."""

import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from openbrowser.browser.events import StorageStateFailedEvent
from openbrowser.browser.session import BrowserSession
from openbrowser.browser.storage_state import StorageStatePhase, StorageStateSavePhase
from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog


def _make_watchdog(path):
    session = create_autospec(BrowserSession, instance=True)
    session.logger = logging.getLogger('test-storage-state-lifecycle')
    session.cdp_client = MagicMock()
    session._cdp_client_root = MagicMock()
    session._cdp_set_cookies = AsyncMock()
    session._cdp_add_init_script = AsyncMock()
    session._storage_state_warnings = []
    profile = MagicMock()
    profile.storage_state = path
    session.browser_profile = profile
    event_bus = MagicMock()
    return StorageStateWatchdog.model_construct(
        event_bus=event_bus,
        browser_session=session,
        auto_save_interval=30.0,
        save_on_change=True,
        max_script_bytes_per_origin=256_000,
    )


def test_storage_state_is_not_configured_without_a_source():
    watchdog = _make_watchdog(None)

    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.NOT_CONFIGURED
    assert snapshot.path is None


def test_storage_state_is_pending_before_load(tmp_path):
    state_path = tmp_path / 'state.json'
    state_path.write_text('{"cookies": [], "origins": []}')
    watchdog = _make_watchdog(state_path)

    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.PENDING
    assert snapshot.path == str(state_path.resolve())
    assert snapshot.bytes == state_path.stat().st_size


@pytest.mark.parametrize('state', [{}, {'cookies': [], 'origins': []}])
def test_in_memory_storage_state_is_explicitly_skipped(state):
    watchdog = _make_watchdog(state)

    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.SKIPPED
    assert snapshot.detail == 'in-memory storage state is not loaded from a file'


@pytest.mark.asyncio
async def test_missing_file_is_reported_as_skipped(tmp_path):
    state_path = tmp_path / 'missing.json'
    watchdog = _make_watchdog(state_path)

    assert watchdog.storage_state_snapshot().phase is StorageStatePhase.SKIPPED

    await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.SKIPPED
    assert snapshot.detail == 'storage-state file does not exist'


@pytest.mark.asyncio
async def test_malformed_file_is_reported_as_failed(tmp_path):
    state_path = tmp_path / 'broken.json'
    state_path.write_text('{not json')
    watchdog = _make_watchdog(state_path)

    await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.FAILED
    assert snapshot.error is not None
    assert 'JSONDecodeError' in snapshot.error
    assert snapshot.load_ms is not None
    assert watchdog.browser_session._storage_state_warnings


@pytest.mark.asyncio
async def test_missing_cdp_client_is_reported_as_failed(tmp_path):
    state_path = tmp_path / 'state.json'
    state_path.write_text('{"cookies": [], "origins": []}')
    watchdog = _make_watchdog(state_path)
    watchdog.browser_session._cdp_client_root = None

    await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.FAILED
    assert snapshot.error == 'Browser is not connected; no CDP client is available'


@pytest.mark.asyncio
async def test_non_object_json_is_reported_as_failed(tmp_path):
    state_path = tmp_path / 'state.json'
    state_path.write_text('[]')
    watchdog = _make_watchdog(state_path)

    await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.FAILED
    assert snapshot.error is not None
    assert 'storage state must be a JSON object' in snapshot.error


@pytest.mark.asyncio
async def test_unresolved_home_path_is_reported_as_failed(tmp_path):
    watchdog = _make_watchdog('~openbrowser-user-that-does-not-exist/state.json')

    with patch('pathlib.Path.expanduser', side_effect=RuntimeError('home unavailable')):
        await watchdog._load_storage_state()

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.FAILED
    assert snapshot.error is not None
    assert 'RuntimeError' in snapshot.error


@pytest.mark.asyncio
async def test_snapshot_bytes_come_from_the_bytes_that_were_parsed(tmp_path):
    state_path = tmp_path / 'state.json'
    # The on-disk file is deliberately longer than the bytes the parser sees,
    # so a stat-based size would not match the parsed snapshot.
    state_path.write_text('{\n  "cookies": [],\n  "origins": []\n}\n')
    watchdog = _make_watchdog(state_path)
    parsed = b'{"cookies":[],"origins":[]}'
    assert len(parsed) != state_path.stat().st_size

    with patch('anyio.Path.read_bytes', new_callable=AsyncMock, return_value=parsed):
        await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.bytes == len(parsed)


@pytest.mark.asyncio
async def test_cancelled_load_has_a_terminal_phase(tmp_path):
    state_path = tmp_path / 'state.json'
    state_path.write_text('{"cookies": [], "origins": []}')
    watchdog = _make_watchdog(state_path)

    with (
        patch('anyio.Path.read_bytes', new_callable=AsyncMock, side_effect=asyncio.CancelledError),
        pytest.raises(asyncio.CancelledError),
    ):
        await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.CANCELLED
    assert snapshot.error == 'storage-state load cancelled'


def test_loading_snapshot_reports_elapsed_time(tmp_path):
    state_path = tmp_path / 'state.json'
    state_path.write_text('{"cookies": [], "origins": []}')
    watchdog = _make_watchdog(state_path)
    watchdog._load_phase = StorageStatePhase.LOADING
    watchdog._load_path = str(state_path)
    watchdog._load_started = time.monotonic() - 0.05

    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADING
    assert snapshot.elapsed_ms is not None and snapshot.elapsed_ms >= 40
    assert snapshot.load_ms is None


@pytest.mark.asyncio
async def test_loading_snapshot_publishes_partial_script_progress(tmp_path):
    state_path = tmp_path / 'state.json'
    state = {
        'cookies': [],
        'origins': [
            {'origin': 'https://example.com', 'localStorage': [{'name': 'key', 'value': 'value'}]},
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def add_script(_script):
        entered.set()
        await release.wait()

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    load_task = asyncio.create_task(watchdog._load_storage_state(path=str(state_path)))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        snapshot = watchdog.storage_state_snapshot()
        assert snapshot.phase is StorageStatePhase.LOADING
        assert snapshot.storage_script_attempts == 1
        assert snapshot.script_count == 0
        assert snapshot.origins_count == 1
    finally:
        release.set()
        await asyncio.wait_for(load_task, timeout=1)

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.storage_script_attempts == 1
    assert snapshot.script_count == 1
    assert snapshot.origins_count == 1


@pytest.mark.asyncio
async def test_partial_application_is_reported_on_failure(tmp_path):
    state_path = tmp_path / 'state.json'
    state = {
        'cookies': [],
        'origins': [
            {'origin': 'https://one.example', 'localStorage': [{'name': 'a', 'value': '1'}]},
            {'origin': 'https://two.example', 'localStorage': [{'name': 'b', 'value': '2'}]},
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    watchdog.max_script_bytes_per_origin = 240
    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=[None, RuntimeError('apply failed')])

    await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.FAILED
    assert snapshot.script_count == 1
    assert snapshot.storage_script_attempts == 2
    assert snapshot.origins_count == 2
    assert snapshot.detail == 'failed after attempting 2 initialization-script registration(s); 1 confirmed'
    assert snapshot.error is not None and 'apply failed' in snapshot.error


@pytest.mark.asyncio
async def test_large_storage_state_reports_counts_and_batched_scripts(tmp_path):
    state_path = tmp_path / 'large-state.json'
    items = [
        {'name': f'key-{index}', 'value': 'x' * 8_000}
        for index in range(5_511)
    ]
    state = {
        'cookies': [{'name': 'session', 'domain': '.example.com', 'path': '/', 'value': '1'}],
        'origins': [{'origin': 'https://example.com', 'localStorage': items}],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)

    await asyncio.wait_for(watchdog._load_storage_state(path=str(state_path)), timeout=10)

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.path == str(state_path.resolve())
    assert snapshot.bytes is not None and snapshot.bytes > 40_000_000
    assert snapshot.load_ms is not None and snapshot.load_ms >= 0
    assert snapshot.elapsed_ms == snapshot.load_ms
    assert snapshot.cookies_count == 1
    assert snapshot.origins_count == 1
    assert snapshot.script_count <= 200
    assert snapshot.script_count == len(scripts)
    assert all(len(script.encode('utf-8')) <= 256_000 for script in scripts)
    assert snapshot.storage_script_attempts == snapshot.script_count
    assert watchdog.browser_session._cdp_add_init_script.await_count == snapshot.script_count
    combined = '\n'.join(scripts)
    emitted_keys = set(re.findall(r'"key-(\d+)"', combined))
    assert emitted_keys == {str(index) for index in range(5_511)}
    assert watchdog.event_bus.dispatch.call_count == 1


def _dispatched_events(event_bus, event_type):
    return [call.args[0] for call in event_bus.dispatch.call_args_list if call.args and isinstance(call.args[0], event_type)]


@pytest.mark.asyncio
async def test_many_origins_batch_into_bounded_scripts(tmp_path):
    state_path = tmp_path / 'many-origins.json'
    state = {
        'cookies': [],
        'origins': [
            {
                'origin': f'https://origin-{index}.example',
                'localStorage': [{'name': f'key-{index}', 'value': 'x' * 64}],
            }
            for index in range(5_511)
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)
        await asyncio.sleep(0.001)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    await asyncio.wait_for(watchdog._load_storage_state(path=str(state_path)), timeout=10)
    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.origins_count == 5_511
    assert scripts
    assert snapshot.script_count <= 12
    assert snapshot.script_count == len(scripts)
    assert all(len(script.encode('utf-8')) <= 256_000 for script in scripts)
    combined = '\n'.join(scripts)
    emitted_origins = set(re.findall(r'window\.location\.origin==="https://origin-(\d+)\.example"', combined))
    emitted_keys = set(re.findall(r'"key-(\d+)"', combined))
    assert emitted_origins == {str(index) for index in range(5_511)}
    assert emitted_keys == {str(index) for index in range(5_511)}


@pytest.mark.asyncio
async def test_oversized_single_entry_is_skipped_with_a_warning(tmp_path):
    state_path = tmp_path / 'oversized-entry.json'
    state = {
        'cookies': [],
        'origins': [
            {
                'origin': 'https://example.com',
                'localStorage': [
                    {'name': 'huge', 'value': 'x' * 300_000},
                    {'name': 'small', 'value': 'kept'},
                ],
            }
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    await watchdog._load_storage_state(path=str(state_path))
    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.skipped_items == 1
    assert snapshot.script_count == 1
    assert all(len(script.encode('utf-8')) <= 256_000 for script in scripts)
    assert 'kept' in scripts[0]
    assert 'huge' not in scripts[0]
    assert any('huge' in warning and 'would emit' in warning and '256000' in warning for warning in watchdog.browser_session._storage_state_warnings)


@pytest.mark.asyncio
async def test_empty_origin_is_skipped_and_never_injected_globally(tmp_path):
    state_path = tmp_path / 'empty-origin.json'
    state = {
        'cookies': [],
        'origins': [
            {'origin': '', 'localStorage': [{'name': 'leak', 'value': 'secret-value'}]},
            {'origin': 'https://safe.example', 'localStorage': [{'name': 'safe', 'value': 'ok'}]},
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    await watchdog._load_storage_state(path=str(state_path))
    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.skipped_origins == 1
    assert snapshot.skipped_items == 1
    assert snapshot.script_count == 1
    assert all('secret-value' not in script for script in scripts)
    assert any('unsafe origin' in warning.lower() for warning in watchdog.browser_session._storage_state_warnings)


@pytest.mark.asyncio
async def test_null_and_about_blank_origins_are_skipped(tmp_path):
    state_path = tmp_path / 'unsafe-origins.json'
    state = {
        'cookies': [],
        'origins': [
            {'origin': 'null', 'localStorage': [{'name': 'opaque', 'value': 'secret-null'}]},
            {'origin': 'about:blank', 'localStorage': [{'name': 'blank', 'value': 'secret-blank'}]},
            {'origin': 'https://safe.example', 'localStorage': [{'name': 'safe', 'value': 'ok'}]},
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    await watchdog._load_storage_state(path=str(state_path))
    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.skipped_origins == 2
    assert snapshot.skipped_items == 2
    assert all('secret-null' not in script and 'secret-blank' not in script for script in scripts)
    assert any('unsafe origin' in warning for warning in watchdog.browser_session._storage_state_warnings)


@pytest.mark.asyncio
async def test_non_mapping_origin_entry_is_skipped_without_failing_the_load(tmp_path):
    state_path = tmp_path / 'bad-origin-entry.json'
    state = {
        'cookies': [],
        'origins': [
            'not-an-object',
            {'origin': 'https://safe.example', 'localStorage': [{'name': 'safe', 'value': 'ok'}]},
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    await watchdog._load_storage_state(path=str(state_path))
    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.skipped_origins == 1
    assert scripts
    assert 'ok' in scripts[0]


@pytest.mark.asyncio
async def test_malformed_origin_urls_are_skipped(tmp_path):
    state_path = tmp_path / 'malformed-origins.json'
    state = {
        'cookies': [],
        'origins': [
            {'origin': 'https://', 'localStorage': [{'name': 'a', 'value': '1'}]},
            {'origin': 'https://example.com/path', 'localStorage': [{'name': 'b', 'value': '2'}]},
            {'origin': 'https://example.com?q=1', 'localStorage': [{'name': 'c', 'value': '3'}]},
            {'origin': 'https://user:pass@example.com', 'localStorage': [{'name': 'd', 'value': '4'}]},
            {'origin': 'https://safe.example', 'localStorage': [{'name': 'safe', 'value': 'ok'}]},
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    await watchdog._load_storage_state(path=str(state_path))
    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.skipped_origins == 4
    assert scripts
    assert 'safe' in scripts[0]


@pytest.mark.asyncio
async def test_lone_surrogate_uses_ascii_escaped_javascript(tmp_path):
    state_path = tmp_path / 'surrogate.json'
    state = {
        'cookies': [],
        'origins': [
            {'origin': 'https://example.com', 'localStorage': [{'name': 'surrogate', 'value': '\ud800'}]},
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)
    scripts = []

    async def add_script(script):
        scripts.append(script)

    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=add_script)
    await watchdog._load_storage_state(path=str(state_path))
    snapshot = watchdog.storage_state_snapshot()

    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.script_count == 1
    assert '\\ud800' in scripts[0]


@pytest.mark.asyncio
async def test_failed_load_emits_one_failure_event_for_each_preflight_path(tmp_path):
    cases = []

    missing_cdp = _make_watchdog(tmp_path / 'state.json')
    (tmp_path / 'state.json').write_text('{"cookies": [], "origins": []}')
    missing_cdp.browser_session._cdp_client_root = None
    cases.append((missing_cdp, 'Browser is not connected'))

    malformed = _make_watchdog(tmp_path / 'broken.json')
    (tmp_path / 'broken.json').write_text('{not json')
    cases.append((malformed, 'JSONDecodeError'))

    for watchdog, expected_error in cases:
        await watchdog._load_storage_state()
        failures = _dispatched_events(watchdog.event_bus, StorageStateFailedEvent)
        assert len(failures) == 1
        assert failures[0].operation == 'load'
        assert failures[0].phase == 'failed'
        assert expected_error in (failures[0].error or '')


def _make_save_watchdog(tmp_path, *, cookies=None, origins=None):
    state_path = tmp_path / 'state.json'
    watchdog = _make_watchdog(state_path)
    watchdog.browser_session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock())
    watchdog.browser_session._cdp_get_storage_state = AsyncMock(
        return_value={'cookies': cookies or [], 'origins': origins or []}
    )
    return watchdog, state_path


@pytest.mark.asyncio
async def test_save_success_records_phase_backup_and_event(tmp_path):
    from openbrowser.browser.events import StorageStateSavedEvent

    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    old_state = {
        'cookies': [{'name': 'old', 'domain': '.example.com', 'path': '/', 'value': '1'}],
        'origins': [],
    }
    state_path.write_text(json.dumps(old_state))

    await watchdog._save_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_save_snapshot()
    final = json.loads(state_path.read_text())
    backup = json.loads(state_path.with_suffix('.json.bak').read_text())
    events = _dispatched_events(watchdog.event_bus, StorageStateSavedEvent)

    assert snapshot.phase is StorageStateSavePhase.SAVED
    assert snapshot.path == str(state_path.resolve())
    assert snapshot.bytes is not None and snapshot.bytes > 0
    assert snapshot.save_ms is not None
    assert snapshot.cookies_count == 2
    assert {cookie['name'] for cookie in final['cookies']} == {'old', 'new'}
    assert [cookie['name'] for cookie in backup['cookies']] == ['old']
    assert len(events) == 1
    assert events[0].bytes == snapshot.bytes
    assert events[0].save_ms == snapshot.save_ms
    assert watchdog._last_cookie_state == [
        {'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}
    ]


@pytest.mark.asyncio
async def test_save_replace_failure_preserves_the_old_file(tmp_path):
    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    original = '{"cookies": [], "origins": []}'
    state_path.write_text(original)

    with patch('openbrowser.browser.watchdogs.storage_state_watchdog.os.replace', side_effect=OSError('replace failed')):
        await watchdog._save_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_save_snapshot()
    failures = _dispatched_events(watchdog.event_bus, StorageStateFailedEvent)

    assert snapshot.phase is StorageStateSavePhase.FAILED
    assert 'replace failed' in (snapshot.error or '')
    assert state_path.read_text() == original
    assert not list(state_path.parent.glob('*.tmp'))
    assert len(failures) == 1
    assert failures[0].operation == 'save'


@pytest.mark.asyncio
async def test_save_malformed_existing_file_fails_closed(tmp_path):
    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    state_path.write_text('not json')

    await watchdog._save_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_save_snapshot()
    failures = _dispatched_events(watchdog.event_bus, StorageStateFailedEvent)
    assert snapshot.phase is StorageStateSavePhase.FAILED
    assert 'Could not read existing storage state' in (snapshot.error or '')
    assert state_path.read_text() == 'not json'
    assert len(failures) == 1
    assert failures[0].operation == 'save'


@pytest.mark.asyncio
async def test_save_preview_phases_follow_the_configured_source(tmp_path):
    not_configured = _make_watchdog(None)
    assert not_configured.storage_state_save_snapshot().phase is StorageStateSavePhase.NOT_CONFIGURED

    in_memory = _make_watchdog({'cookies': [], 'origins': []})
    assert in_memory.storage_state_save_snapshot().phase is StorageStateSavePhase.SKIPPED

    file_backed = _make_watchdog(tmp_path / 'state.json')
    preview = file_backed.storage_state_save_snapshot()
    assert preview.phase is StorageStateSavePhase.IDLE
    assert preview.path == str((tmp_path / 'state.json').resolve())


@pytest.mark.asyncio
async def test_save_uses_a_thread_for_file_work(tmp_path):
    import threading

    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    observed_threads = []
    original_write_temp = watchdog._write_temp

    def record_thread(temp_path, data):
        observed_threads.append(threading.current_thread().name)
        return original_write_temp(temp_path, data)

    with patch.object(watchdog, '_write_temp', side_effect=record_thread):
        await watchdog._save_storage_state(path=str(state_path))

    assert observed_threads
    assert observed_threads[0] != threading.main_thread().name
    assert watchdog.storage_state_save_snapshot().phase is StorageStateSavePhase.SAVED


@pytest.mark.asyncio
async def test_pre_gate_cancellation_preserves_the_old_file(tmp_path):
    import threading

    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    original = '{"cookies": [], "origins": []}'
    state_path.write_text(original)
    started = threading.Event()
    release = threading.Event()
    original_write_temp = watchdog._write_temp

    def blocked_write_temp(temp_path, data):
        path = original_write_temp(temp_path, data)
        started.set()
        release.wait(2)
        return path

    with patch.object(watchdog, '_write_temp', side_effect=blocked_write_temp):
        task = asyncio.create_task(watchdog._save_storage_state(path=str(state_path)))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    snapshot = watchdog.storage_state_save_snapshot()
    assert snapshot.phase is StorageStateSavePhase.CANCELLED
    assert state_path.read_text() == original
    assert not list(state_path.parent.glob('*.tmp'))
    assert watchdog._last_cookie_state == []


@pytest.mark.asyncio
async def test_post_gate_backup_cancellation_reports_saved(tmp_path):
    import threading

    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    state_path.write_text('{"cookies": [], "origins": []}')
    started = threading.Event()
    release = threading.Event()
    original_backup = watchdog._backup_existing

    def blocked_backup(path, backup_path):
        started.set()
        release.wait(2)
        return original_backup(path, backup_path)

    with patch.object(watchdog, '_backup_existing', side_effect=blocked_backup):
        task = asyncio.create_task(watchdog._save_storage_state(path=str(state_path)))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    snapshot = watchdog.storage_state_save_snapshot()
    assert snapshot.phase is StorageStateSavePhase.SAVED
    assert json.loads(state_path.read_text())['cookies'][0]['name'] == 'new'


@pytest.mark.asyncio
async def test_two_watchdogs_on_one_path_merge_both_cookie_sets(tmp_path):
    state_path = tmp_path / 'state.json'
    first = _make_watchdog(state_path)
    second = _make_watchdog(state_path)
    for watchdog, name in ((first, 'a'), (second, 'b')):
        watchdog.browser_session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock())
        watchdog.browser_session._cdp_get_storage_state = AsyncMock(
            return_value={
                'cookies': [{'name': name, 'domain': '.example.com', 'path': '/', 'value': name}],
                'origins': [],
            }
        )

    await asyncio.gather(
        first._save_storage_state(path=str(state_path)),
        second._save_storage_state(path=str(state_path)),
    )

    final = json.loads(state_path.read_text())
    assert {cookie['name'] for cookie in final['cookies']} == {'a', 'b'}
    assert first.storage_state_save_snapshot().phase is StorageStateSavePhase.SAVED
    assert second.storage_state_save_snapshot().phase is StorageStateSavePhase.SAVED


@pytest.mark.asyncio
async def test_save_lock_timeout_fails_without_changing_the_file(tmp_path):
    import filelock

    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    original = '{"cookies": [], "origins": []}'
    state_path.write_text(original)

    class TimedOutLock:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise filelock.Timeout('lock timed out')

        def __exit__(self, *args):
            return False

    with patch('openbrowser.browser.watchdogs.storage_state_watchdog.FileLock', TimedOutLock):
        await watchdog._save_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_save_snapshot()
    assert snapshot.phase is StorageStateSavePhase.FAILED
    assert 'lock timed out' in (snapshot.error or '')
    assert state_path.read_text() == original


@pytest.mark.asyncio
async def test_failed_save_keeps_the_cookie_retry_path_open(tmp_path):
    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    state_path.write_text('{"cookies": [], "origins": []}')
    watchdog._last_cookie_state = [
        {'name': 'old', 'domain': '.example.com', 'path': '/', 'value': '1'}
    ]

    with patch('openbrowser.browser.watchdogs.storage_state_watchdog.os.replace', side_effect=OSError('replace failed')):
        await watchdog._save_storage_state(path=str(state_path))

    assert watchdog._last_cookie_state == [
        {'name': 'old', 'domain': '.example.com', 'path': '/', 'value': '1'}
    ]
    watchdog.browser_session._cdp_get_cookies = AsyncMock(
        return_value=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}]
    )
    assert await watchdog._have_cookies_changed() is True


@pytest.mark.asyncio
async def test_save_uses_a_resolved_path_lock(tmp_path):
    real_path = tmp_path / 'real-state.json'
    alias_path = tmp_path / 'alias.json'
    try:
        alias_path.symlink_to(real_path)
    except OSError:
        pytest.skip('symlinks are unavailable')

    watchdog = _make_watchdog(alias_path)
    watchdog.browser_session.get_or_create_cdp_session = AsyncMock(return_value=MagicMock())
    watchdog.browser_session._cdp_get_storage_state = AsyncMock(
        return_value={
            'cookies': [{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
            'origins': [],
        }
    )
    observed = {}

    class RecordingLock:
        def __init__(self, path, timeout):
            observed['path'] = path
            observed['timeout'] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch('openbrowser.browser.watchdogs.storage_state_watchdog.FileLock', RecordingLock):
        await watchdog._save_storage_state(path=str(alias_path))

    assert observed['path'].endswith('real-state.json.lock')
    assert observed['timeout'] == 5.0
    assert watchdog.storage_state_save_snapshot().phase is StorageStateSavePhase.SAVED


@pytest.mark.asyncio
async def test_save_rejects_tuple_cookies_before_commit(tmp_path):
    watchdog, state_path = _make_save_watchdog(tmp_path)
    original = '{"cookies": [], "origins": []}'
    state_path.write_text(original)
    cookie = {'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}
    watchdog.browser_session._cdp_get_storage_state = AsyncMock(
        return_value={'cookies': (cookie,), 'origins': []}
    )

    await watchdog._save_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_save_snapshot()
    assert snapshot.phase is StorageStateSavePhase.FAILED
    assert 'cookies must be a list' in (snapshot.error or '')
    assert state_path.read_text() == original
    assert not list(state_path.parent.glob('*.tmp'))


@pytest.mark.asyncio
async def test_save_path_resolution_failure_keeps_the_requested_path(tmp_path):
    watchdog, state_path = _make_save_watchdog(tmp_path)

    with patch('pathlib.Path.resolve', side_effect=RuntimeError('symlink loop')):
        await watchdog._save_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_save_snapshot()
    failures = _dispatched_events(watchdog.event_bus, StorageStateFailedEvent)
    assert snapshot.phase is StorageStateSavePhase.FAILED
    assert snapshot.path == str(state_path.resolve())
    assert len(failures) == 1
    assert failures[0].path == str(state_path)


@pytest.mark.asyncio
async def test_post_gate_replacement_failure_with_cancellation_reports_failed(tmp_path):
    import threading

    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    original = '{"cookies": [], "origins": []}'
    state_path.write_text(original)
    started = threading.Event()
    release = threading.Event()
    original_backup = watchdog._backup_existing

    def blocked_backup(path, backup_path):
        started.set()
        release.wait(2)
        return original_backup(path, backup_path)

    with (
        patch.object(watchdog, '_backup_existing', side_effect=blocked_backup),
        patch('openbrowser.browser.watchdogs.storage_state_watchdog.os.replace', side_effect=OSError('replace failed')),
    ):
        task = asyncio.create_task(watchdog._save_storage_state(path=str(state_path)))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0.05)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    snapshot = watchdog.storage_state_save_snapshot()
    failures = _dispatched_events(watchdog.event_bus, StorageStateFailedEvent)
    assert snapshot.phase is StorageStateSavePhase.FAILED
    assert state_path.read_text() == original
    assert not list(state_path.parent.glob('*.tmp'))
    assert len(failures) == 1


@pytest.mark.asyncio
async def test_cross_process_saves_merge_both_cookie_sets(tmp_path):
    state_path = tmp_path / 'cross-process.json'
    start_at = time.time() + 0.6
    script = """
import sys, threading, time
from pathlib import Path
from openbrowser.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog
path = Path(sys.argv[1])
name = sys.argv[2]
start_at = float(sys.argv[3])
w = StorageStateWatchdog.model_construct(save_lock_timeout=5.0)
original_write_temp = w._write_temp
def slow_write_temp(temp_path, data):
    time.sleep(0.2)
    return original_write_temp(temp_path, data)
StorageStateWatchdog._write_temp = staticmethod(slow_write_temp)
while time.time() < start_at:
    time.sleep(0.005)
w._write_storage_state(
    path,
    {'cookies': [{'name': name, 'domain': '.example.com', 'path': '/', 'value': name}], 'origins': []},
    threading.Event(),
)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, '-c', script, str(state_path), name, str(start_at)],
            cwd=Path(__file__).resolve().parents[1],
        )
        for name in ('a', 'b')
    ]
    for process in processes:
        assert process.wait(timeout=15) == 0

    final = json.loads(state_path.read_text())
    assert {cookie['name'] for cookie in final['cookies']} == {'a', 'b'}


@pytest.mark.asyncio
async def test_save_without_a_cdp_session_fails_explicitly(tmp_path):
    watchdog, state_path = _make_save_watchdog(tmp_path)
    watchdog.browser_session.get_or_create_cdp_session = AsyncMock(return_value=None)

    await watchdog._save_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_save_snapshot()
    assert snapshot.phase is StorageStateSavePhase.FAILED
    assert 'No CDP session available' in (snapshot.error or '')


@pytest.mark.asyncio
async def test_unsafe_origin_warnings_are_bounded(tmp_path, caplog):
    state_path = tmp_path / 'many-unsafe-origins.json'
    state = {
        'cookies': [],
        'origins': [
            {'origin': 'null', 'localStorage': [{'name': f'key-{index}', 'value': 'secret'}]}
            for index in range(25)
        ],
    }
    state_path.write_text(json.dumps(state))
    watchdog = _make_watchdog(state_path)

    with caplog.at_level(logging.WARNING):
        await watchdog._load_storage_state(path=str(state_path))

    warnings = watchdog.browser_session._storage_state_warnings
    messages = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 20
    assert sum('unsafe origin' in message for message in messages) == 20
    assert any('additional storage-state warnings suppressed' in message for message in messages)


@pytest.mark.asyncio
async def test_not_configured_save_clears_the_previous_byte_count(tmp_path):
    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    await watchdog._save_storage_state(path=str(state_path))
    assert watchdog.storage_state_save_snapshot().bytes is not None

    watchdog.browser_session.browser_profile.storage_state = None
    await watchdog._save_storage_state(path=None)

    snapshot = watchdog.storage_state_save_snapshot()
    assert snapshot.phase is StorageStateSavePhase.NOT_CONFIGURED
    assert snapshot.bytes is None


@pytest.mark.asyncio
async def test_cancellation_while_waiting_for_the_path_lock_reports_cancelled(tmp_path):
    watchdog, state_path = _make_save_watchdog(
        tmp_path,
        cookies=[{'name': 'new', 'domain': '.example.com', 'path': '/', 'value': '2'}],
    )
    original = '{"cookies": [], "origins": []}'
    state_path.write_text(original)
    path_lock = asyncio.Lock()
    await path_lock.acquire()

    with patch('openbrowser.browser.watchdogs.storage_state_watchdog._path_save_lock', return_value=path_lock):
        task = asyncio.create_task(watchdog._save_storage_state(path=str(state_path)))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    path_lock.release()
    snapshot = watchdog.storage_state_save_snapshot()
    assert snapshot.phase is StorageStateSavePhase.CANCELLED
    assert state_path.read_text() == original


def test_path_lock_registry_does_not_retain_idle_locks(tmp_path):
    import gc
    import weakref

    from openbrowser.browser.watchdogs import storage_state_watchdog as module

    lock = module._path_save_lock(tmp_path / 'state.json')
    reference = weakref.ref(lock)
    del lock
    gc.collect()

    assert reference() is None
