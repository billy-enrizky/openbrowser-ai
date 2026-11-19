"""Behavior tests for explicit storage-state load outcomes."""

import asyncio
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from openbrowser.browser.session import BrowserSession
from openbrowser.browser.storage_state import StorageStatePhase
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
    assert snapshot.path == str(state_path)
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
        assert snapshot.origins_count == 0
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
    watchdog.browser_session._cdp_add_init_script = AsyncMock(side_effect=[None, RuntimeError('apply failed')])

    await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.FAILED
    assert snapshot.script_count == 1
    assert snapshot.storage_script_attempts == 2
    assert snapshot.origins_count == 1
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

    await asyncio.wait_for(watchdog._load_storage_state(path=str(state_path)), timeout=10)

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.LOADED
    assert snapshot.path == str(state_path)
    assert snapshot.bytes is not None and snapshot.bytes > 40_000_000
    assert snapshot.load_ms is not None and snapshot.load_ms >= 0
    assert snapshot.elapsed_ms == snapshot.load_ms
    assert snapshot.cookies_count == 1
    assert snapshot.origins_count == 1
    assert snapshot.script_count > 1
    assert snapshot.storage_script_attempts == snapshot.script_count
    assert watchdog.browser_session._cdp_add_init_script.await_count == snapshot.script_count
    assert watchdog.event_bus.dispatch.call_count == 1
