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


def test_in_memory_storage_state_is_explicitly_skipped():
    watchdog = _make_watchdog({'cookies': [], 'origins': []})

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
    watchdog.browser_session.cdp_client = None

    await watchdog._load_storage_state(path=str(state_path))

    snapshot = watchdog.storage_state_snapshot()
    assert snapshot.phase is StorageStatePhase.FAILED
    assert snapshot.error == 'No CDP client available for loading storage state'


@pytest.mark.asyncio
async def test_cancelled_load_has_a_terminal_phase(tmp_path):
    state_path = tmp_path / 'state.json'
    state_path.write_text('{"cookies": [], "origins": []}')
    watchdog = _make_watchdog(state_path)

    with (
        patch('anyio.Path.read_text', new_callable=AsyncMock, side_effect=asyncio.CancelledError),
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
    assert snapshot.origins_count == 1
    assert snapshot.detail == 'failed after applying 1 initialization scripts'
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
    assert watchdog.browser_session._cdp_add_init_script.await_count == snapshot.script_count
    assert watchdog.event_bus.dispatch.call_count == 1
