"""Regression tests for ownership-safe LocalBrowserWatchdog cleanup."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import psutil

from openbrowser.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
from openbrowser.browser.watchdogs.profile_lease import ProfileInUseError, ProfileLease


def _browser_record(profile_dir: Path, *, pid: int = 123, start_time: float = 20.0) -> dict[str, object]:
	return {
		'pid': pid,
		'start_time': start_time,
		'profile_dir': str(profile_dir.resolve()),
		'instance_id': 'instance-a',
		'ownership_marker': '--openbrowser-instance-id=instance-a',
		'cdp_port': 9223,
	}


@pytest.mark.asyncio
async def test_matching_chrome_without_recorded_metadata_is_never_killed(tmp_path: Path):
	process = MagicMock()
	process.pid = 123
	process.info = {
		'name': 'chrome',
		'cmdline': ['chrome', f'--user-data-dir={tmp_path.resolve()}'],
	}

	with patch('psutil.process_iter', return_value=[process]):
		result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(str(tmp_path))

	assert result is False
	process.kill.assert_not_called()
	process.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_recorded_browser_with_reused_pid_is_never_killed(tmp_path: Path):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 99.0
	process.info = {'name': 'chrome'}
	process.cmdline.return_value = [
		'chrome',
		f'--user-data-dir={tmp_path.resolve()}',
		'--openbrowser-instance-id=instance-a',
	]

	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(tmp_path.resolve()),
		'browser': _browser_record(tmp_path, pid=123, start_time=20.0),
	}

	with (
		patch('psutil.process_iter', return_value=[process]),
		patch('psutil.Process', return_value=process),
	):
		result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(str(tmp_path), metadata=metadata)

	assert result is False
	process.kill.assert_not_called()
	process.terminate.assert_not_called()


@pytest.mark.asyncio
async def test_malformed_browser_metadata_without_pid_fails_closed(tmp_path: Path):
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(tmp_path.resolve()),
		'browser': {
			'start_time': 20.0,
			'profile_dir': str(tmp_path.resolve()),
			'instance_id': 'instance-a',
			'ownership_marker': '--openbrowser-instance-id=instance-a',
		},
	}

	with patch.object(ProfileLease, 'process_identity_status', return_value='missing'):
		assert await LocalBrowserWatchdog._kill_stale_chrome_for_profile(
			str(tmp_path), metadata=metadata, instance_id='instance-b'
		) is False


@pytest.mark.asyncio
async def test_metadata_cannot_redirect_cleanup_to_another_instance_or_profile(tmp_path: Path):
	target_profile = tmp_path / 'target-profile'
	other_profile = tmp_path / 'other-profile'
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.info = {'name': 'chrome'}
	process.cmdline.return_value = [
		'chrome',
		f'--user-data-dir={other_profile.resolve()}',
		'--openbrowser-instance-id=instance-b',
	]
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(target_profile.resolve()),
		'browser': _browser_record(other_profile, pid=123, start_time=20.0)
		| {'instance_id': 'instance-b', 'ownership_marker': '--openbrowser-instance-id=instance-b'},
	}

	with (
		patch('psutil.process_iter', return_value=[process]),
		patch.object(LocalBrowserWatchdog, '_cleanup_process', new_callable=AsyncMock) as cleanup,
	):
		result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(
			str(target_profile), metadata=metadata, instance_id='instance-c'
		)

	assert result is False
	cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_posix_cleanup_terminates_then_force_kills_only_after_grace(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.is_running.side_effect = [True] * 50 + [True, False]
	record = _browser_record(Path('/tmp/profile'), pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	with patch('asyncio.sleep', new_callable=AsyncMock) as sleep:
		result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is True
	assert process.method_calls.index(call.terminate()) < process.method_calls.index(call.kill())
	assert sleep.await_count >= 50


@pytest.mark.asyncio
async def test_posix_cleanup_reports_failure_after_post_kill_recheck(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.is_running.return_value = True
	record = _browser_record(Path('/tmp/profile'), pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	with patch('asyncio.sleep', new_callable=AsyncMock):
		result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is False
	process.terminate.assert_called_once()
	process.kill.assert_called_once()
	assert process.is_running.call_count >= 52


@pytest.mark.asyncio
async def test_windows_cleanup_uses_cdp_and_never_force_kills(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.is_running.side_effect = [True, False]
	record = _browser_record(Path('/tmp/profile'), pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		True,
	)
	with patch.object(LocalBrowserWatchdog, '_close_browser_via_cdp', new_callable=AsyncMock) as close_browser:
		close_browser.return_value = True
		with patch('asyncio.sleep', new_callable=AsyncMock):
			result = await LocalBrowserWatchdog._cleanup_process(
				process,
				browser_record=record,
				cdp_url='http://127.0.0.1:9223/',
			)

	assert result is True
	close_browser.assert_awaited_once_with(
		'http://127.0.0.1:9223/', expected_pid=123, expected_start_time=20.0
	)
	process.terminate.assert_not_called()
	process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_windows_cleanup_failure_never_falls_back_to_force_kill(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	record = _browser_record(Path('/tmp/profile'), pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		True,
	)
	with patch.object(LocalBrowserWatchdog, '_close_browser_via_cdp', new_callable=AsyncMock) as close_browser:
		close_browser.return_value = False
		result = await LocalBrowserWatchdog._cleanup_process(
			process,
			browser_record=record,
			cdp_url='http://127.0.0.1:9223/',
		)

	assert result is False
	close_browser.assert_awaited_once()
	process.terminate.assert_not_called()
	process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_can_fail_closed_when_identity_is_unavailable(monkeypatch):
	process = MagicMock()
	process.pid = 123

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	result = await LocalBrowserWatchdog._cleanup_process(process, require_identity=True)

	assert result is False
	process.terminate.assert_not_called()
	process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_requires_identity_by_default(monkeypatch):
	process = MagicMock()
	process.pid = 123

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	result = await LocalBrowserWatchdog._cleanup_process(process)

	assert result is False
	process.terminate.assert_not_called()
	process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_treats_missing_recorded_process_as_already_stopped(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.side_effect = psutil.NoSuchProcess(pid=123)
	process.is_running.side_effect = psutil.NoSuchProcess(pid=123)
	record = _browser_record(Path('/tmp/profile'), pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record, require_identity=True)

	assert result is True
	process.terminate.assert_not_called()
	process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_windows_cleanup_uses_recorded_cdp_endpoint(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.is_running.side_effect = [True, False]
	record = _browser_record(Path('/tmp/profile'), pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		True,
	)
	with patch.object(LocalBrowserWatchdog, '_close_browser_via_cdp', new_callable=AsyncMock) as close_browser:
		close_browser.return_value = True
		result = await LocalBrowserWatchdog._cleanup_process(
			process,
			browser_record=record,
			cdp_url='http://127.0.0.1:9999/',
		)

	assert result is True
	close_browser.assert_awaited_once_with(
		'http://127.0.0.1:9223/', expected_pid=123, expected_start_time=20.0
	)


@pytest.mark.asyncio
async def test_windows_cdp_cleanup_rejects_port_reuse(monkeypatch):
	version_response = MagicMock()
	version_response.json.return_value = {'webSocketDebuggerUrl': 'ws://127.0.0.1:9223/devtools/browser/test'}
	http_client = MagicMock()
	http_client.get = AsyncMock(return_value=version_response)
	http_context = MagicMock()
	http_context.__aenter__ = AsyncMock(return_value=http_client)
	http_context.__aexit__ = AsyncMock(return_value=False)

	websocket = MagicMock()
	websocket.send = AsyncMock()
	websocket.recv = AsyncMock(
		return_value=json.dumps({'id': 1, 'result': {'processInfo': {'id': 999}}})
	)
	websocket_context = MagicMock()
	websocket_context.__aenter__ = AsyncMock(return_value=websocket)
	websocket_context.__aexit__ = AsyncMock(return_value=False)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		True,
	)
	with (
		patch('httpx.AsyncClient', return_value=http_context),
		patch('websockets.connect', return_value=websocket_context),
	):
		result = await LocalBrowserWatchdog._close_browser_via_cdp(
			'http://127.0.0.1:9223/', expected_pid=123
		)

	assert result is False
	websocket.send.assert_awaited_once_with(json.dumps({'id': 1, 'method': 'SystemInfo.getProcessInfo'}))


@pytest.mark.asyncio
async def test_posix_cleanup_stops_owned_browser_descendants(monkeypatch, tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.is_running.return_value = False
	owned_child = MagicMock()
	owned_child.pid = 124
	owned_child.create_time.return_value = 21.0
	owned_child.is_running.side_effect = [True, False]
	owned_child.cmdline.return_value = [
		'chrome',
		f'--user-data-dir={profile_dir.resolve()}',
		'--openbrowser-instance-id=instance-a',
	]
	foreign_child = MagicMock()
	foreign_child.pid = 125
	foreign_child.create_time.return_value = 22.0
	foreign_child.is_running.return_value = False
	foreign_child.cmdline.return_value = [
		'chrome',
		f'--user-data-dir={(tmp_path / "foreign").resolve()}',
		'--openbrowser-instance-id=instance-b',
	]
	process.children.return_value = [owned_child, foreign_child]
	record = _browser_record(profile_dir, pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	with patch('asyncio.sleep', new_callable=AsyncMock):
		result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is True
	owned_child.terminate.assert_called_once()
	owned_child.kill.assert_not_called()
	foreign_child.terminate.assert_not_called()
	foreign_child.kill.assert_not_called()


@pytest.mark.asyncio
async def test_browser_kill_cleans_cache_before_releasing_profile_lease(tmp_path: Path):
	profile = MagicMock()
	profile.user_data_dir = str(tmp_path / 'profile')
	profile.profile_directory = 'Default'
	profile.cdp_url = None
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())
	lease = MagicMock()
	watchdog._profile_lease = lease
	watchdog._subprocess = None
	watchdog._temp_dirs_to_cleanup = []
	actions: list[str] = []
	watchdog._cleanup_profile_cache = MagicMock(side_effect=lambda *args: actions.append('cache') or 0)
	lease.clear_browser.side_effect = lambda: actions.append('clear')
	lease.release.side_effect = lambda: actions.append('release')

	await watchdog.on_BrowserKillEvent(MagicMock())

	assert actions == ['cache', 'clear', 'release']


@pytest.mark.asyncio
async def test_launch_failure_keeps_owned_process_and_lease_when_cleanup_fails(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	profile = MagicMock()
	profile.user_data_dir = str(profile_dir)
	profile.profile_directory = 'Default'
	profile.executable_path = '/custom/chrome'
	profile.get_args.return_value = [f'--user-data-dir={profile_dir}']
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())

	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	with (
		patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=MagicMock(pid=123)),
		patch('psutil.Process', return_value=process),
		patch.object(watchdog, '_find_free_port', return_value=9223),
		patch.object(watchdog, '_wait_for_cdp_url', new_callable=AsyncMock, side_effect=TimeoutError('startup failed')),
		patch.object(watchdog, '_cleanup_process', new_callable=AsyncMock, return_value=False) as cleanup,
		pytest.raises(RuntimeError, match='Unable to safely close'),
	):
		await watchdog._launch_browser(max_retries=1)

	try:
		assert watchdog._subprocess is process
		assert watchdog._browser_record is not None
		assert watchdog._profile_lease is not None
		cleanup.assert_awaited_once()
		assert cleanup.await_args.kwargs['require_identity'] is True
	finally:
		await watchdog._release_profile_lease()


@pytest.mark.asyncio
async def test_launch_cancellation_cleans_owned_process_before_propagating(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	profile = MagicMock()
	profile.user_data_dir = str(profile_dir)
	profile.profile_directory = 'Default'
	profile.executable_path = '/custom/chrome'
	profile.get_args.return_value = [f'--user-data-dir={profile_dir}']
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())

	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	with (
		patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=MagicMock(pid=123)),
		patch('psutil.Process', return_value=process),
		patch.object(watchdog, '_find_free_port', return_value=9223),
		patch.object(watchdog, '_wait_for_cdp_url', new_callable=AsyncMock, side_effect=asyncio.CancelledError),
		patch.object(watchdog, '_cleanup_process', new_callable=AsyncMock, return_value=True) as cleanup,
	):
		with pytest.raises(asyncio.CancelledError):
			await watchdog._launch_browser(max_retries=1)

	cleanup.assert_awaited_once()
	assert watchdog._subprocess is None
	assert watchdog._profile_lease is None


@pytest.mark.asyncio
async def test_psutil_conversion_failure_cleans_raw_subprocess_before_releasing_lease(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	profile = MagicMock()
	profile.user_data_dir = str(profile_dir)
	profile.profile_directory = 'Default'
	profile.executable_path = '/custom/chrome'
	profile.get_args.return_value = [f'--user-data-dir={profile_dir}']
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session)
	raw_process = MagicMock(pid=123, returncode=None)

	with (
		patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=raw_process),
		patch('psutil.Process', side_effect=OSError('process lookup failed')),
		patch.object(watchdog, '_ensure_profile_lease', new_callable=AsyncMock),
		patch.object(watchdog, '_release_profile_lease', new_callable=AsyncMock) as release_lease,
		patch.object(
			watchdog, '_cleanup_spawned_subprocess', new_callable=AsyncMock, return_value=True, create=True
		) as cleanup,
		patch.object(watchdog, '_find_free_port', return_value=9223),
		pytest.raises(OSError, match='process lookup failed'),
	):
		await watchdog._launch_browser(max_retries=1)

	cleanup.assert_awaited_once_with(raw_process)
	release_lease.assert_awaited_once()


@pytest.mark.asyncio
async def test_raw_subprocess_cleanup_failure_retains_ownership(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	profile = MagicMock()
	profile.user_data_dir = str(profile_dir)
	profile.profile_directory = 'Default'
	profile.executable_path = '/custom/chrome'
	profile.get_args.return_value = [f'--user-data-dir={profile_dir}']
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session)
	raw_process = MagicMock(pid=123, returncode=None)

	with (
		patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=raw_process),
		patch('psutil.Process', side_effect=OSError('process lookup failed')),
		patch.object(watchdog, '_ensure_profile_lease', new_callable=AsyncMock),
		patch.object(watchdog, '_release_profile_lease', new_callable=AsyncMock) as release_lease,
		patch.object(
			watchdog, '_cleanup_spawned_subprocess', new_callable=AsyncMock, return_value=False, create=True
		) as cleanup,
		patch.object(watchdog, '_find_free_port', return_value=9223),
		pytest.raises(RuntimeError, match='Unable to safely close'),
	):
		await watchdog._launch_browser(max_retries=1)

	cleanup.assert_awaited_once_with(raw_process)
	release_lease.assert_not_awaited()
	assert watchdog._spawned_subprocess is raw_process


@pytest.mark.asyncio
async def test_cleanup_spawned_subprocess_uses_exact_child_handle():
	class ChildProcess:
		returncode = None

		def __init__(self):
			self.terminated = False
			self.killed = False

		def terminate(self):
			self.terminated = True

		def kill(self):
			self.killed = True

		async def wait(self):
			self.returncode = 0

	process = ChildProcess()

	result = await LocalBrowserWatchdog._cleanup_spawned_subprocess(process)

	assert result is True
	assert process.terminated is True
	assert process.killed is False


@pytest.mark.asyncio
async def test_stale_reclaim_fails_closed_when_owner_identity_is_inaccessible(tmp_path: Path):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.info = {'name': 'chrome'}
	process.cmdline.return_value = [
		'chrome',
		f'--user-data-dir={tmp_path.resolve()}',
		'--openbrowser-instance-id=instance-a',
	]
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(tmp_path.resolve()),
		'browser': _browser_record(tmp_path, pid=123, start_time=20.0),
	}

	with (
		patch('psutil.Process', side_effect=psutil.AccessDenied(pid=999)),
		patch('psutil.process_iter', return_value=[process]),
		patch.object(LocalBrowserWatchdog, '_cleanup_process', new_callable=AsyncMock) as cleanup,
	):
		result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(
			str(tmp_path), metadata=metadata, instance_id='instance-b'
		)

	assert result is False
	cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_reclaim_refuses_to_replace_an_unverified_browser(tmp_path: Path):
	profile = MagicMock()
	profile.user_data_dir = str(tmp_path)
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=MagicMock(browser_profile=profile))
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(tmp_path.resolve()),
		'browser': _browser_record(tmp_path),
	}

	with (
		patch.object(ProfileLease, 'process_identity_status', return_value='missing'),
		patch.object(watchdog, '_kill_stale_chrome_for_profile', new_callable=AsyncMock, return_value=False),
		pytest.raises(ProfileInUseError),
	):
		await watchdog._reclaim_previous_metadata(metadata, '--openbrowser-instance-id=instance-b')


@pytest.mark.asyncio
async def test_reclaim_scans_for_orphaned_browser_without_record(tmp_path: Path):
	profile = MagicMock()
	profile.user_data_dir = str(tmp_path)
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=MagicMock(browser_profile=profile))
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(tmp_path.resolve()),
		'browser': None,
	}

	with (
		patch.object(ProfileLease, 'process_identity_status', return_value='missing'),
		patch.object(watchdog, '_kill_stale_chrome_for_profile', new_callable=AsyncMock, return_value=True) as cleanup,
	):
		await watchdog._reclaim_previous_metadata(metadata, '--openbrowser-instance-id=instance-b')

	cleanup.assert_awaited_once_with(
		str(tmp_path.resolve()),
		metadata=metadata,
		instance_id=watchdog._instance_id,
	)


@pytest.mark.asyncio
async def test_profile_lease_is_released_when_reclaim_is_cancelled(tmp_path: Path):
	profile = MagicMock()
	profile.user_data_dir = str(tmp_path)
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=MagicMock(browser_profile=profile))
	lease = MagicMock()
	lease.acquire.return_value = None

	with (
		patch('openbrowser.browser.watchdogs.local_browser_watchdog.ProfileLease', return_value=lease),
		patch.object(watchdog, '_reclaim_previous_metadata', new_callable=AsyncMock, side_effect=asyncio.CancelledError),
	):
		with pytest.raises(asyncio.CancelledError):
			await watchdog._ensure_profile_lease(str(tmp_path), '--openbrowser-instance-id=instance-a')

	lease.release.assert_called_once()
	assert watchdog._profile_lease is None


@pytest.mark.asyncio
async def test_orphaned_marker_process_is_reclaimed_without_browser_record(tmp_path: Path):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.info = {
		'name': 'chrome',
		'cmdline': [
			'chrome',
			f'--user-data-dir={tmp_path.resolve()}',
			'--openbrowser-instance-id=instance-a',
		],
	}
	process.cmdline.return_value = process.info['cmdline']
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(tmp_path.resolve()),
		'browser': None,
	}

	with (
		patch.object(ProfileLease, 'process_identity_status', return_value='missing'),
		patch('psutil.process_iter', return_value=[process]),
		patch.object(LocalBrowserWatchdog, '_cleanup_process', new_callable=AsyncMock, return_value=True) as cleanup,
	):
		result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(
			str(tmp_path), metadata=metadata, instance_id='instance-b'
		)

	assert result is True
	cleanup.assert_awaited_once()
	assert cleanup.await_args.kwargs['browser_record']['pid'] == 123


@pytest.mark.asyncio
async def test_orphaned_marker_process_with_inaccessible_command_line_fails_closed(tmp_path: Path):
	process = MagicMock()
	process.pid = 123
	process.info = {'name': 'chrome', 'cmdline': None}
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(tmp_path.resolve()),
		'browser': None,
	}

	with (
		patch.object(ProfileLease, 'process_identity_status', return_value='missing'),
		patch('psutil.process_iter', return_value=[process]),
	):
		result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(
			str(tmp_path), metadata=metadata, instance_id='instance-b'
		)

	assert result is False


@pytest.mark.asyncio
async def test_cleanup_rejects_reused_root_pid_before_terminate(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.side_effect = [20.0, 99.0]
	process.children.return_value = []
	process.is_running.return_value = False
	record = _browser_record(Path('/tmp/profile'), pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is False
	process.terminate.assert_not_called()
	process.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_reports_owned_descendant_failure_after_root_exit(monkeypatch, tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	process = MagicMock()
	process.pid = 123
	process.create_time.side_effect = [20.0, psutil.NoSuchProcess(pid=123)]
	process.is_running.return_value = False
	child = MagicMock()
	child.pid = 124
	child.create_time.return_value = 21.0
	child.is_running.return_value = True
	child.cmdline.return_value = [
		'chrome',
		f'--user-data-dir={profile_dir.resolve()}',
		'--openbrowser-instance-id=instance-a',
	]
	process.children.return_value = [child]
	record = _browser_record(profile_dir, pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	with patch('asyncio.sleep', new_callable=AsyncMock):
		result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is False


@pytest.mark.asyncio
async def test_cleanup_rescans_after_root_disappears_during_terminate(monkeypatch, tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.children.return_value = []
	process.terminate.side_effect = psutil.NoSuchProcess(pid=123)
	child = MagicMock()
	child.pid = 124
	child.create_time.return_value = 21.0
	child.info = {
		'name': 'chrome',
		'cmdline': [
			'chrome',
			f'--user-data-dir={profile_dir.resolve()}',
			'--openbrowser-instance-id=instance-a',
		],
	}
	child.cmdline.return_value = child.info['cmdline']
	record = _browser_record(profile_dir, pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	with (
		patch('psutil.process_iter', return_value=[child]),
		patch.object(LocalBrowserWatchdog, '_wait_for_process_exit', new_callable=AsyncMock, side_effect=[False, True]),
	):
		result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is True
	child.kill.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_fails_closed_for_inaccessible_browser_descendant(monkeypatch, tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.is_running.return_value = False
	child = MagicMock()
	child.pid = 124
	child.create_time.return_value = 21.0
	child.info = {'name': 'chrome', 'cmdline': None}
	child.cmdline.side_effect = psutil.AccessDenied(pid=124)
	process.children.return_value = [child]
	record = _browser_record(profile_dir, pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is False
	process.terminate.assert_not_called()


def test_orphan_scan_includes_microsoft_edge(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	edge = MagicMock()
	edge.pid = 124
	edge.create_time.return_value = 21.0
	edge.info = {
		'name': 'msedge.exe',
		'cmdline': [
			'msedge.exe',
			f'--user-data-dir={profile_dir.resolve()}',
			'--openbrowser-instance-id=instance-a',
		],
	}
	edge.cmdline.return_value = edge.info['cmdline']

	with patch('psutil.process_iter', return_value=[edge]):
		owned, scan_succeeded = LocalBrowserWatchdog._find_owned_descendants(
			_browser_record(profile_dir, pid=123, start_time=20.0), excluded_pids={123}
		)

	assert scan_succeeded is True
	assert [process.pid for process, _ in owned] == [124]


@pytest.mark.asyncio
async def test_cleanup_scans_for_owned_descendants_when_root_is_gone(monkeypatch, tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	process = MagicMock()
	process.pid = 123
	process.create_time.side_effect = [20.0, psutil.NoSuchProcess(pid=123)]
	process.children.side_effect = psutil.NoSuchProcess(pid=123)
	child = MagicMock()
	child.pid = 124
	child.create_time.return_value = 21.0
	child.info = {
		'name': 'chrome',
		'cmdline': [
			'chrome',
			f'--user-data-dir={profile_dir.resolve()}',
			'--openbrowser-instance-id=instance-a',
		],
	}
	child.cmdline.return_value = child.info['cmdline']
	record = _browser_record(profile_dir, pid=123, start_time=20.0)

	monkeypatch.setattr(
		'openbrowser.browser.watchdogs.local_browser_watchdog.IS_WINDOWS',
		False,
	)
	with (
		patch('psutil.process_iter', return_value=[child]),
		patch.object(LocalBrowserWatchdog, '_wait_for_process_exit', new_callable=AsyncMock, side_effect=[False, True]),
	):
		result = await LocalBrowserWatchdog._cleanup_process(process, browser_record=record)

	assert result is True
	child.kill.assert_called_once()


@pytest.mark.asyncio
async def test_stale_reclaim_cleans_orphaned_descendant_when_root_is_gone(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	child = MagicMock()
	child.pid = 124
	child.create_time.return_value = 21.0
	child.info = {
		'name': 'chrome',
		'cmdline': [
			'chrome',
			f'--user-data-dir={profile_dir.resolve()}',
			'--openbrowser-instance-id=instance-a',
		],
	}
	child.cmdline.return_value = child.info['cmdline']
	browser = _browser_record(profile_dir, pid=123, start_time=20.0)
	metadata = {
		'instance_id': 'instance-a',
		'owner_pid': 999,
		'owner_start_time': 1.0,
		'profile_dir': str(profile_dir.resolve()),
		'browser': browser,
	}

	with (
		patch.object(ProfileLease, 'process_identity_status', return_value='missing'),
		patch('psutil.Process', side_effect=psutil.NoSuchProcess(pid=123)),
		patch('psutil.process_iter', return_value=[child]),
		patch.object(LocalBrowserWatchdog, '_wait_for_process_exit', new_callable=AsyncMock, side_effect=[False, True]),
	):
		result = await LocalBrowserWatchdog._kill_stale_chrome_for_profile(
			str(profile_dir), metadata=metadata, instance_id='instance-b'
		)

	assert result is True
	child.kill.assert_called_once()


@pytest.mark.asyncio
async def test_browser_kill_releases_lease_when_metadata_clear_fails(tmp_path: Path):
	profile = MagicMock()
	profile.user_data_dir = str(tmp_path / 'profile')
	profile.profile_directory = 'Default'
	profile.cdp_url = None
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())
	lease = MagicMock()
	lease.clear_browser.side_effect = OSError('metadata unavailable')
	watchdog._profile_lease = lease

	await watchdog.on_BrowserKillEvent(MagicMock())

	lease.release.assert_called_once()


@pytest.mark.asyncio
async def test_launch_fails_closed_when_browser_identity_cannot_be_recorded(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	profile = MagicMock()
	profile.user_data_dir = str(profile_dir)
	profile.profile_directory = 'Default'
	profile.executable_path = '/custom/chrome'
	profile.get_args.return_value = [f'--user-data-dir={profile_dir}']
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())

	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	with (
		patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=MagicMock(pid=123)),
		patch('psutil.Process', return_value=process),
		patch.object(watchdog, '_find_free_port', return_value=9223),
		patch.object(watchdog, '_wait_for_cdp_url', new_callable=AsyncMock, return_value='http://127.0.0.1:9223/'),
		patch.object(ProfileLease, 'record_browser', side_effect=TypeError('unserializable identity')),
		patch.object(watchdog, '_cleanup_process', new_callable=AsyncMock, return_value=False) as cleanup,
		pytest.raises(RuntimeError, match='Unable to safely close'),
	):
		await watchdog._launch_browser(max_retries=1)

	try:
		assert watchdog._subprocess is process
		assert watchdog._browser_record is None
		assert watchdog._profile_lease is not None
		cleanup.assert_awaited_once()
		assert cleanup.await_args.kwargs['browser_record'] is None
		assert cleanup.await_args.kwargs['require_identity'] is True
	finally:
		await watchdog._release_profile_lease()


@pytest.mark.asyncio
async def test_explicit_profile_conflict_is_not_replaced_with_temp_profile(tmp_path: Path):
	profile_dir = tmp_path / 'shared-profile'
	owner = ProfileLease(profile_dir, instance_id='other-instance')
	owner.acquire()

	try:
		profile = MagicMock()
		profile.user_data_dir = str(profile_dir)
		profile.profile_directory = 'Default'
		profile.executable_path = '/custom/chrome'
		profile.get_args.return_value = [f'--user-data-dir={profile_dir}']
		session = MagicMock()
		session.browser_profile = profile
		watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())

		with (
			patch('tempfile.mkdtemp') as make_temp_dir,
			pytest.raises(ProfileInUseError, match='Browser profile already in use'),
		):
			await watchdog._launch_browser(max_retries=2)

		make_temp_dir.assert_not_called()
		assert profile.user_data_dir == str(profile_dir)
	finally:
		owner.release()


@pytest.mark.asyncio
async def test_launch_records_browser_identity_in_profile_metadata(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	profile = MagicMock()
	profile.user_data_dir = str(profile_dir)
	profile.profile_directory = 'Default'
	profile.executable_path = '/custom/chrome'
	profile.get_args.side_effect = lambda: [f'--user-data-dir={profile.user_data_dir}']
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())

	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	with (
		patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, return_value=MagicMock(pid=123)),
		patch('psutil.Process', return_value=process),
		patch.object(watchdog, '_find_free_port', return_value=9223),
		patch.object(watchdog, '_wait_for_cdp_url', new_callable=AsyncMock, return_value='http://127.0.0.1:9223/'),
	):
		await watchdog._launch_browser(max_retries=1)

	try:
		metadata = ProfileLease.read_metadata(profile_dir)
		assert metadata is not None
		assert metadata['browser'] == {
			'pid': 123,
			'start_time': 20.0,
			'profile_dir': str(profile_dir.resolve()),
			'instance_id': watchdog._instance_id,
			'ownership_marker': f'--openbrowser-instance-id={watchdog._instance_id}',
			'executable': '/custom/chrome',
			'cdp_port': 9223,
		}
	finally:
		await watchdog._release_profile_lease()


@pytest.mark.asyncio
async def test_successful_temp_profile_stays_until_browser_cleanup(tmp_path: Path):
	original_profile = tmp_path / 'original-profile'
	temporary_profile = tmp_path / 'openbrowser-tmp-profile'
	profile = MagicMock()
	profile.user_data_dir = str(original_profile)
	profile.profile_directory = 'Default'
	profile.executable_path = '/custom/chrome'
	profile.get_args.side_effect = lambda: [f'--user-data-dir={profile.user_data_dir}']
	session = MagicMock()
	session.browser_profile = profile
	watchdog = LocalBrowserWatchdog.model_construct(browser_session=session, event_bus=MagicMock())

	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	launches = 0

	async def launch(*args, **kwargs):
		nonlocal launches
		launches += 1
		if launches == 1:
			raise RuntimeError('user data directory already in use')
		return MagicMock(pid=123)

	with (
		patch('asyncio.create_subprocess_exec', new_callable=AsyncMock, side_effect=launch),
		patch('psutil.Process', return_value=process),
		patch('tempfile.mkdtemp', return_value=str(temporary_profile)),
		patch.object(watchdog, '_find_free_port', return_value=9223),
		patch.object(watchdog, '_wait_for_cdp_url', new_callable=AsyncMock, return_value='http://127.0.0.1:9223/'),
		patch('asyncio.sleep', new_callable=AsyncMock),
	):
		await watchdog._launch_browser(max_retries=2)

	try:
		assert temporary_profile.exists()
		assert watchdog._profile_lease is not None
		assert watchdog._profile_lease.profile_dir == temporary_profile.resolve()
	finally:
		await watchdog._release_profile_lease()


@pytest.mark.asyncio
async def test_stop_dispatches_cleanup_when_lease_exists():
	watchdog = MagicMock(spec=LocalBrowserWatchdog)
	watchdog.browser_session = MagicMock()
	watchdog.browser_session.is_local = True
	watchdog._subprocess = None
	watchdog._profile_lease = MagicMock()
	watchdog.event_bus = MagicMock()

	await LocalBrowserWatchdog.on_BrowserStopEvent(watchdog, MagicMock())

	watchdog.event_bus.dispatch.assert_called_once()


def test_lease_metadata_helpers_are_used_for_recorded_browser(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	try:
		lease.write_metadata(browser=_browser_record(tmp_path / 'profile'))
		metadata = ProfileLease.read_metadata(tmp_path / 'profile')
		assert metadata is not None
		assert metadata['browser']['ownership_marker'] == '--openbrowser-instance-id=instance-a'
	finally:
		lease.release()
