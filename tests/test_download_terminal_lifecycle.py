"""Wiring tests for the explicit download terminal lifecycle."""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from bubus import EventBus

from openbrowser.browser.download_state import DownloadPhase, DownloadTimeoutError
from openbrowser.browser.session import BrowserSession


def _make_session(downloads_path):
	session = create_autospec(BrowserSession, instance=True)
	session.logger = logging.getLogger('test-download-lifecycle')
	session._cdp_client_root = MagicMock()
	session.agent_focus = None
	session.get_or_create_cdp_session = AsyncMock()
	session.get_current_page_url = AsyncMock(return_value='https://example.com')
	session.cdp_client = MagicMock()
	session.cdp_client.send.Browser.setDownloadBehavior = AsyncMock()
	session.cdp_client.send.Network.enable = AsyncMock()
	session.cdp_client.register.Browser.downloadWillBegin = MagicMock()
	session.cdp_client.register.Browser.downloadProgress = MagicMock()
	session.cdp_client.register.Network.responseReceived = MagicMock()
	session.id = 'test-session-1234'
	session.is_local = True
	session.get_target_id_from_session_id = MagicMock(return_value=None)
	session.cdp_client_for_frame = AsyncMock()
	session.browser_profile = MagicMock()
	session.browser_profile.downloads_path = downloads_path
	session.browser_profile.auto_download_pdfs = False
	return session


def _make_watchdog(downloads_path):
	from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

	session = _make_session(downloads_path)
	# A real dispatch leaves loop-bound bubus work that makes pytest-asyncio
	# teardown flaky. These tests only assert the registry, so stub dispatch.
	event_bus = EventBus()
	event_bus.dispatch = MagicMock()
	return DownloadsWatchdog(event_bus=event_bus, browser_session=session), session


async def _registered_handlers(watchdog):
	await watchdog.attach_to_target('target-1')
	return (
		watchdog.browser_session.cdp_client.register.Browser.downloadWillBegin.call_args.args[0],
		watchdog.browser_session.cdp_client.register.Browser.downloadProgress.call_args.args[0],
	)


@pytest.mark.asyncio
async def test_will_begin_records_a_started_outcome():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, _ = await _registered_handlers(watchdog)

		will_begin(
			{'guid': 'g-1', 'url': 'https://example.com/a.pdf', 'suggestedFilename': 'a.pdf'},
			'session-1',
		)

		outcome = watchdog.latest_download_outcome()
		assert outcome['phase'] == 'started'
		assert outcome['url'] == 'https://example.com/a.pdf'
		assert outcome['file_name'] == 'a.pdf'
		assert outcome['source'] == 'cdp'


@pytest.mark.asyncio
async def test_progress_completed_records_the_terminal_path():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, progress = await _registered_handlers(watchdog)
		will_begin(
			{'guid': 'g-2', 'url': 'https://example.com/b.bin', 'suggestedFilename': 'b.bin'},
			'session-1',
		)

		Path(tmpdir, 'b.bin').write_bytes(b'payload')
		progress(
			{
				'guid': 'g-2',
				'state': 'completed',
				'filePath': str(Path(tmpdir) / 'b.bin'),
				'totalBytes': 42,
			},
			'session-1',
		)

		outcome = watchdog.latest_download_outcome()
		assert outcome['phase'] == 'completed'
		assert outcome['path'] == str(Path(tmpdir) / 'b.bin')
		assert outcome['file_size'] == 42


@pytest.mark.asyncio
async def test_progress_canceled_is_a_terminal_failure():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, progress = await _registered_handlers(watchdog)
		will_begin(
			{'guid': 'g-3', 'url': 'https://example.com/c.bin', 'suggestedFilename': 'c.bin'},
			'session-1',
		)

		progress({'guid': 'g-3', 'state': 'canceled'}, 'session-1')

		outcome = watchdog.latest_download_outcome()
		assert outcome['phase'] == 'cancelled'
		assert 'canceled' in outcome['detail']


@pytest.mark.asyncio
async def test_a_late_failure_cannot_overwrite_completed():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, progress = await _registered_handlers(watchdog)
		will_begin(
			{'guid': 'g-4', 'url': 'https://example.com/d.bin', 'suggestedFilename': 'd.bin'},
			'session-1',
		)
		Path(tmpdir, 'd.bin').write_bytes(b'payload')
		progress(
			{'guid': 'g-4', 'state': 'completed', 'filePath': str(Path(tmpdir) / 'd.bin')},
			'session-1',
		)
		progress({'guid': 'g-4', 'state': 'canceled'}, 'session-1')

		assert watchdog.latest_download_outcome()['phase'] == 'completed'


@pytest.mark.asyncio
async def test_browser_stop_cancels_pending_downloads():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, _ = await _registered_handlers(watchdog)
		will_begin(
			{'guid': 'g-5', 'url': 'https://example.com/e.bin', 'suggestedFilename': 'e.bin'},
			'session-1',
		)

		await watchdog.on_BrowserStoppedEvent(MagicMock())

		outcome = watchdog.latest_download_outcome()
		assert outcome['phase'] == 'cancelled'
		assert outcome['detail'] == 'browser stopped'
		assert watchdog._cdp_event_tasks == set()


@pytest.mark.asyncio
async def test_download_file_from_url_records_a_fetch_failure():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, session = _make_watchdog(tmpdir)
		fake_session = MagicMock()
		fake_session.session_id = 'cdp-1'
		fake_session.cdp_client.send.Runtime.evaluate = AsyncMock(side_effect=RuntimeError('boom'))
		session.get_or_create_cdp_session = AsyncMock(return_value=fake_session)

		result = await watchdog.download_file_from_url('https://example.com/x.bin', 'target-1')

		assert result is None
		outcome = watchdog.latest_download_outcome()
		assert outcome['phase'] == 'failed'
		assert outcome['stage'] == 'fetch'
		assert 'boom' in outcome['error']


@pytest.mark.asyncio
async def test_pdf_download_records_a_probe_failure():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, session = _make_watchdog(tmpdir)
		fake_session = MagicMock()
		fake_session.session_id = 'cdp-2'
		fake_session.cdp_client.send.Runtime.evaluate = AsyncMock(return_value={'result': {'value': {}}})
		session.get_or_create_cdp_session = AsyncMock(return_value=fake_session)

		result = await watchdog.trigger_pdf_download('target-2')

		assert result is None
		outcome = watchdog.latest_download_outcome()
		assert outcome['phase'] == 'failed'
		assert outcome['stage'] == 'pdf_probe'


@pytest.mark.asyncio
async def test_wait_for_download_returns_the_terminal_outcome():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, progress = await _registered_handlers(watchdog)

		async def finish_later():
			await asyncio.sleep(0.05)
			will_begin(
				{'guid': 'g-6', 'url': 'https://example.com/f.bin', 'suggestedFilename': 'f.bin'},
				'session-1',
			)
			Path(tmpdir, 'f.bin').write_bytes(b'payload')
			progress(
				{'guid': 'g-6', 'state': 'completed', 'filePath': str(Path(tmpdir) / 'f.bin')},
				'session-1',
			)

		waiter = asyncio.create_task(watchdog.wait_for_download(timeout=2.0))
		await finish_later()

		outcome = await waiter
		assert outcome['phase'] == 'completed'
		assert outcome['path'] == str(Path(tmpdir) / 'f.bin')


@pytest.mark.asyncio
async def test_wait_for_download_timeout_reports_the_last_phase():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, _ = await _registered_handlers(watchdog)
		will_begin(
			{'guid': 'g-7', 'url': 'https://example.com/g.bin', 'suggestedFilename': 'g.bin'},
			'session-1',
		)

		with pytest.raises(DownloadTimeoutError) as exc_info:
			await watchdog.wait_for_download(timeout=0.15)

		assert exc_info.value.outcome['phase'] == 'started'


@pytest.mark.asyncio
async def test_outcomes_are_json_safe():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, _ = await _registered_handlers(watchdog)
		will_begin(
			{'guid': 'g-8', 'url': 'https://example.com/h.bin', 'suggestedFilename': 'h.bin'},
			'session-1',
		)

		payload = json.dumps(watchdog.download_outcomes())

		assert 'g-8' in payload


def _fake_watchdog(registry):
	class FakeWatchdog:
		_download_registry = registry

		def download_outcomes(self):
			return [outcome.to_dict() for outcome in registry.outcomes()]

		def latest_download_outcome(self):
			outcome = registry.latest()
			return outcome.to_dict() if outcome is not None else None

		async def shutdown(self):
			registry.cancel_pending(detail='browser stopped')
			return [outcome.to_dict() for outcome in registry.outcomes()]

		async def wait_for_download(self, *, timeout, url=None, download_id=None, lookback=0.0):
			outcome = await registry.wait_for_terminal(timeout=timeout, url=url, download_id=download_id, lookback=lookback)
			return outcome.to_dict()

	return FakeWatchdog()


@pytest.mark.asyncio
async def test_session_exposes_download_outcomes_and_wait():
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession

	session = BrowserSession()
	registry = DownloadRegistry()
	registry.complete('session-1', path='/tmp/s.bin', file_size=7)
	session._downloads_watchdog = _fake_watchdog(registry)

	assert session.downloads_registry is registry
	assert session.download_outcomes()[0]['phase'] == 'completed'
	assert session.latest_download_outcome()['download_id'] == 'session-1'

	outcome = await session.wait_for_download(timeout=1.0, download_id='session-1')
	assert outcome['phase'] == 'completed'


@pytest.mark.asyncio
async def test_namespace_exposes_download_helpers():
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession
	from openbrowser.code_use.namespace import create_namespace

	session = BrowserSession()
	registry = DownloadRegistry()
	registry.complete('ns-1', path='/tmp/n.bin', file_size=8)
	registry.fail('ns-2', stage='fetch', error='boom')
	session._downloads_watchdog = _fake_watchdog(registry)

	tools = MagicMock()
	tools.registry.registry.actions = {}
	namespace = create_namespace(session, tools=tools)

	assert len(namespace['download_outcomes']()) == 2
	failures = namespace['download_failures']()
	assert len(failures) == 1
	assert failures[0]['download_id'] == 'ns-2'
	assert failures[0]['stage'] == 'fetch'
	assert len(namespace['download_outcomes']()) == 2


@pytest.mark.asyncio
async def test_namespace_download_file_records_a_completed_outcome():
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession
	from openbrowser.code_use.namespace import create_namespace

	session = BrowserSession()
	registry = DownloadRegistry()
	session._downloads_watchdog = _fake_watchdog(registry)
	object.__setattr__(session, 'get_or_create_cdp_session', AsyncMock(side_effect=RuntimeError('no browser')))

	with tempfile.TemporaryDirectory(prefix='ob-download-lifecycle-') as downloads_dir:
		session.browser_profile.downloads_path = downloads_dir
		tools = MagicMock()
		tools.registry.registry.actions = {}
		namespace = create_namespace(session, tools=tools)

		response = MagicMock()
		response.raise_for_status = MagicMock()
		response.iter_content = MagicMock(return_value=[b'payload'])

		with patch('openbrowser.code_use.namespace.requests.get', return_value=response):
			path = await namespace['download_file']('https://example.com/file.bin')

		assert Path(path).read_bytes() == b'payload'

	outcome = registry.latest()
	assert outcome.phase == DownloadPhase.COMPLETED
	assert outcome.source == 'namespace'
	assert outcome.path == path


@pytest.mark.asyncio
async def test_missing_downloads_path_records_a_configuration_failure():
	watchdog, _ = _make_watchdog(None)

	result = await watchdog.download_file_from_url('https://example.com/x.bin', 'target-1')

	assert result is None
	outcome = watchdog.latest_download_outcome()
	assert outcome['phase'] == 'failed'
	assert outcome['stage'] == 'configuration'


@pytest.mark.asyncio
async def test_pdf_missing_downloads_path_records_a_configuration_failure():
	watchdog, _ = _make_watchdog(None)

	result = await watchdog.trigger_pdf_download('target-1')

	assert result is None
	outcome = watchdog.latest_download_outcome()
	assert outcome['phase'] == 'failed'
	assert outcome['stage'] == 'configuration'


def test_name_matching_accepts_browser_duplicate_names():
	assert watchdog_module()._download_name_matches('report.pdf', 'report.pdf')
	assert watchdog_module()._download_name_matches('report (1).pdf', 'report.pdf')
	assert watchdog_module()._download_name_matches('report (12).pdf', 'report.pdf')
	assert not watchdog_module()._download_name_matches('other.pdf', 'report.pdf')
	assert not watchdog_module()._download_name_matches('report.pdf.part', 'report.pdf')
	assert not watchdog_module()._download_name_matches('report (draft).pdf', 'report.pdf')
	assert watchdog_module()._download_name_matches('report (1) (2).pdf', 'report.pdf')
	assert not watchdog_module()._download_name_matches('report (١).pdf', 'report.pdf')


def watchdog_module():
	from openbrowser.browser.watchdogs.downloads_watchdog import DownloadsWatchdog

	return DownloadsWatchdog


@pytest.mark.asyncio
async def test_ambiguous_progress_without_a_guid_is_ignored():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, _ = _make_watchdog(tmpdir)
		will_begin, progress = await _registered_handlers(watchdog)
		will_begin({'guid': 'g-a', 'url': 'https://example.com/a', 'suggestedFilename': 'a.bin'}, None)
		will_begin({'guid': 'g-b', 'url': 'https://example.com/b', 'suggestedFilename': 'b.bin'}, None)

		progress({'state': 'completed', 'filePath': str(Path(tmpdir) / 'a.bin')}, None)

		assert [outcome['phase'] for outcome in watchdog.download_outcomes()] == ['started', 'started']


@pytest.mark.asyncio
async def test_session_keeps_download_history_after_detaching_the_watchdog():
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession

	session = BrowserSession()
	registry = DownloadRegistry()
	registry.start('pending', url='https://example.com/p')
	registry.complete('done', path='/tmp/done.bin', file_size=2)
	session._downloads_watchdog = _fake_watchdog(registry)

	history = await session._capture_download_history()
	session._downloads_watchdog = None
	session._download_history = history

	outcomes = session.download_outcomes()
	assert [outcome['download_id'] for outcome in outcomes] == ['pending', 'done']
	assert outcomes[0]['phase'] == 'cancelled'
	assert session.latest_download_outcome()['download_id'] == 'done'


@pytest.mark.asyncio
async def test_zero_byte_fetch_is_a_completed_download():
	with tempfile.TemporaryDirectory() as tmpdir:
		watchdog, session = _make_watchdog(tmpdir)
		fake_session = MagicMock()
		fake_session.session_id = 'cdp-empty'
		fake_session.cdp_client.send.Runtime.evaluate = AsyncMock(
			return_value={'result': {'value': {'data': [], 'size': 0, 'type': 'text/plain'}}}
		)
		session.get_or_create_cdp_session = AsyncMock(return_value=fake_session)

		path = await watchdog.download_file_from_url('https://example.com/empty.txt', 'target-1')

		assert path is not None
		assert Path(path).read_bytes() == b''
		outcome = watchdog.latest_download_outcome()
		assert outcome['phase'] == 'completed'
		assert outcome['file_size'] == 0


def test_synthetic_download_ids_are_unique():
	watchdog, _ = _make_watchdog('/tmp')

	first = watchdog._next_synthetic_download_id('cdp')
	second = watchdog._next_synthetic_download_id('cdp')

	assert first != second
	assert first.startswith('cdp-')


def test_session_download_outcomes_error_is_public():
	from openbrowser.browser.session import BrowserSession

	session = BrowserSession()

	class BrokenWatchdog:
		def download_outcomes(self):
			raise RuntimeError('reader broke')

	session._downloads_watchdog = BrokenWatchdog()

	session.download_outcomes()

	assert session.download_outcomes_error == 'RuntimeError: reader broke'


@pytest.mark.asyncio
async def test_namespace_download_file_accepts_an_empty_browser_fetch():
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession
	from openbrowser.code_use.namespace import create_namespace

	session = BrowserSession()
	registry = DownloadRegistry()
	session._downloads_watchdog = _fake_watchdog(registry)
	fake_cdp = MagicMock()
	fake_cdp.session_id = 'cdp-empty'
	fake_cdp.cdp_client.send.Runtime.evaluate = AsyncMock(
		return_value={'result': {'value': {'base64': '', 'size': 0, 'type': 'text/plain'}}}
	)
	object.__setattr__(session, 'get_or_create_cdp_session', AsyncMock(return_value=fake_cdp))

	with tempfile.TemporaryDirectory(prefix='ob-empty-fetch-') as downloads_dir:
		session.browser_profile.downloads_path = downloads_dir
		tools = MagicMock()
		tools.registry.registry.actions = {}
		namespace = create_namespace(session, tools=tools)

		path = await namespace['download_file']('https://example.com/empty.bin')

		assert Path(path).read_bytes() == b''

	outcome = registry.latest()
	assert outcome.phase is DownloadPhase.COMPLETED
	assert outcome.file_size == 0


def test_download_family_strips_numeric_duplicate_suffixes():
	watchdog, _ = _make_watchdog('/tmp')

	assert watchdog._download_family('report.pdf') == 'report.pdf'
	assert watchdog._download_family('report (1).pdf') == 'report.pdf'
	assert watchdog._download_family('report (12).pdf') == 'report.pdf'
	assert watchdog._download_family('report (1) (2).pdf') == 'report.pdf'
	assert watchdog._download_family('notes.txt') == 'notes.txt'


@pytest.mark.asyncio
async def test_session_capture_does_not_cancel_unrelated_caller_work():
	"""A download is a caller-issued operation. Reset must not kill the caller."""
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession

	session = BrowserSession()
	registry = DownloadRegistry()
	session._downloads_watchdog = _fake_watchdog(registry)
	task = asyncio.create_task(asyncio.sleep(60))
	await asyncio.sleep(0)

	await session._capture_download_history()

	assert not task.cancelled()
	task.cancel()
	await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_namespace_download_file_refuses_at_capacity():
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession
	from openbrowser.code_use.namespace import create_namespace

	session = BrowserSession()
	registry = DownloadRegistry(max_active=1)
	registry.start('already-running', url='https://example.com/slow')
	session._downloads_watchdog = _fake_watchdog(registry)

	with tempfile.TemporaryDirectory(prefix='ob-capacity-') as downloads_dir:
		session.browser_profile.downloads_path = downloads_dir
		tools = MagicMock()
		tools.registry.registry.actions = {}
		namespace = create_namespace(session, tools=tools)

		with pytest.raises(RuntimeError, match='Too many active downloads'):
			await namespace['download_file']('https://example.com/other.bin')


@pytest.mark.asyncio
async def test_namespace_download_file_records_a_directory_failure(tmp_path):
	from openbrowser.browser.download_state import DownloadRegistry
	from openbrowser.browser.session import BrowserSession
	from openbrowser.code_use.namespace import create_namespace

	session = BrowserSession()
	registry = DownloadRegistry()
	session._downloads_watchdog = _fake_watchdog(registry)
	blocker = tmp_path / 'not-a-directory'
	blocker.write_text('x')
	session.browser_profile.downloads_path = str(blocker)
	tools = MagicMock()
	tools.registry.registry.actions = {}
	namespace = create_namespace(session, tools=tools)

	with pytest.raises(OSError):
		await namespace['download_file']('https://example.com/x.bin')

	outcome = registry.latest()
	assert outcome.phase is DownloadPhase.FAILED
	assert outcome.stage == 'write'
	assert outcome.rejected is True


@pytest.mark.asyncio
async def test_lookback_still_works_when_an_unrelated_download_is_pending():
	from openbrowser.browser.download_state import DownloadRegistry

	registry = DownloadRegistry()
	registry.complete('recent-b', path='/tmp/b.bin', file_size=1, url='https://example.com/b')
	registry.start('pending-a', url='https://example.com/a')

	outcome = await registry.wait_for_terminal(
		timeout=0.5,
		url='https://example.com/b',
		lookback=5.0,
	)

	assert outcome.download_id == 'recent-b'


def test_prune_ambiguous_paths_releases_deleted_files(tmp_path):
	watchdog, _ = _make_watchdog(str(tmp_path))
	live = tmp_path / 'report.pdf'
	live.write_bytes(b'x')
	gone = tmp_path / 'gone.pdf'
	gone.write_bytes(b'x')
	watchdog._ambiguous_paths = {str(live), str(gone)}
	gone.unlink()

	watchdog._prune_ambiguous_paths()

	assert watchdog._ambiguous_paths == {str(live)}
