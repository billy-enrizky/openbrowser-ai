"""Behavior tests for the explicit download terminal lifecycle."""

import asyncio
import json

import pytest

from openbrowser.browser.download_state import (
	DownloadOutcome,
	DownloadPhase,
	DownloadRegistry,
	DownloadTimeoutError,
	redacted_outcome,
)


def test_request_start_complete_records_the_terminal_path():
	registry = DownloadRegistry()

	registry.request('dl-1', url='https://example.com/report.pdf', file_name='report.pdf', source='cdp')
	assert registry.get('dl-1').phase is DownloadPhase.REQUESTED

	registry.start('dl-1', url='https://example.com/report.pdf', file_name='report.pdf')
	assert registry.get('dl-1').phase is DownloadPhase.STARTED

	outcome = registry.complete(
		'dl-1',
		path='/tmp/report.pdf',
		file_size=1234,
		file_name='report.pdf',
		file_type='pdf',
		mime_type='application/pdf',
	)

	assert outcome.phase is DownloadPhase.COMPLETED
	assert outcome.terminal is True
	assert outcome.path == '/tmp/report.pdf'
	assert outcome.file_size == 1234
	assert outcome.file_type == 'pdf'
	assert outcome.mime_type == 'application/pdf'
	assert outcome.finished_at is not None


def test_complete_without_start_is_allowed():
	registry = DownloadRegistry()

	outcome = registry.complete('dl-2', path='/tmp/a.zip', file_size=10, file_name='a.zip')

	assert outcome.phase is DownloadPhase.COMPLETED
	assert outcome.path == '/tmp/a.zip'


def test_terminal_outcome_is_immutable():
	registry = DownloadRegistry()
	registry.complete('dl-3', path='/tmp/a.bin', file_size=5)

	after_failure = registry.fail('dl-3', stage='cdp', error='late cancel')
	after_cancel = registry.cancel('dl-3')

	assert after_failure.phase is DownloadPhase.COMPLETED
	assert after_failure.path == '/tmp/a.bin'
	assert after_cancel.phase is DownloadPhase.COMPLETED


def test_duplicate_completion_keeps_the_first_terminal_record():
	registry = DownloadRegistry()
	first = registry.complete('dl-4', path='/tmp/one.bin', file_size=5)
	second = registry.complete('dl-4', path='/tmp/two.bin', file_size=6)

	assert second is first
	assert second.path == '/tmp/one.bin'
	assert second.file_size == 5


def test_a_late_completion_upgrades_an_earlier_failure():
	registry = DownloadRegistry()
	registry.start('dl-upgrade')
	registry.fail('dl-upgrade', stage='native_poll', error='timed out')

	outcome = registry.complete('dl-upgrade', path='/tmp/late.bin', file_size=11)

	assert outcome.phase is DownloadPhase.COMPLETED
	assert outcome.path == '/tmp/late.bin'
	assert outcome.stage is None
	assert outcome.error is None


def test_failure_records_stage_and_error():
	registry = DownloadRegistry()
	registry.start('dl-5', url='https://example.com/x')

	outcome = registry.fail('dl-5', stage='browser_fetch', error='HTTP 403')

	assert outcome.phase is DownloadPhase.FAILED
	assert outcome.stage == 'browser_fetch'
	assert outcome.error == 'HTTP 403'
	assert outcome.terminal is True


def test_failure_without_prior_request_is_recorded():
	registry = DownloadRegistry()

	outcome = registry.fail('dl-6', stage='pdf_probe', error='no data')

	assert outcome.phase is DownloadPhase.FAILED
	assert outcome.stage == 'pdf_probe'


def test_cancel_is_terminal_and_reports_detail():
	registry = DownloadRegistry()
	registry.start('dl-7')

	outcome = registry.cancel('dl-7', detail='browser stopped')

	assert outcome.phase is DownloadPhase.CANCELLED
	assert outcome.terminal is True
	assert outcome.detail == 'browser stopped'


def test_history_is_bounded():
	registry = DownloadRegistry(max_outcomes=3)
	for index in range(5):
		registry.complete(f'dl-{index}', path=f'/tmp/{index}.bin', file_size=1)

	ids = [outcome.download_id for outcome in registry.outcomes()]

	assert ids == ['dl-2', 'dl-3', 'dl-4']


def test_latest_can_filter_by_url_and_terminal_state():
	registry = DownloadRegistry()
	registry.complete('dl-a', path='/tmp/a', url='https://example.com/a')
	registry.start('dl-b', url='https://example.com/b')

	assert registry.latest(url='https://example.com/a').download_id == 'dl-a'
	assert registry.latest(terminal=True).download_id == 'dl-a'
	assert registry.latest(url='https://example.com/b').phase is DownloadPhase.STARTED
	assert len(registry.pending()) == 1


def test_trim_never_evicts_a_pending_attempt():
	registry = DownloadRegistry(max_outcomes=2)
	registry.start('pending-1')
	registry.start('pending-2')
	registry.complete('finished', path='/tmp/f.bin', file_size=1)

	ids = {outcome.download_id for outcome in registry.outcomes()}

	assert {'pending-1', 'pending-2'} <= ids
	# The newest terminal record is retained so a waiter can still observe it.
	assert 'finished' in ids


def test_redacted_outcome_strips_query_and_caps_text():
	payload = {
		'download_id': 'url:https://example.com/file?token=secret:' + 'x' * 400,
		'url': 'https://user:password@example.com/vault/file.pdf?X-Amz-Signature=secret&token=abc',
		'path': '/tmp/' + 'p' * 800,
		'error': 'e' * 900,
		'detail': None,
		'phase': 'failed',
	}

	redacted = redacted_outcome(payload)

	assert 'download_id' not in redacted
	assert len(redacted['download_ref']) == 12
	assert 'secret' not in redacted['url']
	assert 'password' not in redacted['url']
	assert redacted['url'].startswith('https://example.com/')
	assert redacted['url'].endswith('?redacted')
	assert len(redacted['path']) <= 303
	assert len(redacted['error']) <= 303
	assert redacted['phase'] == 'failed'


@pytest.mark.asyncio
async def test_wait_returns_a_terminal_outcome_created_after_the_call():
	registry = DownloadRegistry()

	async def finish_later():
		await asyncio.sleep(0.05)
		registry.complete('dl-wait', path='/tmp/wait.bin', file_size=3)

	waiter = asyncio.create_task(registry.wait_for_terminal(timeout=2.0))
	await finish_later()

	outcome = await waiter
	assert outcome.download_id == 'dl-wait'
	assert outcome.phase is DownloadPhase.COMPLETED


@pytest.mark.asyncio
async def test_wait_ignores_an_older_terminal_outcome():
	registry = DownloadRegistry()
	registry.complete('dl-old', path='/tmp/old.bin', file_size=1)

	async def finish_later():
		await asyncio.sleep(0.05)
		registry.complete('dl-new', path='/tmp/new.bin', file_size=2)

	waiter = asyncio.create_task(registry.wait_for_terminal(timeout=2.0))
	await finish_later()

	outcome = await waiter
	assert outcome.download_id == 'dl-new'


@pytest.mark.asyncio
async def test_wait_returns_a_pending_outcome_that_finishes_after_the_call():
	registry = DownloadRegistry()
	registry.start('dl-already-pending', url='https://example.com/p.bin')

	async def finish_later():
		await asyncio.sleep(0.05)
		registry.complete('dl-already-pending', path='/tmp/p.bin', file_size=4)

	waiter = asyncio.create_task(registry.wait_for_terminal(timeout=2.0))
	await finish_later()

	outcome = await waiter
	assert outcome.download_id == 'dl-already-pending'


@pytest.mark.asyncio
async def test_url_wait_does_not_return_a_previous_terminal_outcome():
	registry = DownloadRegistry()
	registry.complete('dl-old', path='/tmp/old.bin', url='https://example.com/same')
	registry.start('dl-new', url='https://example.com/same')

	async def finish_later():
		await asyncio.sleep(0.05)
		registry.complete('dl-new', path='/tmp/new.bin', file_size=2, url='https://example.com/same')

	waiter = asyncio.create_task(registry.wait_for_terminal(timeout=2.0, url='https://example.com/same'))
	await finish_later()

	outcome = await waiter
	assert outcome.download_id == 'dl-new'


@pytest.mark.asyncio
async def test_wait_can_match_a_specific_url():
	registry = DownloadRegistry()

	async def finish_later():
		await asyncio.sleep(0.05)
		registry.complete('dl-url', path='/tmp/u.bin', file_size=1, url='https://example.com/u')

	waiter = asyncio.create_task(registry.wait_for_terminal(timeout=2.0, url='https://example.com/u'))
	await finish_later()

	outcome = await waiter
	assert outcome.download_id == 'dl-url'


@pytest.mark.asyncio
async def test_wait_timeout_reports_the_last_observed_phase():
	registry = DownloadRegistry()
	registry.start('dl-stuck', url='https://example.com/stuck')

	with pytest.raises(DownloadTimeoutError) as exc_info:
		await registry.wait_for_terminal(timeout=0.15)

	error = exc_info.value
	assert error.outcome is not None
	assert error.outcome['phase'] == 'started'
	assert error.outcome['url'] == 'https://example.com/stuck'


def test_to_dict_is_json_safe():
	registry = DownloadRegistry()
	registry.complete('dl-json', path='/tmp/j.bin', file_size=9, file_name='j.bin', file_type='bin')

	payload = registry.get('dl-json').to_dict()
	encoded = json.dumps(payload)

	assert payload['phase'] == 'completed'
	assert payload['download_id'] == 'dl-json'
	assert 'completed' in encoded


def test_outcome_is_not_shared_between_instances():
	first = DownloadOutcome(download_id='a', phase=DownloadPhase.REQUESTED)
	second = DownloadOutcome(download_id='b', phase=DownloadPhase.REQUESTED)

	first.path = '/tmp/mutated'

	assert second.path is None


@pytest.mark.asyncio
async def test_wait_accepts_a_recent_terminal_outcome_with_lookback():
	registry = DownloadRegistry()
	registry.complete('dl-recent', path='/tmp/recent.bin', file_size=3)

	outcome = await registry.wait_for_terminal(timeout=0.5, lookback=5.0)

	assert outcome.download_id == 'dl-recent'


@pytest.mark.asyncio
async def test_wait_rejects_an_old_terminal_outcome_with_zero_lookback():
	registry = DownloadRegistry()
	registry.complete('dl-old', path='/tmp/old.bin', file_size=3)

	with pytest.raises(DownloadTimeoutError):
		await registry.wait_for_terminal(timeout=0.15, lookback=0.0)


def test_trim_keeps_the_just_completed_attempt():
	registry = DownloadRegistry(max_outcomes=3, max_active=10)
	registry.start('p1')
	registry.start('p2')
	registry.start('p3')
	registry.complete('t1', path='/tmp/t1', file_size=1)
	registry.complete('t2', path='/tmp/t2', file_size=1)

	registry.complete('p1', path='/tmp/p1', file_size=1)

	ids = {outcome.download_id for outcome in registry.outcomes()}
	assert 'p1' in ids
	assert len(ids) <= 3


def test_active_capacity_fails_the_oldest_pending_attempt():
	registry = DownloadRegistry(max_outcomes=10, max_active=2)
	registry.start('p1')
	registry.start('p2')
	registry.start('p3')

	phases = {outcome.download_id: outcome.phase for outcome in registry.outcomes()}

	assert phases['p1'] is DownloadPhase.FAILED
	assert phases['p2'] is DownloadPhase.STARTED
	assert phases['p3'] is DownloadPhase.STARTED
	assert registry.get('p1').stage == 'capacity'


def test_redacted_outcome_handles_malformed_and_non_string_urls():
	malformed = redacted_outcome({'url': 'http://[', 'phase': 'failed'})
	non_string = redacted_outcome({'url': 123, 'phase': 'failed'})

	assert malformed['url'] == '<invalid-url>'
	assert non_string['url'] == '<redacted>'


def test_cancelled_is_final():
	registry = DownloadRegistry()
	registry.cancel('dl-cancel', detail='user cancelled')

	outcome = registry.complete('dl-cancel', path='/tmp/late.bin', file_size=3)

	assert outcome.phase is DownloadPhase.CANCELLED
	assert outcome.path is None


def test_active_count_tracks_only_pending_attempts():
	registry = DownloadRegistry()
	registry.start('p1')
	registry.start('p2')
	registry.complete('done', path='/tmp/d.bin', file_size=1)

	assert registry.active_count() == 2


@pytest.mark.asyncio
async def test_a_wait_pins_the_outcome_it_targets():
	registry = DownloadRegistry(max_outcomes=1)
	waiter = asyncio.create_task(
		registry.wait_for_terminal(timeout=2.0, download_id='a', poll_interval=0.2)
	)
	await asyncio.sleep(0.05)
	registry.complete('a', path='/tmp/a.bin', file_size=1)
	registry.complete('b', path='/tmp/b.bin', file_size=1)

	outcome = await waiter

	assert outcome.download_id == 'a'


@pytest.mark.asyncio
async def test_wait_releases_pinned_history_after_it_returns():
	registry = DownloadRegistry(max_outcomes=1)
	waiter = asyncio.create_task(
		registry.wait_for_terminal(timeout=2.0, download_id='target', poll_interval=0.05)
	)
	await asyncio.sleep(0.02)
	registry.complete('target', path='/tmp/target.bin', file_size=1)
	for index in range(5):
		registry.complete(f'other-{index}', path=f'/tmp/o{index}.bin', file_size=1)

	outcome = await waiter

	assert outcome.download_id == 'target'
	# The waiter has released its pin, so history is bounded again.
	assert len(registry.outcomes()) <= 1


@pytest.mark.asyncio
async def test_lookback_is_ignored_while_a_new_attempt_is_pending():
	registry = DownloadRegistry()
	registry.complete('old', path='/tmp/old.bin', file_size=1, url='https://example.com/same')
	registry.start('new', url='https://example.com/same')

	async def finish_later():
		await asyncio.sleep(0.05)
		registry.complete('new', path='/tmp/new.bin', file_size=2, url='https://example.com/same')

	waiter = asyncio.create_task(
		registry.wait_for_terminal(timeout=2.0, url='https://example.com/same', lookback=5.0)
	)
	await finish_later()

	outcome = await waiter
	assert outcome.download_id == 'new'


def test_redaction_strips_queries_from_file_urls_and_paths():
	file_url = redacted_outcome({'url': 'file:///tmp/report?token=secret', 'phase': 'completed'})
	path_like = redacted_outcome({'url': '/tmp/report?token=secret', 'phase': 'completed'})

	assert 'secret' not in file_url['url']
	assert file_url['url'].endswith('?redacted')
	assert 'secret' not in path_like['url']
	assert path_like['url'].endswith('?redacted')


def test_capacity_rejection_does_not_fail_an_admitted_attempt():
	registry = DownloadRegistry(max_outcomes=10, max_active=2)
	registry.start('admitted', url='https://example.com/a')

	rejected = registry.reject(
		'rejected',
		stage='capacity',
		error='Too many active downloads (limit 2)',
		url='https://example.com/b',
	)

	assert rejected.phase is DownloadPhase.FAILED
	assert registry.get('admitted').phase is DownloadPhase.STARTED


def test_capacity_rejection_cannot_be_upgraded_by_a_late_completion():
	registry = DownloadRegistry()
	registry.reject('rejected', stage='capacity', error='too many')

	outcome = registry.complete('rejected', path='/tmp/late.bin', file_size=3)

	assert outcome.phase is DownloadPhase.FAILED
	assert outcome.stage == 'capacity'
	assert outcome.path is None


def test_redacted_outcome_scrubs_urls_inside_error_text():
	payload = {
		'error': 'Requests failed for https://user:pass@example.com/file?token=secret with 403',
		'detail': 'see https://example.com/other?X-Amz-Signature=abc',
	}

	redacted = redacted_outcome(payload)

	assert 'secret' not in redacted['error']
	assert 'pass' not in redacted['error']
	assert 'X-Amz-Signature' not in redacted['detail']
	assert 'https://example.com/file?redacted' in redacted['error']
