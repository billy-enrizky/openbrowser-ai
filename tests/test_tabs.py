"""Tests for explicit tab identity and state."""

import pytest

from openbrowser.browser.tabs import TabError, TabRegistry, TabState


def test_register_and_snapshot():
	registry = TabRegistry()
	registry.register(
		'ABCDEF1234567890ABCDEF1234567890',
		url='https://a.example',
		title='A',
		state=TabState.OPEN,
	)

	snapshot = registry.snapshot(active_target_id='ABCDEF1234567890ABCDEF1234567890')
	assert len(snapshot) == 1
	assert snapshot[0]['target_id'] == 'ABCDEF1234567890ABCDEF1234567890'
	assert snapshot[0]['short_id'] == '7890'
	assert snapshot[0]['url'] == 'https://a.example'
	assert snapshot[0]['state'] == 'open'
	assert snapshot[0]['active'] is True


def test_resolve_full_and_short_ids():
	registry = TabRegistry()
	registry.register('ABCDEF1234567890ABCDEF1234567890')

	assert registry.resolve('ABCDEF1234567890ABCDEF1234567890') == 'ABCDEF1234567890ABCDEF1234567890'
	assert registry.resolve('7890') == 'ABCDEF1234567890ABCDEF1234567890'


def test_resolve_rejects_ambiguous_short_id():
	registry = TabRegistry()
	first = 'AAAABBBBCCCCDDDD1111222233337890'
	second = 'EEEEFFFF000011112222333344447890'
	registry.register(first)
	registry.register(second)

	with pytest.raises(TabError) as error:
		registry.resolve('7890')

	assert set(error.value.candidates) == {first, second}
	assert 'full CDP target_id' in (error.value.hint or '')


def test_resolve_rejects_unknown_id_with_hint():
	registry = TabRegistry()
	registry.register('AAAABBBBCCCCDDDD1111222233334444')

	with pytest.raises(TabError) as error:
		registry.resolve('9999')

	assert 'No live tab' in str(error.value)
	assert error.value.candidates == []
	assert 'tabs list' in (error.value.hint or '')


def test_closed_tab_is_not_a_live_candidate():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233337890'
	registry.register(target)
	registry.mark_closed(target)

	with pytest.raises(TabError):
		registry.resolve('7890')


def test_sync_from_pages_closes_missing_tabs():
	registry = TabRegistry()
	first = 'AAAABBBBCCCCDDDD1111222233331111'
	second = 'AAAABBBBCCCCDDDD1111222233332222'
	registry.register(first)
	registry.register(second)

	registry.sync_from_pages([{'targetId': first, 'url': 'https://a.example', 'title': 'A'}])

	snapshot = {item['target_id']: item for item in registry.snapshot()}
	assert snapshot[first]['state'] == 'open'
	assert snapshot[second]['state'] == 'closed'


def test_known_closed_resolves_a_unique_suffix_and_an_exact_full_id():
	registry = TabRegistry()
	first = 'AAAABBBBCCCCDDDD111122223333FF34'
	second = 'AAAABBBBCCCCDDDD111122223333EE34'
	registry.register(first, state=TabState.CLOSED)
	registry.register(second, state=TabState.CLOSED)

	assert registry.known_closed('FF34') == first
	assert registry.known_closed(first) == first
	assert registry.known_closed('zzzz') is None


def test_known_closed_refuses_an_ambiguous_suffix():
	registry = TabRegistry()
	registry.register('AAAABBBBCCCCDDDD111122223333FF34', state=TabState.CLOSED)
	registry.register('BBBBCCCCDDDDEEEE111122223333FF34', state=TabState.CLOSED)

	assert registry.known_closed('FF34') is None


def test_register_applies_an_explicit_empty_url_and_title():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233331234'
	registry.register(target, url='https://old.example', title='Old')

	registry.register(target, url='', title='')

	record = registry.snapshot()[0]
	assert record['url'] == ''
	assert record['title'] == ''


def test_sync_from_pages_reopens_existing_tab():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233331234'
	registry.register(target, url='https://old.example')
	registry.mark_closed(target)

	registry.sync_from_pages(
		[{'targetId': target, 'url': 'https://new.example', 'title': 'New'}],
		allow_reopen=True,
	)

	record = registry.snapshot()[0]
	assert record['state'] == 'open'
	assert record['url'] == 'https://new.example'
	assert record['title'] == 'New'


def test_sync_from_pages_applies_an_empty_title():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233331234'
	registry.register(target, title='Old title')

	registry.sync_from_pages([{'targetId': target, 'url': 'https://new.example', 'title': ''}])

	assert registry.snapshot()[0]['title'] == ''


def test_long_suffix_is_not_treated_as_a_short_id():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233334444'
	registry.register(target)

	with pytest.raises(TabError):
		registry.resolve('334444')


def test_mark_crashed_keeps_tab_resolvable():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233335678'
	registry.register(target)
	registry.mark_crashed(target)

	assert registry.resolve('5678') == target
	assert registry.snapshot()[0]['state'] == TabState.CRASHED.value


def test_clear_removes_all_tabs():
	registry = TabRegistry()
	registry.register('AAAABBBBCCCCDDDD1111222233331234')
	registry.clear()

	assert len(registry) == 0
	assert registry.snapshot() == []


def test_closing_survives_register_attach_and_sync():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233331234'
	registry.register(target, state=TabState.OPEN)
	registry.mark_closing(target)

	registry.register(target, url='https://new.example', state=TabState.OPEN)
	assert registry.snapshot()[0]['state'] == 'closing'

	registry.sync_from_pages([{'targetId': target, 'type': 'page', 'url': 'https://new.example'}])
	assert registry.snapshot()[0]['state'] == 'closing'


def test_closed_and_crashed_reopen_only_via_update_allow_reopen():
	registry = TabRegistry()
	closed = 'AAAABBBBCCCCDDDD1111222233331111'
	crashed = 'AAAABBBBCCCCDDDD1111222233332222'
	registry.register(closed, state=TabState.CLOSED)
	registry.register(crashed, state=TabState.CRASHED)

	registry.update(closed, state=TabState.OPEN)
	registry.update(crashed, state=TabState.OPEN)
	assert registry.get(closed).state is TabState.CLOSED
	assert registry.get(crashed).state is TabState.CRASHED

	registry.update(closed, state=TabState.OPEN, allow_reopen=True)
	registry.update(crashed, state=TabState.OPEN, allow_reopen=True)
	assert registry.get(closed).state is TabState.OPEN
	assert registry.get(crashed).state is TabState.OPEN


def test_crash_error_is_bounded_in_snapshot():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233333333'
	registry.register(target, state=TabState.OPEN)
	registry.mark_crashed(target, 'x' * 400)

	record = registry.snapshot()[0]
	assert record['state'] == 'crashed'
	assert record['error'].endswith('...')
	assert len(record['error']) == 300


def test_live_page_candidates_excludes_protected_states_and_orders_by_creation():
	registry = TabRegistry()
	first = 'AAAABBBBCCCCDDDD1111222233331111'
	second = 'AAAABBBBCCCCDDDD1111222233332222'
	crashed = 'AAAABBBBCCCCDDDD1111222233333333'
	closing = 'AAAABBBBCCCCDDDD1111222233334444'
	registry.register(first, state=TabState.OPEN)
	registry.register(second, state=TabState.OPEN)
	registry.register(crashed, state=TabState.CRASHED)
	registry.register(closing, state=TabState.CLOSING)

	pages = [
		{'targetId': closing, 'type': 'page'},
		{'targetId': crashed, 'type': 'page'},
		{'targetId': second, 'type': 'page'},
		{'targetId': first, 'type': 'page'},
		{'targetId': 'IFRAME-TARGET-0000', 'type': 'iframe'},
	]
	candidates = registry.live_page_candidates(pages)

	assert [record.target_id for record in candidates] == [first, second]


def test_cold_reconciliation_breaks_equal_timestamps_by_target_id():
	registry = TabRegistry()
	first = 'AAAABBBBCCCCDDDD1111222233331111'
	second = 'AAAABBBBCCCCDDDD1111222233332222'

	candidates = registry.live_page_candidates(
		[
			{'targetId': second, 'type': 'page'},
			{'targetId': first, 'type': 'page'},
		]
	)

	assert [record.target_id for record in candidates] == [first, second]
	assert registry.get(first).created_at == registry.get(second).created_at


def test_mark_closed_retires_a_crashed_record():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233335555'
	registry.register(target, state=TabState.OPEN)
	registry.mark_crashed(target, 'crashed')

	registry.mark_closed(target)

	record = registry.get(target)
	assert record is not None and record.state is TabState.CLOSED


def test_sync_from_pages_retires_missing_crashed_only_when_complete():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233336666'
	registry.register(target, state=TabState.OPEN)
	registry.mark_crashed(target, 'crashed')

	registry.sync_from_pages([], complete=False)
	assert registry.get(target).state is TabState.CRASHED

	registry.sync_from_pages([])
	assert registry.get(target).state is TabState.CLOSED


def test_reopen_clears_the_crash_error():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD1111222233337777'
	registry.register(target, state=TabState.OPEN)
	registry.mark_crashed(target, 'crashed')

	registry.update(target, state=TabState.OPEN, allow_reopen=True)

	record = registry.get(target)
	assert record is not None and record.state is TabState.OPEN
	assert record.error is None


def test_filtered_live_page_candidates_do_not_close_omitted_tabs():
	registry = TabRegistry()
	visible = 'AAAABBBBCCCCDDDD1111222233338888'
	settings = 'AAAABBBBCCCCDDDD1111222233339999'
	registry.register(visible, state=TabState.OPEN)
	registry.register(settings, url='chrome://settings', state=TabState.OPEN)

	candidates = registry.live_page_candidates([{'targetId': visible, 'type': 'page'}])

	assert [record.target_id for record in candidates] == [visible]
	assert registry.get(settings).state is TabState.OPEN


def test_mark_closing_increments_the_close_attempt_token():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD111122223333ABCD'
	registry.register(target, state=TabState.OPEN)

	registry.mark_closing(target)
	first = registry.get(target).close_attempt
	registry.mark_closing(target)

	assert registry.get(target).close_attempt == first + 1


def test_authoritative_sync_reopens_closed_but_preserves_protected_states():
	registry = TabRegistry()
	closed = 'AAAABBBBCCCCDDDD111122223333AAAA'
	closing = 'AAAABBBBCCCCDDDD111122223333BBBB'
	crashed = 'AAAABBBBCCCCDDDD111122223333CCCC'
	registry.register(closed, state=TabState.OPEN)
	registry.register(closing, state=TabState.OPEN)
	registry.register(crashed, state=TabState.OPEN)
	registry.mark_closed(closed)
	registry.mark_closing(closing)
	registry.mark_crashed(crashed, 'crashed')

	registry.sync_from_pages(
		[
			{'targetId': closed, 'type': 'page'},
			{'targetId': closing, 'type': 'page'},
			{'targetId': crashed, 'type': 'page'},
		],
		allow_reopen=True,
	)

	assert registry.get(closed).state is TabState.OPEN
	assert registry.get(closing).state is TabState.CLOSING
	assert registry.get(crashed).state is TabState.CRASHED


def test_stale_page_enumeration_cannot_reopen_a_newer_closed_state():
	registry = TabRegistry()
	target = 'AAAABBBBCCCCDDDD111122223333DDDD'
	registry.register(target, state=TabState.OPEN)
	stale_revisions = registry.state_revisions()
	registry.mark_closed(target)

	registry.sync_from_pages(
		[{'targetId': target, 'type': 'page'}],
		allow_reopen=True,
		state_revisions=stale_revisions,
	)
	assert registry.get(target).state is TabState.CLOSED

	registry.sync_from_pages(
		[{'targetId': target, 'type': 'page'}],
		allow_reopen=True,
		state_revisions=registry.state_revisions(),
	)
	assert registry.get(target).state is TabState.OPEN
