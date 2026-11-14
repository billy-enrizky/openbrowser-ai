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

	registry.sync_from_pages([{'targetId': target, 'url': 'https://new.example', 'title': 'New'}])

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
