"""Behavior tests for the explicit namespace transfer policy (F6)."""

import json

from openbrowser.code_use.namespace_transfer import (
	NamespaceValueKind,
	transfer_namespace,
)


class FakeBrowserSession:
	"""Stand-in for the old BrowserSession value stored in a namespace."""

	_cdp_client_root = object()


class OpaquePage:
	"""Stand-in for a page or CDP object that must not be carried."""


def test_scalars_and_nested_containers_are_preserved():
	old = {
		'count': 3,
		'name': 'report',
		'flag': True,
		'ratio': 1.5,
		'blob': b'bytes',
		'items': [1, 'two', {'three': 3}],
		'pair': (1, 2),
		'tags': {'a', 'b'},
	}

	result = transfer_namespace(old)

	assert result.variables['count'] == 3
	assert result.variables['name'] == 'report'
	assert result.variables['items'] == [1, 'two', {'three': 3}]
	assert result.variables['pair'] == (1, 2)
	assert result.variables['tags'] == {'a', 'b'}
	assert result.dropped == {}
	assert result.warnings == []


def test_preserved_containers_are_copies():
	old = {'items': [1, 2], 'mapping': {'a': 1}}
	result = transfer_namespace(old)

	result.variables['items'].append(3)
	result.variables['mapping']['a'] = 99

	assert old['items'] == [1, 2]
	assert old['mapping'] == {'a': 1}


def test_framework_names_are_rebuilt_and_not_carried():
	old = {
		'browser': FakeBrowserSession(),
		'file_system': object(),
		'navigate': object(),
		'json': object(),
		'PdfReader': object(),
		'user_value': 7,
	}

	result = transfer_namespace(old)

	assert result.variables == {'user_value': 7}
	assert 'browser' in result.framework_names
	assert 'file_system' in result.framework_names
	assert 'navigate' in result.framework_names
	assert 'json' in result.framework_names
	assert 'PdfReader' in result.framework_names
	assert result.dropped == {}


def test_direct_old_browser_value_is_dropped_with_a_stale_reason():
	old_browser = FakeBrowserSession()
	old = {'handle': old_browser, 'count': 1}

	result = transfer_namespace(old, old_browser_session=old_browser)

	assert result.variables == {'count': 1}
	assert 'handle' in result.dropped
	assert 'previous browser session' in result.dropped['handle']
	assert len(result.warnings) == 1
	assert "'handle'" in result.warnings[0]


def test_container_holding_the_old_browser_is_dropped():
	old_browser = FakeBrowserSession()
	old = {'pages': [old_browser], 'count': 1}

	result = transfer_namespace(old, old_browser_session=old_browser)

	assert result.variables == {'count': 1}
	assert 'pages' in result.dropped
	assert 'previous browser session' in result.dropped['pages']


def test_nested_dict_holding_a_page_object_is_opaque():
	old = {'state': {'page': OpaquePage()}}

	result = transfer_namespace(old)

	assert result.variables == {}
	assert 'state' in result.dropped
	assert 'non-scalar' in result.dropped['state'] or 'unsupported' in result.dropped['state']


def test_opaque_objects_are_dropped_with_a_type_reason():
	result = transfer_namespace({'page': OpaquePage()})

	assert result.variables == {}
	assert 'OpaquePage' in result.dropped['page']


def test_cycles_are_dropped_without_recursing_forever():
	cyclic = []
	cyclic.append(cyclic)

	result = transfer_namespace({'cycle': cyclic})

	assert result.variables == {}
	assert 'cycle' in result.dropped


def test_private_names_are_skipped():
	result = transfer_namespace({'_task_done': True, '__internal': 1, 'visible': 2})

	assert result.variables == {'visible': 2}
	assert result.dropped == {}
	assert result.framework_names == []


def test_kinds_and_summary_are_json_safe():
	result = transfer_namespace({'count': 1, 'browser': object()})

	payload = result.to_dict()

	assert payload['preserved'] == ['count']
	assert 'browser' in payload['framework']
	assert payload['dropped'] == {}
	assert NamespaceValueKind.SCALAR.value == 'scalar'
	json.dumps(payload)


def test_warnings_are_bounded():
	old = {f'value_{index}': OpaquePage() for index in range(50)}

	result = transfer_namespace(old)

	assert len(result.warnings) <= 20
	assert len(result.dropped) == 50


def test_user_defined_functions_and_classes_are_opaque():
	class UserType:
		pass

	def user_function():
		return 1

	result = transfer_namespace({'UserType': UserType, 'user_function': user_function})

	assert result.variables == {}
	assert 'UserType' in result.dropped
	assert 'user_function' in result.dropped
	assert len(result.warnings) == 2


def test_framework_functions_are_classified_as_framework():
	from openbrowser.code_use import namespace as namespace_module

	result = transfer_namespace({'evaluate': namespace_module.evaluate})

	assert 'evaluate' in result.framework_names
	assert result.dropped == {}


def test_shared_container_aliases_are_preserved():
	shared = [1, 2]
	result = transfer_namespace({'left': shared, 'right': shared})

	assert result.variables['left'] is result.variables['right']
	result.variables['left'].append(3)
	assert result.variables['right'] == [1, 2, 3]


def test_scalar_subclass_with_extra_state_is_opaque():
	old_browser = FakeBrowserSession()

	class TrackingInt(int):
		pass

	value = TrackingInt(3)
	value.old_browser = old_browser
	result = transfer_namespace({'tracking': value}, old_browser_session=old_browser)

	assert result.variables == {}
	assert 'tracking' in result.dropped


def test_unknown_modules_are_opaque():
	import math

	result = transfer_namespace({'math': math})

	assert result.variables == {}
	assert 'math' in result.dropped
	assert len(result.warnings) == 1


def test_status_summary_is_bounded_for_large_namespaces():
	old = {f'value_{index}': OpaquePage() for index in range(50)}
	result = transfer_namespace(old)

	payload = result.to_dict()

	assert payload['dropped_count'] == 50
	assert len(payload['dropped']) == 20
	assert payload['dropped_truncated'] is True


def test_framework_module_under_a_custom_name_is_opaque():
	import json as json_module

	result = transfer_namespace({'j': json_module})

	assert result.variables == {}
	assert 'j' in result.dropped


def test_container_subclass_is_opaque():
	class WeirdList(list):
		pass

	result = transfer_namespace({'items': WeirdList([1, 2])})

	assert result.variables == {}
	assert 'items' in result.dropped


def test_summary_truncates_long_names():
	long_name = 'x' * 100_000
	result = transfer_namespace({long_name: 1})

	payload = result.to_dict()

	assert len(payload['preserved'][0]) <= 123
	assert payload['preserved_count'] == 1
