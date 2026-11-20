"""Behavior tests for explicit element-index readiness (F13)."""

from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from openbrowser.models import ActionResult
from openbrowser.tools.action_readiness import (
	ActionReadinessCode,
	ActionReadinessError,
	require_element_index,
	validate_index_value,
)
from openbrowser.tools.service import Tools
from openbrowser.tools.views import (
	ClickElementAction,
	GetDropdownOptionsAction,
	InputTextAction,
	SelectDropdownOptionAction,
	UploadFileAction,
)

INDEXED_MODELS = (
	ClickElementAction,
	InputTextAction,
	UploadFileAction,
	GetDropdownOptionsAction,
	SelectDropdownOptionAction,
)


def _map(*indexes):
	return {index: object() for index in indexes}


class _FakeBrowser:
	"""A browser session stub for the early readiness return paths."""

	is_local = False
	downloaded_files: list[str] = []
	current_target_id = None
	cdp_client = object()

	def __init__(self, selector_map):
		self._selector_map = selector_map

	async def get_selector_map(self):
		return self._selector_map


def _action(name, tools_class=Tools):
	return tools_class().registry.registry.actions[name].function


# ---------------------------------------------------------------------------
# validate_index_value
# ---------------------------------------------------------------------------


def test_missing_index_is_a_typed_error_with_a_refresh_hint():
	with pytest.raises(ActionReadinessError) as exc_info:
		require_element_index(None, selector_map=_map(1, 2), action='click')

	error = exc_info.value
	assert isinstance(error, ValueError)
	assert error.code is ActionReadinessCode.MISSING
	message = str(error)
	assert 'element_index_missing' in message
	assert 'click()' in message
	assert 'get_status' in message
	assert 'get_browser_state_summary' in message


@pytest.mark.parametrize('index', [0, -1])
def test_zero_and_negative_indexes_are_invalid(index):
	with pytest.raises(ActionReadinessError) as exc_info:
		require_element_index(index, selector_map=_map(1), action='click')

	assert exc_info.value.code is ActionReadinessCode.INVALID
	assert '1-based' in str(exc_info.value)


@pytest.mark.parametrize('index', [3.5, 'three', object()])
def test_non_integer_indexes_are_invalid(index):
	with pytest.raises(ActionReadinessError) as exc_info:
		require_element_index(index, selector_map=_map(3), action='input_text')

	assert exc_info.value.code is ActionReadinessCode.INVALID
	assert type(index).__name__ in str(exc_info.value)


@pytest.mark.parametrize('index', [True, False])
def test_boolean_indexes_are_rejected_explicitly(index):
	with pytest.raises(ActionReadinessError) as exc_info:
		validate_index_value(index, action='click')

	assert exc_info.value.code is ActionReadinessCode.INVALID
	assert 'boolean' in str(exc_info.value)


def test_rejected_values_are_not_echoed_in_full():
	with pytest.raises(ActionReadinessError) as exc_info:
		validate_index_value('x' * 5000, action='click')

	message = str(exc_info.value)
	assert 'chars)' in message
	assert message.count('x') < 200
	assert len(message) < 700


def test_long_browser_state_labels_are_bounded():
	with pytest.raises(ActionReadinessError) as exc_info:
		validate_index_value('i_' + '9' * 1000, action='click')

	message = str(exc_info.value)
	assert 'the bare integer' in message
	assert len(message) < 700


@pytest.mark.parametrize('label', ['i_3', '[i_3]'])
def test_browser_state_labels_name_the_integer_to_pass(label):
	with pytest.raises(ActionReadinessError) as exc_info:
		validate_index_value(label, action='click')

	assert exc_info.value.code is ActionReadinessCode.INVALID
	assert 'Pass 3 without' in str(exc_info.value)


@pytest.mark.parametrize('index', [3, 3.0, '3', ' 3 ', '3.0', '+3', '1_0'])
def test_numeric_forms_previous_pydantic_accepted_still_work(index):
	expected = 10 if index == '1_0' else 3
	assert validate_index_value(index, action='click') == expected


class _PlainIndexModel(BaseModel):
	index: int


@pytest.mark.parametrize(
	'index',
	[3, 3.0, '3', ' 3 ', '+3', '3.0', '1_0', Decimal('3'), '1e3', '0x10', '\u0663', 3.5, 1e20, 'three'],
)
def test_readiness_acceptance_matches_a_plain_pydantic_int(index):
	try:
		pydantic_value = _PlainIndexModel(index=index).index
	except ValidationError:
		pydantic_value = None

	try:
		readiness_value = validate_index_value(index, action='click')
	except ActionReadinessError:
		readiness_value = None

	assert (readiness_value is None) == (pydantic_value is None)
	if pydantic_value is not None:
		assert readiness_value == pydantic_value


def test_empty_selector_map_is_state_not_ready():
	with pytest.raises(ActionReadinessError) as exc_info:
		require_element_index(1, selector_map={}, action='click')

	assert exc_info.value.code is ActionReadinessCode.STATE_NOT_READY
	assert 'get_browser_state_summary' in str(exc_info.value)


def test_stale_index_lists_available_indexes():
	with pytest.raises(ActionReadinessError) as exc_info:
		require_element_index(9, selector_map=_map(1, 2, 5), action='click')

	error = exc_info.value
	assert error.code is ActionReadinessCode.STALE
	assert 'index 9 is not in the current browser state' in str(error)
	assert '1, 2, 5' in str(error)
	assert 'get_status' in str(error)


def test_available_index_list_is_bounded():
	selector_map = _map(*range(1, 51))

	with pytest.raises(ActionReadinessError) as exc_info:
		require_element_index(99, selector_map=selector_map, action='click')

	assert 'showing 10 of 50' in str(exc_info.value)


def test_valid_index_is_returned_and_normalized():
	assert require_element_index(2, selector_map=_map(1, 2), action='click') == 2
	assert require_element_index('2', selector_map=_map(1, 2), action='click') == 2


# ---------------------------------------------------------------------------
# Model-level prevalidation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('model', INDEXED_MODELS)
@pytest.mark.parametrize('index', [None, 0, -1, 1.5, 'three', True])
def test_action_models_surface_the_readiness_message(model, index):
	kwargs = {'index': index}
	for field in ('text', 'path'):
		if field in model.model_fields:
			kwargs[field] = 'value'

	with pytest.raises(ValidationError) as exc_info:
		model(**kwargs)

	message = str(exc_info.value)
	assert 'Action readiness' in message
	assert 'get_browser_state_summary' in message


def test_action_models_keep_accepting_numeric_strings():
	assert ClickElementAction(index='3').index == 3
	assert ClickElementAction(index='3.0').index == 3


@pytest.mark.parametrize('model', INDEXED_MODELS)
def test_omitted_index_is_a_readiness_error(model):
	kwargs = {}
	for field in ('text', 'path'):
		if field in model.model_fields:
			kwargs[field] = 'value'

	with pytest.raises(ValidationError) as exc_info:
		model(**kwargs)

	message = str(exc_info.value)
	assert 'element_index_missing' in message
	assert 'get_status' in message


def test_json_validation_uses_the_same_readiness_contract():
	assert ClickElementAction.model_validate_json('{"index": 3}').index == 3

	with pytest.raises(ValidationError) as exc_info:
		ClickElementAction.model_validate_json('{"index": null}')

	message = str(exc_info.value)
	assert 'element_index_missing' in message
	assert 'get_status' in message


def test_scroll_keeps_its_zero_sentinel_and_rejects_labels():
	from openbrowser.tools.views import ScrollAction

	assert ScrollAction(down=True).index is None
	assert ScrollAction(down=True, index=0).index == 0
	assert ScrollAction(down=True, index=0.0).index == 0
	assert ScrollAction(down=True, index='0').index == 0
	assert ScrollAction(down=True, index='-0').index == 0
	assert ScrollAction(down=True, index=Decimal('0')).index == 0
	assert ScrollAction(down=True, index=3).index == 3

	with pytest.raises(ValidationError) as exc_info:
		ScrollAction(down=True, index='i_3')

	assert 'Action readiness' in str(exc_info.value)
	assert 'Pass 3 without' in str(exc_info.value)


def test_descriptions_explain_the_readiness_contract():
	from openbrowser.code_use.descriptions import (
		EXECUTE_CODE_DESCRIPTION,
		EXECUTE_CODE_DESCRIPTION_COMPACT,
	)

	for description in (EXECUTE_CODE_DESCRIPTION, EXECUTE_CODE_DESCRIPTION_COMPACT):
		assert 'ActionReadinessError' in description
		assert 'latest browser state' in description


# ---------------------------------------------------------------------------
# CodeAgent namespace wrapper
# ---------------------------------------------------------------------------


class _NamespaceTools:
	"""A minimal Tools stand-in with one registered action."""

	def __init__(self, param_model, action_name='click'):
		from types import SimpleNamespace
		from unittest.mock import AsyncMock, MagicMock

		self.action = MagicMock()
		self.action.param_model = param_model
		self.action.function = AsyncMock(return_value=ActionResult(extracted_content='ok'))
		registry = SimpleNamespace(actions={action_name: self.action})
		self.registry = SimpleNamespace(registry=registry)


@pytest.mark.asyncio
@pytest.mark.parametrize('index', [None, 0, 'i_3', True])
async def test_namespace_wrapper_raises_typed_readiness_error(index):
	from openbrowser.code_use.namespace import create_namespace

	tools = _NamespaceTools(ClickElementAction)
	namespace = create_namespace(_FakeBrowser(_map(1)), tools=tools)

	with pytest.raises(ActionReadinessError) as exc_info:
		await namespace['click'](index=index)

	assert exc_info.value.action == 'click'
	assert 'get_status' in str(exc_info.value)
	tools.action.function.assert_not_awaited()


@pytest.mark.asyncio
async def test_namespace_wrapper_handles_an_omitted_index():
	from openbrowser.code_use.namespace import create_namespace

	tools = _NamespaceTools(ClickElementAction)
	namespace = create_namespace(_FakeBrowser(_map(1)), tools=tools)

	with pytest.raises(ActionReadinessError) as exc_info:
		await namespace['click']()

	assert exc_info.value.code is ActionReadinessCode.MISSING
	assert 'get_status' in str(exc_info.value)
	tools.action.function.assert_not_awaited()


@pytest.mark.asyncio
async def test_namespace_wrapper_passes_a_valid_index_through():
	from openbrowser.code_use.namespace import create_namespace

	tools = _NamespaceTools(ClickElementAction)
	namespace = create_namespace(_FakeBrowser(_map(1, 4)), tools=tools)

	result = await namespace['click'](index='4')

	assert result == 'ok'
	params = tools.action.function.await_args.kwargs['params']
	assert params.index == 4


# ---------------------------------------------------------------------------
# Service-level readiness results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_stale_index_returns_an_error_result():
	result = await _action('click')(
		params=ClickElementAction(index=9),
		browser_session=_FakeBrowser(_map(1, 2)),
	)

	assert isinstance(result, ActionResult)
	assert result.error is not None
	assert 'element_index_stale' in result.error
	assert 'Available indexes: 1, 2' in result.error
	assert result.extracted_content is None


@pytest.mark.asyncio
async def test_input_before_state_returns_an_error_result():
	result = await _action('input')(
		params=InputTextAction(index=1, text='hello'),
		browser_session=_FakeBrowser({}),
	)

	assert result.error is not None
	assert 'browser_state_not_ready' in result.error
	assert result.extracted_content is None


@pytest.mark.asyncio
async def test_upload_file_stale_index_returns_an_error_result():
	result = await _action('upload_file')(
		params=UploadFileAction(index=9, path='/tmp/x.txt'),
		browser_session=_FakeBrowser(_map(1)),
		available_file_paths=['/tmp/x.txt'],
	)

	assert result.error is not None
	assert 'element_index_stale' in result.error
	assert result.extracted_content is None


@pytest.mark.asyncio
async def test_dropdown_actions_stale_index_return_error_results():
	dropdown = await _action('dropdown_options')(
		params=GetDropdownOptionsAction(index=9),
		browser_session=_FakeBrowser(_map(1)),
	)
	select = await _action('select_dropdown')(
		params=SelectDropdownOptionAction(index=9, text='Option'),
		browser_session=_FakeBrowser(_map(1)),
	)

	for result in (dropdown, select):
		assert result.error is not None
		assert 'element_index_stale' in result.error
		assert result.extracted_content is None


@pytest.mark.asyncio
async def test_tool_calling_agent_path_returns_the_readiness_error():
	tools = Tools()
	action_model = tools.registry.create_action_model().model_validate({'click': {'index': 9}})

	result = await tools.act(action=action_model, browser_session=_FakeBrowser(_map(1)))

	assert result.error is not None
	assert 'element_index_stale' in result.error
	assert 'get_status' in result.error
	assert result.extracted_content is None


@pytest.mark.asyncio
async def test_code_agent_tools_upload_reports_stale_indexes():
	from openbrowser.tools.service import CodeAgentTools

	result = await _action('upload_file', CodeAgentTools)(
		params=UploadFileAction(index=9, path='/tmp/x.txt'),
		browser_session=_FakeBrowser(_map(1)),
		available_file_paths=['/tmp/x.txt'],
	)

	assert result.error is not None
	assert 'element_index_stale' in result.error


@pytest.mark.asyncio
async def test_code_executor_returns_the_readiness_message_to_the_agent():
	from openbrowser.code_use.executor import CodeExecutor
	from openbrowser.code_use.namespace import create_namespace

	executor = CodeExecutor()
	executor.set_namespace(create_namespace(_FakeBrowser(_map(1)), tools=Tools()))

	result = await executor.execute('await click(index=9)')

	assert result.success is False
	assert 'element_index_stale' in result.output
	assert 'get_status' in result.output
	assert 'element_index_stale' in (result.error or '')


def test_llm_output_parsing_surfaces_the_readiness_message():
	tools = Tools()
	action_model = tools.registry.create_action_model()

	with pytest.raises(ValidationError) as exc_info:
		action_model.model_validate({'click': {'index': None}})

	message = str(exc_info.value)
	assert 'element_index_missing' in message
	assert 'get_status' in message
