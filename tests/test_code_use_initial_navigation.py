"""Initial navigation through the real CodeAgent, tools and navigation owner."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from bubus import EventBus

from openbrowser.browser.events import NavigationCompleteEvent
from openbrowser.code_use import service as code_service
from openbrowser.code_use.views import ExecutionStatus
from openbrowser.filesystem.file_system import FileSystem
from openbrowser.models import ActionResult
from openbrowser.tools.views import NavigateAction
from test_code_use_service_deep_coverage import _create_agent, _setup_agent_for_run
from test_navigation import handler_fixture


@pytest.fixture
async def initial_navigation(tmp_path, monkeypatch):
    browser, client, focus, frame, patches = handler_fixture()
    browser.event_bus = EventBus()
    browser._core_handlers_attached = False
    browser._attach_core_handlers()
    agent = _create_agent(
        task="Inspect https://example.com/",
        browser=browser,
        file_system=FileSystem(base_dir=tmp_path),
    )
    _setup_agent_for_run(agent)
    model_called = asyncio.Event()
    delayed = asyncio.Event()
    release_delay = asyncio.Event()
    positive_initial_sleeps = []
    state_phases = []
    tasks = []

    async def replay_model(messages):
        model_called.set()
        return SimpleNamespace(
            completion='```python\nawait done("ready", success=True)\n```',
            usage=None,
            stop_reason="end_turn",
        )

    async def browser_state():
        navigation = browser.lifecycle_snapshot()["navigation"]
        state_phases.append(navigation["phase"] if navigation else None)
        return "observed browser state", None

    async def controlled_sleep(delay):
        if delay > 0 and not model_called.is_set():
            positive_initial_sleeps.append(delay)
            delayed.set()
            await release_delay.wait()
        else:
            await asyncio.sleep(delay)

    agent.llm.ainvoke = AsyncMock(side_effect=replay_model)
    agent._get_browser_state = browser_state
    # Intercept this agent's scheduling only, not EventBus/navigation internals.
    monkeypatch.setattr(code_service, "asyncio", SimpleNamespace(sleep=controlled_sleep))

    def start():
        task = asyncio.create_task(agent.run(max_steps=1))
        tasks.append(task)
        return task

    async def first_model_without_delay(run):
        model_wait = asyncio.create_task(model_called.wait())
        delay_wait = asyncio.create_task(delayed.wait())
        try:
            await asyncio.wait_for(
                asyncio.wait({run, model_wait, delay_wait}, return_when=asyncio.FIRST_COMPLETED),
                5,
            )
            if run.done():
                run.result()
            assert not positive_initial_sleeps, (
                "Initial state/model must not wait on an extra positive sleep after navigation: "
                f"{positive_initial_sleeps}"
            )
            assert model_called.is_set()
            return await asyncio.wait_for(asyncio.shield(run), 5)
        finally:
            for waiter in (model_wait, delay_wait):
                waiter.cancel()
            await asyncio.gather(model_wait, delay_wait, return_exceptions=True)

    try:
        with patches[0], patches[1]:
            yield SimpleNamespace(
                agent=agent,
                browser=browser,
                client=client,
                focus=focus,
                frame=frame,
                model_called=model_called,
                delayed=delayed,
                state_phases=state_phases,
                start=start,
                first_model_without_delay=first_model_without_delay,
            )
    finally:
        release_delay.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await browser._retire_event_bus(browser.event_bus, timeout=1)


@pytest.mark.parametrize("same_document", [False, True])
async def test_initial_state_and_model_use_matching_readiness_and_observers(initial_navigation, same_document):
    case = initial_navigation
    command_sent = asyncio.Event()
    observer_entered = asyncio.Event()
    release_observer = asyncio.Event()
    loader = "old" if same_document else "next"

    async def navigate(**kwargs):
        case.frame.update(loaderId=loader, url=kwargs["params"]["url"])
        command_sent.set()
        return {"frameId": "frame"} if same_document else {"frameId": "frame", "loaderId": loader}

    async def completion_observer(event: NavigationCompleteEvent):
        observer_entered.set()
        await release_observer.wait()

    case.client.send.Page.navigate.side_effect = navigate
    case.browser.event_bus.on(NavigationCompleteEvent, completion_observer)
    run = case.start()
    try:
        await asyncio.wait_for(command_sent.wait(), 5)
        lifecycle = case.client.register.Page.lifecycleEvent.call_args.args[0]
        operation = case.browser._navigation_monitor.active
        for frame_id, loader_id in (("child", loader), ("frame", "unrelated-loader")):
            lifecycle({"frameId": frame_id, "loaderId": loader_id, "name": "load"}, case.focus.session_id)
            await asyncio.sleep(0)
            assert not operation.ready
            assert not observer_entered.is_set()
            assert not case.state_phases and not case.model_called.is_set()
        lifecycle({"frameId": "frame", "loaderId": loader, "name": "load"}, case.focus.session_id)
        await asyncio.wait_for(observer_entered.wait(), 5)
        assert not case.state_phases and not case.model_called.is_set()
        assert not case.delayed.is_set()
        release_observer.set()
        notebook = await case.first_model_without_delay(run)
    finally:
        release_observer.set()

    assert case.state_phases[:2] == ["succeeded", "succeeded"]
    assert case.agent.llm.ainvoke.await_count == 1
    navigation, done = notebook.cells
    assert navigation.status == ExecutionStatus.SUCCESS
    assert navigation.source == "await navigate('https://example.com/')"
    assert navigation.output == "Navigated to https://example.com/"
    assert navigation.browser_state == "observed browser state"
    assert navigation.execution_count == 1
    assert done.status == ExecutionStatus.SUCCESS
    assert done.execution_count == 2
    assert case.agent.namespace["_task_result"] == "ready"
    assert not case.browser._navigation_lock.locked()


@pytest.mark.parametrize("failure", ["protocol", "observer"])
async def test_initial_navigation_error_cell_preserves_model_continuation(initial_navigation, failure):
    case = initial_navigation
    if failure == "protocol":
        case.client.send.Page.navigate = AsyncMock(
            return_value={"frameId": "frame", "errorText": "net::ERR_NAME_NOT_RESOLVED"}
        )
        expected = "site unavailable"
    else:
        async def failing_observer(event: NavigationCompleteEvent):
            raise ValueError("completion observer failed")

        case.browser.event_bus.on(NavigationCompleteEvent, failing_observer)
        expected = "completion observer failed"

    notebook = await case.first_model_without_delay(case.start())
    navigation, done = notebook.cells
    assert navigation.status == ExecutionStatus.ERROR
    assert expected in navigation.error
    assert navigation.execution_count == 1
    assert done.status == ExecutionStatus.SUCCESS
    assert case.agent.llm.ainvoke.await_count == 1
    assert case.state_phases[0] == "failed"
    assert not case.browser._navigation_lock.locked()


async def test_initial_navigation_cancellation_propagates_before_state_or_model(initial_navigation):
    case = initial_navigation
    command_sent = asyncio.Event()

    async def navigate(**kwargs):
        command_sent.set()
        await asyncio.Event().wait()

    case.client.send.Page.navigate.side_effect = navigate
    run = case.start()
    await asyncio.wait_for(command_sent.wait(), 5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert not case.state_phases and not case.model_called.is_set()
    assert not case.agent.session.cells
    assert case.browser.lifecycle_snapshot()["navigation"]["phase"] == "cancelled"
    assert not case.browser._navigation_lock.locked()


@pytest.mark.parametrize("task", ["Inspect the current page", "Compare https://one.example/ with https://two.example/"])
async def test_absent_or_ambiguous_initial_url_does_not_navigate(initial_navigation, task):
    case = initial_navigation
    case.agent.task = task
    notebook = await case.first_model_without_delay(case.start())
    case.client.send.Page.navigate.assert_not_awaited()
    assert len(notebook.cells) == 1
    assert notebook.cells[0].status == ExecutionStatus.SUCCESS
    assert case.state_phases == [None]


async def test_initial_download_handoff_does_not_wait_for_document_or_file_readiness(initial_navigation):
    case = initial_navigation
    case.client.send.Page.navigate = AsyncMock(
        return_value={"frameId": "frame", "isDownload": True, "errorText": "net::ERR_ABORTED"}
    )
    notebook = await case.first_model_without_delay(case.start())
    assert case.browser.lifecycle_snapshot()["navigation"]["phase"] == "download"
    assert case.state_phases[:2] == ["download", "download"]
    assert notebook.cells[0].status == ExecutionStatus.SUCCESS
    assert notebook.cells[0].output == "Navigated to https://example.com/"
    assert case.agent.llm.ainvoke.await_count == 1


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_registered_navigation_override_owns_its_completion_boundary(initial_navigation, outcome):
    case = initial_navigation
    entered = asyncio.Event()
    release = asyncio.Event()

    @case.agent.tools.registry.action("Custom navigation readiness", param_model=NavigateAction)
    async def navigate(params: NavigateAction):
        assert params.url == "https://example.com/"
        entered.set()
        await release.wait()
        if outcome == "error":
            return ActionResult(error="custom navigation failed")
        return ActionResult(extracted_content="custom navigation complete")

    run = case.start()
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not case.state_phases and not case.model_called.is_set()
        assert not case.agent.session.cells
        if outcome == "cancel":
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
            assert not case.state_phases and not case.model_called.is_set()
            assert not case.agent.session.cells
        else:
            release.set()
            notebook = await case.first_model_without_delay(run)
            navigation, done = notebook.cells
            if outcome == "error":
                assert navigation.status == ExecutionStatus.ERROR
                assert navigation.error == "custom navigation failed"
            else:
                assert navigation.status == ExecutionStatus.SUCCESS
                assert navigation.output == "Navigated to https://example.com/"
            assert navigation.execution_count == 1
            assert done.status == ExecutionStatus.SUCCESS
            assert case.agent.llm.ainvoke.await_count == 1
    finally:
        release.set()

    case.client.send.Page.navigate.assert_not_awaited()
    assert case.browser.lifecycle_snapshot()["navigation"] is None
