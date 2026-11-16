import asyncio
from unittest.mock import AsyncMock

import pytest
from test_navigation import policy_handler_fixture

from openbrowser.browser.navigation import ContainmentPhase, NavigationError


@pytest.mark.parametrize('successor', ['same_document', 'new_loader'])
@pytest.mark.parametrize('observe_blank', [False, True])
async def test_initiated_containment_retains_authority_after_a_changed_commit(monkeypatch, successor, observe_blank):
    monkeypatch.setattr('openbrowser.browser.session.NAVIGATION_CONTAINMENT_GRACE', .2)
    session, client, focus, frame, patches, _controls = policy_handler_fixture()
    initiated = asyncio.Event()
    polled_again = asyncio.Event()
    release_observation = asyncio.Event()
    original_read = client.send.Page.getFrameTree
    polls = 0
    task = observer = operation = None

    async def action(**_kwargs):
        frame['url'] += '#changed-after-initiation'
        if successor == 'new_loader':
            frame['loaderId'] = 'successor-loader'
        initiated.set()
        return {'result': {'value': {'initiated': True}}}

    async def read(**kwargs):
        nonlocal polls
        if initiated.is_set():
            polls += 1
            if polls >= 2:
                polled_again.set()
                if observe_blank:
                    await release_observation.wait()
        return await original_read(**kwargs)

    client.send.Runtime.callFunctionOn = AsyncMock(side_effect=action)
    client.send.Page.getFrameTree = AsyncMock(side_effect=read)
    try:
        with patches[0], patches[1]:
            task = asyncio.create_task(session.navigate_to('https://allowed.test/start', timeout_ms=1000))
            observer = asyncio.create_task(polled_again.wait())
            await asyncio.wait_for(asyncio.wait({task, observer}, return_when=asyncio.FIRST_COMPLETED), 1)
            assert polled_again.is_set(), 'A changed commit retired an initiated containment action before verification'
            operation = session._navigation_monitor.active
            assert operation is not None and operation.policy_pending
            assert operation.containment.phase in {ContainmentPhase.VERIFYING, ContainmentPhase.UNCONFIRMED}
            with pytest.raises(NavigationError, match='retir'):
                await session.connect(cdp_url='ws://successor.invalid')
            assert session._cdp_client_root is client and session.agent_focus is focus
            if observe_blank:
                frame.update(loaderId='contained', url='about:blank')
                release_observation.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
            await asyncio.wait_for(asyncio.shield(operation.policy_task), 1)
            if observe_blank:
                assert operation.containment.phase is ContainmentPhase.CONTAINED
                assert not operation.policy_pending
            else:
                assert operation.containment.phase is ContainmentPhase.UNCONFIRMED
                assert operation.policy_pending
                with pytest.raises(NavigationError, match='retir'):
                    await session.reset(stop_event_bus=False)
                assert session._cdp_client_root is client
    finally:
        frame.update(loaderId='contained', url='about:blank')
        release_observation.set()
        if observer is not None:
            observer.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if observer is not None:
            await asyncio.gather(observer, return_exceptions=True)
        if operation is not None and operation.policy_task is not None:
            await asyncio.wait({operation.policy_task}, timeout=1)
        session._navigation_policies.clear()
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)


@pytest.mark.parametrize('successor', ['same_document', 'new_loader'])
@pytest.mark.parametrize('rejection', [None, 'Cannot find context with specified id', 'Could not find object with given id'])
async def test_changed_commit_does_not_retire_unknown_sent_command_errors(monkeypatch, successor, rejection):
    monkeypatch.setattr('openbrowser.browser.session.NAVIGATION_CONTAINMENT_GRACE', .2)
    session, client, _focus, frame, patches, controls = policy_handler_fixture()

    def change_commit():
        frame['url'] += '#changed-after-send'
        if successor == 'new_loader':
            frame['loaderId'] = 'successor-loader'

    controls.on_action = change_commit
    controls.action_error = RuntimeError({'code': -32000 if rejection else -32001,
        'message': rejection or 'Unknown command outcome'})
    operation = None
    try:
        with patches[0], patches[1], pytest.raises(NavigationError):
            await session.navigate_to('https://allowed.test/start', timeout_ms=1000)
        state = session.lifecycle_snapshot()['navigation']
        assert state['containment']['phase'] == ('skipped' if rejection else 'unconfirmed')
        assert client.send.Runtime.callFunctionOn.await_count == 1
        assert client.send.Page.navigate.await_count == 1
        assert frame['url'] != 'about:blank'
        if rejection is None:
            operation = session._navigation_monitor.active
            assert operation is not None and operation.policy_pending and operation.policy_task.done()
            with pytest.raises(NavigationError, match='retir'):
                await session.connect(cdp_url='ws://successor.invalid')
            assert session._cdp_client_root is client
    finally:
        session._navigation_policies.clear()
        if operation is not None:
            session._navigation_monitor.complete(operation)
        await session.event_bus.stop(clear=True, timeout=.1)
        await session._drain_navigation_event_pumps(session.event_bus)
