"""Terminal event routing shared by the stream and WebSocket surfaces."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(ROOT / 'backend')
sys.path.insert(0, _BACKEND)
try:
    from app.services.task_terminal import classify_task_terminal  # noqa: E402
finally:
    sys.path.remove(_BACKEND)


def test_codeagent_cancellation_routes_to_cancelled():
    terminal = classify_task_terminal(
        {'cancelled': True, 'success': False, 'result': 'partial data'}
    )

    assert terminal.kind == 'cancelled'
    assert terminal.content == 'partial data'


def test_codeagent_terminal_failure_routes_to_failed():
    terminal = classify_task_terminal(
        {
            'terminal_phase': 'failed',
            'terminal_reason': 'step_error',
            'success': False,
        }
    )

    assert terminal.kind == 'failed'
    assert terminal.content == 'Task failed: step_error'


def test_codeagent_success_routes_to_completed():
    terminal = classify_task_terminal(
        {'terminal_phase': 'succeeded', 'success': True, 'result': 'final'}
    )

    assert terminal.kind == 'completed'
    assert terminal.content == 'final'


def test_codeagent_nonterminal_phase_routes_to_failed():
    terminal = classify_task_terminal(
        {
            'phase': 'running',
            'terminal_phase': None,
            'success': False,
            'result': 'Task finished without a terminal state',
        }
    )

    assert terminal.kind == 'failed'
    assert terminal.content == 'Task finished without a terminal state'


def test_browser_agent_completion_contract_is_unchanged():
    terminal = classify_task_terminal({'success': False, 'result': 'browser result'})

    assert terminal.kind == 'completed'
    assert terminal.content == 'browser result'
