"""Terminal event routing shared by the stream and WebSocket surfaces."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(ROOT / 'backend')
sys.path.insert(0, _BACKEND)
try:
    from app.services.task_terminal import (  # noqa: E402
        cancellation_display_content,
        classify_task_terminal,
    )
finally:
    sys.path.remove(_BACKEND)


def test_codeagent_cancellation_routes_to_cancelled():
    terminal = classify_task_terminal(
        {'cancelled': True, 'success': False, 'result': 'partial data'}
    )

    assert terminal.kind == 'cancelled'
    assert terminal.content == 'partial data'


def test_browser_agent_stopped_phase_routes_to_cancelled():
    terminal = classify_task_terminal(
        {
            'terminal_phase': 'stopped',
            'terminal_reason': 'stopped',
            'success': False,
            'result': 'partial',
        }
    )

    assert terminal.kind == 'cancelled'
    assert terminal.content == 'partial'


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


def test_explicit_failed_phase_overrides_a_success_flag():
    terminal = classify_task_terminal(
        {'terminal_phase': 'failed', 'terminal_reason': 'run_error', 'success': True}
    )

    assert terminal.kind == 'failed'


def test_explicit_succeeded_phase_requires_success():
    terminal = classify_task_terminal(
        {'terminal_phase': 'succeeded', 'success': False, 'result': 'partial'}
    )

    assert terminal.kind == 'failed'


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


def test_legacy_browser_result_without_terminal_phase_uses_history():
    terminal = classify_task_terminal({'success': False, 'result': 'browser result'})

    assert terminal.kind == 'failed'
    assert terminal.content == 'browser result'


def test_cancellation_display_content_keeps_partial_results_after_reload():
    assert cancellation_display_content('partial data') == 'Task was cancelled. partial data'
    assert cancellation_display_content('Task was cancelled') == 'Task was cancelled'
    assert cancellation_display_content('Cancelled by user') == 'Task was cancelled'
