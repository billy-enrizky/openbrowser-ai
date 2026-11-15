import pytest
from pydantic import ValidationError

from openbrowser.browser.events import NavigateToUrlEvent, NavigationFinishedEvent


def test_explicit_navigation_timeout_extends_only_default_bus_timeout():
    assert NavigateToUrlEvent(url='about:blank', timeout_ms=60000).event_timeout == 65
    assert NavigateToUrlEvent(url='about:blank', timeout_ms=60000, event_timeout=2).event_timeout == 2
    assert NavigateToUrlEvent(url='about:blank', timeout_ms=60000, event_timeout=None).event_timeout is None


@pytest.mark.parametrize('timeout', [0, -1])
def test_nonpositive_navigation_timeout_is_invalid(timeout):
    with pytest.raises(ValidationError):
        NavigateToUrlEvent(url='about:blank', timeout_ms=timeout)


def test_short_navigation_budget_does_not_shorten_default_enclosing_budget():
    default = NavigateToUrlEvent.model_fields['event_timeout'].default
    assert NavigateToUrlEvent(url='about:blank', timeout_ms=1).event_timeout == max(default, 5.001)


@pytest.mark.parametrize('timeout', [None, 0, float('inf')])
def test_terminal_status_notification_timeout_is_finite(timeout):
    with pytest.raises(ValidationError):
        NavigationFinishedEvent(navigation_id='id', generation=1, requested_url='about:blank',
            outcome='cancelled', event_timeout=timeout)
