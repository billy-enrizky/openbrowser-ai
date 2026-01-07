import pytest
from openbrowser.browser.profile import BrowserProfile
from openbrowser.agent.graph import BrowserAgent

def test_wait_for_network_idle_page_load_time_default():
    """Test that the default wait time is 0.3 seconds."""
    profile = BrowserProfile()
    assert profile.wait_for_network_idle_page_load_time == 0.3

def test_wait_for_network_idle_page_load_time_configurable():
    """Test that the wait time can be configured."""
    profile = BrowserProfile(wait_for_network_idle_page_load_time=1.5)
    assert profile.wait_for_network_idle_page_load_time == 1.5

@pytest.mark.asyncio
async def test_agent_uses_configured_wait_time():
    """Test that the agent uses the configured wait time."""
    profile = BrowserProfile(wait_for_network_idle_page_load_time=0.7)
    agent = BrowserAgent(task="test", browser_profile=profile)
    
    assert agent.browser_session.browser_profile.wait_for_network_idle_page_load_time == 0.7
