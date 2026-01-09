from openbrowser.browser.profile import BrowserProfile


def test_wait_for_network_idle_page_load_time_default():
    """Test that the default wait time is 0.5 seconds."""
    profile = BrowserProfile()
    assert profile.wait_for_network_idle_page_load_time == 0.5


def test_wait_for_network_idle_page_load_time_configurable():
    """Test that the wait time can be configured."""
    profile = BrowserProfile(wait_for_network_idle_page_load_time=1.5)
    assert profile.wait_for_network_idle_page_load_time == 1.5
