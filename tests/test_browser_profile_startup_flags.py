"""Automation startup flags retain explicit per-profile opt-out."""

from openbrowser.browser.profile import BrowserProfile


def test_browser_startup_disables_background_stack_sampling(tmp_path):
    args = BrowserProfile(headless=True, user_data_dir=tmp_path, enable_default_extensions=False).get_args()
    assert args.count("--disable-stack-profiler") == 1


def test_explicit_profile_opt_out_preserves_stack_sampling(tmp_path):
    args = BrowserProfile(
        headless=True, user_data_dir=tmp_path, ignore_default_args=["--disable-stack-profiler"]
    ).get_args()
    assert "--disable-stack-profiler" not in args
