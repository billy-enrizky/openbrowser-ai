"""Regression tests for graceful stale-Chrome termination.

`LocalBrowserWatchdog._kill_stale_chrome_for_profile()` used to SIGKILL every
Chrome whose --user-data-dir matched the target profile. SIGKILL gives Chrome
no chance to flush cookies/Local State/LevelDB, which can corrupt a profile
that another live instance is still using (multi-instance setups share the
default profile). The fix signals SIGTERM first and escalates to SIGKILL only
for processes that ignore it beyond the existing 5s wait window.

These tests pin the ordering contract without launching a real browser:
- a responsive process receives terminate() and never kill()
- an unresponsive process receives terminate() first, then kill()
"""

import asyncio

import psutil

from openbrowser.browser.watchdogs import local_browser_watchdog as lbw


class FakeChrome:
	"""Minimal psutil.Process stand-in for the watchdog's scan loop."""

	def __init__(self, pid: int, user_data_dir: str, *, dies_on_terminate: bool):
		self.pid = pid
		self.info = {
			'pid': pid,
			'name': 'chrome',
			'cmdline': ['/usr/bin/chrome', f'--user-data-dir={user_data_dir}'],
		}
		self.calls: list[str] = []
		self._alive = True
		self._dies_on_terminate = dies_on_terminate

	def terminate(self):
		self.calls.append('terminate')
		if self._dies_on_terminate:
			self._alive = False

	def kill(self):
		self.calls.append('kill')
		self._alive = False

	def is_running(self):
		return self._alive


def _patch_process_iter(monkeypatch, proc: FakeChrome):
	def fake_process_iter(attrs=None):
		return [proc] if proc.is_running() else []

	monkeypatch.setattr(psutil, 'process_iter', fake_process_iter)


async def test_responsive_chrome_is_terminated_not_killed(tmp_path, monkeypatch):
	"""A Chrome that honors SIGTERM must never see SIGKILL."""
	proc = FakeChrome(4242, str(tmp_path), dies_on_terminate=True)
	_patch_process_iter(monkeypatch, proc)

	killed = await lbw.LocalBrowserWatchdog._kill_stale_chrome_for_profile(str(tmp_path))

	assert killed is True
	assert proc.calls == ['terminate']


async def test_unresponsive_chrome_is_escalated_to_sigkill(tmp_path, monkeypatch):
	"""A Chrome that ignores SIGTERM is force-killed after the wait window."""
	proc = FakeChrome(4243, str(tmp_path), dies_on_terminate=False)
	_patch_process_iter(monkeypatch, proc)

	# Collapse the 50 x 0.1s wait window so the test stays fast.
	real_sleep = asyncio.sleep

	async def instant_sleep(_seconds):
		await real_sleep(0)

	monkeypatch.setattr(lbw.asyncio, 'sleep', instant_sleep)

	killed = await lbw.LocalBrowserWatchdog._kill_stale_chrome_for_profile(str(tmp_path))

	assert killed is True
	assert proc.calls[0] == 'terminate'
	assert 'kill' in proc.calls
	assert not proc.is_running()


async def test_no_matching_process_returns_false(tmp_path, monkeypatch):
	"""Unrelated Chromes (different profile dir) are left untouched."""
	other = FakeChrome(4244, '/somewhere/else', dies_on_terminate=True)
	_patch_process_iter(monkeypatch, other)

	killed = await lbw.LocalBrowserWatchdog._kill_stale_chrome_for_profile(str(tmp_path))

	assert killed is False
	assert other.calls == []
