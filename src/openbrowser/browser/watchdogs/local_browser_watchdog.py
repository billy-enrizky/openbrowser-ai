"""Local browser watchdog for managing browser subprocess lifecycle."""

import asyncio
import inspect
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

import psutil
from bubus import BaseEvent
from pydantic import PrivateAttr

from openbrowser.browser.events import (
	BrowserKillEvent,
	BrowserLaunchEvent,
	BrowserLaunchResult,
	BrowserStopEvent,
)
from openbrowser.browser.watchdog_base import BaseWatchdog
from openbrowser.browser.watchdogs.profile_lease import ProfileInUseError, ProfileLease
from openbrowser.config import is_openbrowser_managed_profile_dir
from openbrowser.observability import observe_debug

if TYPE_CHECKING:
	pass

_logger = logging.getLogger(__name__)
IS_WINDOWS = os.name == 'nt'
OWNERSHIP_MARKER_PREFIX = '--openbrowser-instance-id='


class LocalBrowserWatchdog(BaseWatchdog):
	"""Manages local browser subprocess lifecycle."""

	BROWSER_PROCESS_NAMES: ClassVar[tuple[str, ...]] = ('chrome', 'chromium', 'brave', 'msedge')

	# Events this watchdog listens to
	LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = [
		BrowserLaunchEvent,
		BrowserKillEvent,
		BrowserStopEvent,
	]

	# Events this watchdog emits
	EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []

	# Disposable Chromium caches that can be safely cleared without removing
	# cookies, login databases, local storage, or IndexedDB.
	CACHE_PATHS: ClassVar[tuple[str, ...]] = (
		'GraphiteDawnCache',
		'GrShaderCache',
		'ShaderCache',
		'Default/Cache',
		'Default/Code Cache',
		'Default/GPUCache',
		'Default/DawnGraphiteCache',
		'Default/DawnWebGPUCache',
		'Default/Shared Dictionary',
	)

	# Private state for subprocess management
	_subprocess: psutil.Process | None = PrivateAttr(default=None)
	_owns_browser_resources: bool = PrivateAttr(default=True)
	_temp_dirs_to_cleanup: list[Path] = PrivateAttr(default_factory=list)
	_original_user_data_dir: str | None = PrivateAttr(default=None)
	_instance_id: str = PrivateAttr(default_factory=lambda: uuid.uuid4().hex)
	_profile_lease: ProfileLease | None = PrivateAttr(default=None)
	_browser_record: dict[str, Any] | None = PrivateAttr(default=None)
	_spawned_subprocess: asyncio.subprocess.Process | None = PrivateAttr(default=None)

	@observe_debug(ignore_input=True, ignore_output=True, name='browser_launch_event')
	async def on_BrowserLaunchEvent(self, event: BrowserLaunchEvent) -> BrowserLaunchResult:
		"""Launch a local browser process."""

		try:
			self.logger.debug('[LocalBrowserWatchdog] Received BrowserLaunchEvent, launching local browser...')

			# self.logger.debug('[LocalBrowserWatchdog] Calling _launch_browser...')
			process, cdp_url = await self._launch_browser()
			self._subprocess = process
			# self.logger.debug(f'[LocalBrowserWatchdog] _launch_browser returned: process={process}, cdp_url={cdp_url}')

			return BrowserLaunchResult(cdp_url=cdp_url)
		except Exception as e:
			self.logger.error(f'[LocalBrowserWatchdog] Exception in on_BrowserLaunchEvent: {e}', exc_info=True)
			raise

	async def on_BrowserKillEvent(self, event: BrowserKillEvent) -> None:
		"""Kill the local browser subprocess."""
		self.logger.debug('[LocalBrowserWatchdog] Killing local browser process')

		cleanup_succeeded = True
		if self._subprocess:
			cdp_url = getattr(self.browser_session.browser_profile, 'cdp_url', None)
			cleanup_succeeded = await self._cleanup_process(
				self._subprocess,
				browser_record=self._browser_record,
				cdp_url=cdp_url,
				require_identity=True,
			)
			if not cleanup_succeeded:
				raise RuntimeError('Unable to safely close the owned browser process')
			self._subprocess = None
			self._browser_record = None
		elif self._spawned_subprocess:
			cleanup_succeeded = await self._cleanup_spawned_subprocess(self._spawned_subprocess)
			if not cleanup_succeeded:
				raise RuntimeError('Unable to safely close the owned browser process')
			self._spawned_subprocess = None

		try:
			active_user_data_dir = self.browser_session.browser_profile.user_data_dir or self._original_user_data_dir
			profile_directory = self.browser_session.browser_profile.profile_directory or 'Default'
			cleared_bytes = self._cleanup_profile_cache(active_user_data_dir, profile_directory)
			if cleared_bytes > 0:
				self.logger.info(
					f'[LocalBrowserWatchdog] Cleared {cleared_bytes / (1024 * 1024):.1f} MB of browser cache '
					f'from managed profile {active_user_data_dir}'
				)
		finally:
			await self._release_profile_lease()

		# Clean up temp directories if any were created
		for temp_dir in self._temp_dirs_to_cleanup:
			self._cleanup_temp_dir(temp_dir)
		self._temp_dirs_to_cleanup.clear()

		# Restore original user_data_dir if it was modified
		if self._original_user_data_dir is not None:
			self.browser_session.browser_profile.user_data_dir = self._original_user_data_dir
			self._original_user_data_dir = None

		self.logger.debug('[LocalBrowserWatchdog] Browser cleanup completed')

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
		"""Listen for BrowserStopEvent and dispatch BrowserKillEvent without awaiting it."""
		if self.browser_session.is_local and (self._subprocess or self._profile_lease):
			self.logger.debug('[LocalBrowserWatchdog] BrowserStopEvent received, dispatching BrowserKillEvent')
			# Dispatch and await the child event so cleanup failures reach the owning
			# session instead of being reported as a successful stop.
			kill_event = self.event_bus.dispatch(BrowserKillEvent())
			if inspect.isawaitable(kill_event):
				await kill_event
				event_result = getattr(kill_event, 'event_result', None)
				if callable(event_result):
					result = event_result(raise_if_any=True, raise_if_none=False)
					if inspect.isawaitable(result):
						await result

	@observe_debug(ignore_input=True, ignore_output=True, name='launch_browser_process')
	async def _launch_browser(self, max_retries: int = 3) -> tuple[psutil.Process, str]:
		"""Launch browser process and return (process, cdp_url).

		Before launching, acquire the profile lease and reclaim only a browser
		process recorded by a previous OpenBrowser owner. On launch failures,
		retry with a temporary directory as fallback.

		Returns:
			Tuple of (psutil.Process, cdp_url)
		"""
		# Keep track of original user_data_dir to restore if needed
		profile = self.browser_session.browser_profile
		self._original_user_data_dir = str(profile.user_data_dir) if profile.user_data_dir else None
		self._temp_dirs_to_cleanup = []
		self._browser_record = None

		for attempt in range(max_retries):
			launched_process: psutil.Process | None = None
			cdp_url: str | None = None
			try:
				# Get launch args from profile
				launch_args = profile.get_args()
				ownership_marker = self._ensure_ownership_marker(launch_args)
				await self._ensure_profile_lease(profile.user_data_dir, ownership_marker)

				if attempt == 0 and self._original_user_data_dir:
					cleared_bytes = self._cleanup_profile_cache(
						self._original_user_data_dir,
						profile.profile_directory or 'Default',
					)
					if cleared_bytes > 0:
						self.logger.info(
							f'[LocalBrowserWatchdog] Cleared {cleared_bytes / (1024 * 1024):.1f} MB of browser cache '
							f'from managed profile {self._original_user_data_dir}'
						)

				# Add debugging port
				debug_port = self._find_free_port()
				launch_args.extend(
					[
						f'--remote-debugging-port={debug_port}',
					]
				)
				assert '--user-data-dir' in str(launch_args), (
					'User data dir must be set somewhere in launch args to a non-default path, otherwise Chrome will not let us attach via CDP'
				)

				# Get browser executable
				# Priority: custom executable > fallback paths > playwright subprocess
				if profile.executable_path:
					browser_path = profile.executable_path
					self.logger.debug(f'[LocalBrowserWatchdog] Using custom local browser executable_path= {browser_path}')
				else:
					# Try fallback paths first (system browsers preferred)
					browser_path = self._find_installed_browser_path()
					if not browser_path:
						self.logger.error(
							'[LocalBrowserWatchdog] No local browser binary found, installing browser using playwright subprocess...'
						)
						browser_path = await self._install_browser_with_playwright()

				self.logger.debug(f'[LocalBrowserWatchdog] Found local browser installed at executable_path= {browser_path}')
				if not browser_path:
					raise RuntimeError('No local Chrome/Chromium install found, and failed to install with playwright')

				# Launch browser subprocess directly
				self.logger.debug(f'[LocalBrowserWatchdog] Launching browser subprocess with {len(launch_args)} args...')
				self.logger.debug(
					f'[LocalBrowserWatchdog] user_data_dir={profile.user_data_dir}, profile_directory={profile.profile_directory}'
				)
				subprocess = await asyncio.create_subprocess_exec(
					browser_path,
					*launch_args,
					stdout=asyncio.subprocess.PIPE,
					stderr=asyncio.subprocess.PIPE,
				)
				self._spawned_subprocess = subprocess
				self.logger.debug(
					f'[LocalBrowserWatchdog] Browser running with browser_pid= {subprocess.pid} listening on CDP port :{debug_port}'
				)

				# Convert to psutil.Process
				process = psutil.Process(subprocess.pid)
				self._spawned_subprocess = None
				launched_process = process
				self._subprocess = process
				if self._profile_lease:
					self._browser_record = self._profile_lease.record_browser(
						process,
						ownership_marker=ownership_marker,
						executable=str(browser_path),
						cdp_port=debug_port,
					)

				# Wait for CDP to be ready and get the URL
				cdp_url = await self._wait_for_cdp_url(debug_port)

				# Success! Clean up temp dirs we created but did not use. Keep the
				# active fallback profile until the browser itself is cleaned up.
				active_profile = self._profile_lease.profile_dir if self._profile_lease else None
				remaining_temp_dirs: list[Path] = []
				for tmp_dir in self._temp_dirs_to_cleanup:
					if active_profile and tmp_dir.expanduser().resolve() == active_profile:
						remaining_temp_dirs.append(tmp_dir)
						continue
					try:
						shutil.rmtree(tmp_dir, ignore_errors=True)
					except Exception:
						pass
				self._temp_dirs_to_cleanup = remaining_temp_dirs

				return process, cdp_url

			except ProfileInUseError as error:
				if launched_process:
					cleanup_succeeded = await self._cleanup_process(
						launched_process,
						browser_record=self._browser_record,
						cdp_url=cdp_url,
						require_identity=True,
					)
					if not cleanup_succeeded:
						raise RuntimeError('Unable to safely close the owned browser process') from error
					self._subprocess = None
					self._browser_record = None
				elif self._spawned_subprocess:
					cleanup_succeeded = await self._cleanup_spawned_subprocess(self._spawned_subprocess)
					if not cleanup_succeeded:
						raise RuntimeError('Unable to safely close the owned browser process') from error
					self._spawned_subprocess = None
				await self._release_profile_lease()
				self._restore_original_profile(profile)
				self._cleanup_temp_dirs()
				raise
			except BaseException as error:
				error_str = str(error).lower()

				if launched_process:
					cleanup_succeeded = await self._cleanup_process(
						launched_process,
						browser_record=self._browser_record,
						cdp_url=cdp_url,
						require_identity=True,
					)
					if not cleanup_succeeded:
						raise RuntimeError('Unable to safely close the owned browser process') from error
					launched_process = None
					self._subprocess = None
					self._browser_record = None
				elif self._spawned_subprocess:
					cleanup_succeeded = await self._cleanup_spawned_subprocess(self._spawned_subprocess)
					if not cleanup_succeeded:
						raise RuntimeError('Unable to safely close the owned browser process') from error
					self._spawned_subprocess = None

				# Check if this is a user_data_dir related error (profile lock,
				# timeout waiting for CDP, or other startup failure)
				is_profile_error = any(
					err in error_str
					for err in ['singletonlock', 'user data directory', 'cannot create', 'did not start within']
				)
				if is_profile_error:
					self.logger.warning(f'Browser launch failed (attempt {attempt + 1}/{max_retries}): {error}')

					if attempt < max_retries - 1:
						# Release the old profile before falling back to a temporary
						# profile. The next attempt acquires its own lease.
						await self._release_profile_lease()

						tmp_dir = Path(tempfile.mkdtemp(prefix='openbrowser-tmp-'))
						self._temp_dirs_to_cleanup.append(tmp_dir)
						profile.user_data_dir = str(tmp_dir)
						self.logger.debug(f'Retrying with temporary user_data_dir: {tmp_dir}')

						await asyncio.sleep(1.0)
						continue

				# Not a recoverable error or last attempt failed
				# Restore original user_data_dir before raising
				await self._release_profile_lease()
				self._restore_original_profile(profile)

				# Clean up any temp dirs we created
				self._cleanup_temp_dirs()

				raise

		# Should not reach here, but just in case
		self._restore_original_profile(profile)
		await self._release_profile_lease()
		raise RuntimeError(f'Failed to launch browser after {max_retries} attempts')

	def _ensure_ownership_marker(self, launch_args: list[str]) -> str:
		"""Return one stable marker used to prove browser ownership."""
		for argument in launch_args:
			if argument.startswith(OWNERSHIP_MARKER_PREFIX):
				self._instance_id = argument.removeprefix(OWNERSHIP_MARKER_PREFIX)
				return argument

		marker = f'{OWNERSHIP_MARKER_PREFIX}{self._instance_id}'
		launch_args.append(marker)
		return marker

	async def _ensure_profile_lease(self, user_data_dir: str | Path | None, ownership_marker: str) -> None:
		if not user_data_dir:
			return

		resolved_profile = Path(user_data_dir).expanduser().resolve()
		if self._profile_lease and (
			self._profile_lease.profile_dir == resolved_profile and self._profile_lease.instance_id == self._instance_id
		):
			return

		await self._release_profile_lease()
		lease = ProfileLease(resolved_profile, instance_id=self._instance_id)
		previous_metadata = lease.acquire()
		try:
			await self._reclaim_previous_metadata(previous_metadata, ownership_marker)
			lease.write_metadata()
		except BaseException:
			lease.release()
			raise
		self._profile_lease = lease

	async def _reclaim_previous_metadata(
		self, metadata: dict[str, Any] | None, ownership_marker: str
	) -> None:
		if not metadata or metadata.get('profile_dir') != str(self._profile_lease_path()):
			return

		previous_instance_id = metadata.get('instance_id')
		owner_status = ProfileLease.process_identity_status(metadata.get('owner_pid'), metadata.get('owner_start_time'))
		if owner_status == 'unknown':
			raise ProfileInUseError(self._profile_lease_path())
		if owner_status == 'match' and previous_instance_id != self._instance_id:
			raise ProfileInUseError(self._profile_lease_path())

		cleanup_succeeded = await self._kill_stale_chrome_for_profile(
			str(self._profile_lease_path()),
			metadata=metadata,
			instance_id=self._instance_id,
		)
		if not cleanup_succeeded:
			raise ProfileInUseError(self._profile_lease_path())

	def _profile_lease_path(self) -> Path:
		if self._profile_lease:
			return self._profile_lease.profile_dir
		return Path(self.browser_session.browser_profile.user_data_dir).expanduser().resolve()

	async def _release_profile_lease(self) -> None:
		lease = self._profile_lease
		self._profile_lease = None
		self._browser_record = None
		if lease:
			try:
				lease.clear_browser()
			except Exception as error:
				self.logger.warning(f'[LocalBrowserWatchdog] Unable to clear browser metadata: {error}')
			finally:
				try:
					lease.release()
				except Exception as error:
					self.logger.warning(f'[LocalBrowserWatchdog] Unable to release profile lease: {error}')

	def _restore_original_profile(self, profile) -> None:
		if self._original_user_data_dir is not None:
			profile.user_data_dir = self._original_user_data_dir

	def _cleanup_temp_dirs(self) -> None:
		for temp_dir in self._temp_dirs_to_cleanup:
			try:
				shutil.rmtree(temp_dir, ignore_errors=True)
			except Exception:
				pass

	@staticmethod
	def _find_installed_browser_path() -> str | None:
		"""Try to find browser executable from common fallback locations.

		Prioritizes:
		1. System Chrome Stable
		1. Playwright chromium
		2. Other system native browsers (Chromium -> Chrome Canary/Dev -> Brave)
		3. Playwright headless-shell fallback

		Returns:
			Path to browser executable or None if not found
		"""
		import glob
		import platform
		from pathlib import Path

		system = platform.system()
		patterns = []

		# Get playwright browsers path from environment variable if set
		playwright_path = os.environ.get('PLAYWRIGHT_BROWSERS_PATH')

		if system == 'Darwin':  # macOS
			if not playwright_path:
				playwright_path = '~/Library/Caches/ms-playwright'
			patterns = [
				'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
				f'{playwright_path}/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium',
				'/Applications/Chromium.app/Contents/MacOS/Chromium',
				'/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
				'/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
				f'{playwright_path}/chromium_headless_shell-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium',
			]
		elif system == 'Linux':
			if not playwright_path:
				playwright_path = '~/.cache/ms-playwright'
			patterns = [
				'/usr/bin/google-chrome-stable',
				'/usr/bin/google-chrome',
				'/usr/local/bin/google-chrome',
				f'{playwright_path}/chromium-*/chrome-linux/chrome',
				f'{playwright_path}/chromium-*/chrome-linux64/chrome',
				'/usr/bin/chromium',
				'/usr/bin/chromium-browser',
				'/usr/local/bin/chromium',
				'/snap/bin/chromium',
				'/usr/bin/google-chrome-beta',
				'/usr/bin/google-chrome-dev',
				'/usr/bin/brave-browser',
				f'{playwright_path}/chromium_headless_shell-*/chrome-linux/chrome',
				f'{playwright_path}/chromium_headless_shell-*/chrome-linux64/chrome',
			]
		elif system == 'Windows':
			if not playwright_path:
				playwright_path = r'%LOCALAPPDATA%\ms-playwright'
			patterns = [
				r'C:\Program Files\Google\Chrome\Application\chrome.exe',
				r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
				r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe',
				r'%PROGRAMFILES%\Google\Chrome\Application\chrome.exe',
				r'%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe',
				f'{playwright_path}\\chromium-*\\chrome-win\\chrome.exe',
				r'C:\Program Files\Chromium\Application\chrome.exe',
				r'C:\Program Files (x86)\Chromium\Application\chrome.exe',
				r'%LOCALAPPDATA%\Chromium\Application\chrome.exe',
				r'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe',
				r'C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe',
				r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
				r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
				r'%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe',
				f'{playwright_path}\\chromium_headless_shell-*\\chrome-win\\chrome.exe',
			]

		for pattern in patterns:
			# Expand user home directory
			expanded_pattern = Path(pattern).expanduser()

			# Handle Windows environment variables
			if system == 'Windows':
				pattern_str = str(expanded_pattern)
				for env_var in ['%LOCALAPPDATA%', '%PROGRAMFILES%', '%PROGRAMFILES(X86)%']:
					if env_var in pattern_str:
						env_key = env_var.strip('%').replace('(X86)', ' (x86)')
						env_value = os.environ.get(env_key, '')
						if env_value:
							pattern_str = pattern_str.replace(env_var, env_value)
				expanded_pattern = Path(pattern_str)

			# Convert to string for glob
			pattern_str = str(expanded_pattern)

			# Check if pattern contains wildcards
			if '*' in pattern_str:
				# Use glob to expand the pattern
				matches = glob.glob(pattern_str)
				if matches:
					# Sort matches and take the last one (alphanumerically highest version)
					matches.sort()
					browser_path = matches[-1]
					if Path(browser_path).exists() and Path(browser_path).is_file():
						return browser_path
			else:
				# Direct path check
				if expanded_pattern.exists() and expanded_pattern.is_file():
					return str(expanded_pattern)

		return None

	async def _install_browser_with_playwright(self) -> str:
		"""Get browser executable path from playwright in a subprocess to avoid thread issues."""
		import platform

		# Build command - only use --with-deps on Linux (it fails on Windows/macOS)
		cmd = ['uvx', 'playwright', 'install', 'chrome']
		if platform.system() == 'Linux':
			cmd.append('--with-deps')

		# Run in subprocess with timeout
		process = await asyncio.create_subprocess_exec(
			*cmd,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)

		try:
			stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60.0)
			self.logger.debug(f'[LocalBrowserWatchdog] 📦 Playwright install output: {stdout}')
			browser_path = self._find_installed_browser_path()
			if browser_path:
				return browser_path
			self.logger.error(f'[LocalBrowserWatchdog] ❌ Playwright local browser installation error: \n{stdout}\n{stderr}')
			raise RuntimeError('No local browser path found after: uvx playwright install chrome')
		except TimeoutError:
			# Kill the subprocess if it times out
			process.kill()
			await process.wait()
			raise RuntimeError('Timeout getting browser path from playwright')
		except Exception as e:
			# Make sure subprocess is terminated
			if process.returncode is None:
				process.kill()
				await process.wait()
			raise RuntimeError(f'Error getting browser path: {e}')

	@staticmethod
	def _find_free_port() -> int:
		"""Find a free port for the debugging interface."""
		import socket

		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind(('127.0.0.1', 0))
			s.listen(1)
			port = s.getsockname()[1]
		return port

	@staticmethod
	async def _wait_for_cdp_url(port: int, timeout: float = 30) -> str:
		"""Wait for the browser to start and return the CDP URL."""
		import httpx

		start_time = asyncio.get_event_loop().time()

		async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
			while asyncio.get_event_loop().time() - start_time < timeout:
				try:
					resp = await client.get(f'http://localhost:{port}/json/version')
					if resp.status_code == 200:
						# Chrome is ready
						return f'http://localhost:{port}/'
					else:
						# Chrome is starting up and returning 502/500 errors
						await asyncio.sleep(0.1)
				except Exception:
					# Connection error or request timeout - Chrome might not be ready yet
					await asyncio.sleep(0.1)

		raise TimeoutError(f'Browser did not start within {timeout} seconds')

	@staticmethod
	async def _cleanup_spawned_subprocess(process: asyncio.subprocess.Process) -> bool:
		"""Stop a just-spawned child through its exact asyncio process handle."""
		try:
			if process.returncode is not None:
				return True

			process.terminate()
			try:
				await asyncio.wait_for(process.wait(), timeout=5.0)
			except asyncio.TimeoutError:
				process.kill()
				await asyncio.wait_for(process.wait(), timeout=5.0)

			return process.returncode is not None
		except ProcessLookupError:
			return True
		except (OSError, asyncio.TimeoutError):
			return False

	@staticmethod
	def _recorded_process_state(process: psutil.Process, record: dict[str, Any]) -> str:
		"""Classify a process against a recorded PID/start-time identity."""
		try:
			current_pid = int(process.pid)
			current_start_time = float(process.create_time())
			expected_pid = int(record.get('pid'))
			expected_start_time = float(record.get('start_time'))
		except psutil.NoSuchProcess:
			return 'gone'
		except (psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError, OSError):
			return 'unknown'

		if current_pid == expected_pid and current_start_time == expected_start_time:
			return 'match'
		return 'mismatch'

	@staticmethod
	def _get_process_cmdline(process: psutil.Process) -> list[str] | tuple[str, ...] | None:
		"""Read a process command line, falling back to cached psutil info."""
		try:
			cmdline = process.cmdline()
		except psutil.NoSuchProcess:
			raise
		except (AttributeError, OSError, psutil.Error):
			process_info = getattr(process, 'info', None)
			cmdline = process_info.get('cmdline') if isinstance(process_info, dict) else None
		return cmdline if isinstance(cmdline, (list, tuple)) else None

	@staticmethod
	def _merge_owned_descendants(
		children: list[tuple[psutil.Process, dict[str, Any]]], browser_record: dict[str, Any]
	) -> bool:
		"""Rescan and merge owned descendants after the root changes state."""
		try:
			excluded_pids = {int(browser_record['pid'])}
		except (KeyError, TypeError, ValueError):
			return False

		discovered, scan_succeeded = LocalBrowserWatchdog._find_owned_descendants(
			browser_record, excluded_pids=excluded_pids
		)
		if not scan_succeeded:
			return False

		try:
			known_pids = {int(child.pid) for child, _ in children}
			for child, child_record in discovered:
				if int(child.pid) not in known_pids:
					children.append((child, child_record))
					known_pids.add(int(child.pid))
		except (AttributeError, TypeError, ValueError):
			return False
		return True

	@staticmethod
	async def _cleanup_process(
		process: psutil.Process | None,
		*,
		browser_record: dict[str, Any] | None = None,
		cdp_url: str | None = None,
		require_identity: bool = True,
	) -> bool:
		"""Safely stop an owned browser process and report whether it exited."""
		if not process:
			return True

		if require_identity and not browser_record:
			return False

		owned_children: list[tuple[psutil.Process, dict[str, Any]]] = []
		if browser_record:
			try:
				descendants = process.children(recursive=True)
			except psutil.NoSuchProcess:
				descendants = []
			except (psutil.AccessDenied, psutil.ZombieProcess):
				return False

			for child in descendants:
				try:
					# A descendant with an inaccessible command line cannot be
					# proven foreign, so retain ownership and fail closed.
					if LocalBrowserWatchdog._get_process_cmdline(child) is None:
						return False
					child_record = {
						**browser_record,
						'pid': int(child.pid),
						'start_time': float(child.create_time()),
					}
					if ProfileLease.process_matches_browser(child, child_record):
						owned_children.append((child, child_record))
				except psutil.NoSuchProcess:
					continue
				except (psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError, OSError):
					return False

			root_state = LocalBrowserWatchdog._recorded_process_state(process, browser_record)
			if root_state != 'match':
				if root_state == 'gone':
					if not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
						return False
					return await LocalBrowserWatchdog._cleanup_owned_descendants(
						owned_children, force_kill=not IS_WINDOWS
					)
				return False

		try:
			if IS_WINDOWS:
				if not browser_record or not browser_record.get('cdp_port'):
					return False
				try:
					recorded_cdp_url = f"http://127.0.0.1:{int(browser_record['cdp_port'])}/"
				except (TypeError, ValueError):
					return False
				root_state = LocalBrowserWatchdog._recorded_process_state(process, browser_record)
				if root_state != 'match':
					if root_state == 'gone':
						if not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
							return False
						return await LocalBrowserWatchdog._cleanup_owned_descendants(
							owned_children, force_kill=False
						)
					return False
				if not await LocalBrowserWatchdog._close_browser_via_cdp(
					recorded_cdp_url,
					expected_pid=int(browser_record['pid']),
					expected_start_time=float(browser_record['start_time']),
				):
					return False
				root_exited = await LocalBrowserWatchdog._wait_for_process_exit(process)
				if not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
					return False
				children_exited = await LocalBrowserWatchdog._cleanup_owned_descendants(owned_children, force_kill=False)
				return root_exited and children_exited

			# POSIX: give the owned browser its normal shutdown path first.
			root_state = LocalBrowserWatchdog._recorded_process_state(process, browser_record) if browser_record else 'match'
			if root_state != 'match':
				if root_state == 'gone' and browser_record:
					if not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
						return False
					return await LocalBrowserWatchdog._cleanup_owned_descendants(owned_children, force_kill=True)
				return False

			try:
				process.terminate()
			except psutil.NoSuchProcess:
				root_exited = True
			else:
				for child, child_record in owned_children:
					child_state = LocalBrowserWatchdog._recorded_process_state(child, child_record)
					if child_state == 'gone':
						continue
					if child_state != 'match':
						return False
					try:
						child.terminate()
					except psutil.NoSuchProcess:
						continue
				root_exited = await LocalBrowserWatchdog._wait_for_process_exit(process)

			if browser_record and not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
				return False
			children_exited = await LocalBrowserWatchdog._cleanup_owned_descendants(owned_children, force_kill=True)
			if root_exited:
				return children_exited

			# Re-check identity immediately before SIGKILL to protect against PID reuse.
			if browser_record:
				root_state = LocalBrowserWatchdog._recorded_process_state(process, browser_record)
				if root_state != 'match':
					if root_state == 'gone':
						if not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
							return False
						return await LocalBrowserWatchdog._cleanup_owned_descendants(owned_children, force_kill=True)
					return False

			try:
				process.kill()
			except psutil.NoSuchProcess:
				if browser_record and not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
					return False
				return await LocalBrowserWatchdog._cleanup_owned_descendants(owned_children, force_kill=True)
			root_exited = await LocalBrowserWatchdog._wait_for_process_exit(process)
			if not root_exited:
				return False
			if browser_record and not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
				return False
			return await LocalBrowserWatchdog._cleanup_owned_descendants(owned_children, force_kill=True)
		except psutil.NoSuchProcess:
			if browser_record and not LocalBrowserWatchdog._merge_owned_descendants(owned_children, browser_record):
				return False
			return await LocalBrowserWatchdog._cleanup_owned_descendants(owned_children, force_kill=not IS_WINDOWS)
		except (psutil.AccessDenied, psutil.ZombieProcess):
			return False
		except Exception:
			return False

	@staticmethod
	def _find_owned_descendants(
		browser_record: dict[str, Any], *, excluded_pids: set[int]
	) -> tuple[list[tuple[psutil.Process, dict[str, Any]]], bool]:
		"""Find owned Chrome descendants when the recorded root is unavailable."""
		owned: list[tuple[psutil.Process, dict[str, Any]]] = []
		try:
			processes = psutil.process_iter(['pid', 'name', 'cmdline'])
			for process in processes:
				try:
					pid = int(process.pid)
					if pid in excluded_pids:
						continue
					process_info = process.info
					if not isinstance(process_info, dict):
						return [], False
					name = (process_info.get('name') or '').lower()
					if not any(browser_name in name for browser_name in LocalBrowserWatchdog.BROWSER_PROCESS_NAMES):
						continue
					if process_info.get('cmdline') is None:
						return [], False
					child_record = {
						**browser_record,
						'pid': pid,
						'start_time': float(process.create_time()),
					}
					if ProfileLease.process_matches_browser(process, child_record):
						owned.append((process, child_record))
				except psutil.NoSuchProcess:
					continue
				except (psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError, OSError):
					return [], False
		except (psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError, OSError):
			return [], False
		return owned, True

	@staticmethod
	async def _cleanup_owned_descendants(
		children: list[tuple[psutil.Process, dict[str, Any]]], *, force_kill: bool
	) -> bool:
		"""Stop descendants whose profile and ownership marker match the root browser."""
		for child, child_record in children:
			try:
				child_state = LocalBrowserWatchdog._recorded_process_state(child, child_record)
				if child_state == 'gone':
					continue
				if child_state != 'match':
					return False
				if not await LocalBrowserWatchdog._wait_for_process_exit(child):
					if LocalBrowserWatchdog._recorded_process_state(child, child_record) != 'match':
						return False
					if not force_kill:
						return False
					try:
						child.kill()
					except psutil.NoSuchProcess:
						continue
					if not await LocalBrowserWatchdog._wait_for_process_exit(child):
						return False
			except psutil.NoSuchProcess:
				continue
			except (psutil.AccessDenied, psutil.ZombieProcess):
				return False

		return True

	@staticmethod
	async def _wait_for_process_exit(process: psutil.Process, attempts: int = 50) -> bool:
		"""Poll a process without blocking the event loop."""
		for _ in range(attempts):
			try:
				if not process.is_running():
					return True
			except psutil.NoSuchProcess:
				return True
			await asyncio.sleep(0.1)
		try:
			return not process.is_running()
		except psutil.NoSuchProcess:
			return True

	@staticmethod
	async def _close_browser_via_cdp(
		cdp_url: str, *, expected_pid: int | None = None, expected_start_time: float | None = None
	) -> bool:
		"""Ask an owned Windows browser to close through its CDP endpoint."""
		try:
			import httpx
			import websockets

			if expected_pid is not None and expected_start_time is not None:
				port = urlsplit(cdp_url).port
				if port is None:
					return False
				process = psutil.Process(expected_pid)
				if not ProfileLease.process_matches_identity(process, expected_pid, expected_start_time):
					return False
				endpoint_owned = any(
					getattr(connection, 'pid', None) == expected_pid
					and getattr(getattr(connection, 'laddr', None), 'port', None) == port
					for connection in psutil.net_connections(kind='tcp')
				)
				if not endpoint_owned:
					return False

			version_url = f'{cdp_url.rstrip("/")}/json/version'
			async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
				response = await client.get(version_url)
				response.raise_for_status()
				websocket_url = response.json().get('webSocketDebuggerUrl')
			if not websocket_url:
				return False

			async with websockets.connect(websocket_url) as websocket:
				if expected_pid is not None:
					request_id = 1
					await websocket.send(json.dumps({'id': request_id, 'method': 'SystemInfo.getProcessInfo'}))
					response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
					payload = json.loads(response)
					process_info = payload.get('result', {}).get('processInfo', {})
					if isinstance(process_info, list):
						process_ids = {
							int(item['id'])
							for item in process_info
							if isinstance(item, dict) and item.get('id') is not None
						}
					else:
						process_ids = {int(process_info['id'])} if isinstance(process_info, dict) and process_info.get('id') is not None else set()
					if payload.get('id') != request_id or int(expected_pid) not in process_ids:
						return False
				await websocket.send(json.dumps({'id': 2, 'method': 'Browser.close'}))
			return True
		except Exception:
			return False

	def _cleanup_temp_dir(self, temp_dir: Path | str) -> None:
		"""Clean up temporary directory.

		Args:
			temp_dir: Path to temporary directory to remove
		"""
		if not temp_dir:
			return

		try:
			temp_path = Path(temp_dir)
			# Only remove if it's actually a temp directory we created
			if 'openbrowser-tmp-' in str(temp_path):
				shutil.rmtree(temp_path, ignore_errors=True)
		except Exception as e:
			self.logger.debug(f'Failed to cleanup temp dir {temp_dir}: {e}')

	@classmethod
	def _iter_cache_paths(cls, user_data_dir: str | Path, profile_directory: str = 'Default') -> list[Path]:
		"""Return the managed cache paths eligible for cleanup."""
		root = Path(user_data_dir).expanduser()
		profile_cache_paths: list[Path] = []

		for relative_path in cls.CACHE_PATHS:
			if relative_path.startswith('Default/'):
				relative_path = relative_path.replace('Default/', f'{profile_directory}/', 1)
			profile_cache_paths.append(root / relative_path)

		return profile_cache_paths

	@staticmethod
	def _get_path_size_bytes(path: Path) -> int:
		"""Return the recursive size of a file or directory in bytes."""
		try:
			if path.is_file():
				return path.stat().st_size
			if path.is_dir():
				return sum(
					child.stat().st_size for child in path.rglob('*') if child.exists() and not child.is_dir()
				)
		except OSError:
			return 0
		return 0

	@classmethod
	def _cleanup_profile_cache(cls, user_data_dir: str | Path | None, profile_directory: str = 'Default') -> int:
		"""Delete disposable browser caches for OpenBrowser-managed profiles only."""
		if not user_data_dir or not is_openbrowser_managed_profile_dir(user_data_dir):
			return 0

		total_cleared_bytes = 0

		for cache_path in cls._iter_cache_paths(user_data_dir, profile_directory):
			if not cache_path.exists():
				continue

			total_cleared_bytes += cls._get_path_size_bytes(cache_path)
			try:
				if cache_path.is_dir():
					shutil.rmtree(cache_path, ignore_errors=True)
				else:
					cache_path.unlink(missing_ok=True)
			except OSError:
				continue

		return total_cleared_bytes

	@classmethod
	async def _kill_stale_chrome_for_profile(
		cls,
		user_data_dir: str,
		*,
		metadata: dict[str, Any] | None = None,
		instance_id: str | None = None,
	) -> bool:
		"""Close only the recorded browser owned by a stale profile lease.

		A path-only call intentionally does nothing. Matching an arbitrary Chrome
		process by ``--user-data-dir`` is not sufficient proof of ownership.
		"""
		resolved_dir = str(Path(user_data_dir).expanduser().resolve())
		if not metadata or metadata.get('profile_dir') != resolved_dir:
			return False

		owner_status = ProfileLease.process_identity_status(metadata.get('owner_pid'), metadata.get('owner_start_time'))
		if owner_status == 'unknown':
			return False
		if owner_status == 'match' and metadata.get('instance_id') != instance_id:
			return False

		browser = metadata.get('browser')
		if isinstance(browser, dict):
			if browser.get('instance_id') != metadata.get('instance_id'):
				return False
			if browser.get('profile_dir') != resolved_dir:
				return False
			try:
				proc = psutil.Process(int(browser['pid']))
			except psutil.NoSuchProcess:
				descendants, scan_succeeded = cls._find_owned_descendants(
					browser, excluded_pids={int(browser['pid'])}
				)
				if not scan_succeeded:
					return False
				return await cls._cleanup_owned_descendants(descendants, force_kill=not IS_WINDOWS)
			except (psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError, OSError):
				return False

			if not ProfileLease.process_matches_browser(proc, browser):
				return False
			cdp_url = None
			if browser.get('cdp_port'):
				cdp_url = f"http://127.0.0.1:{browser['cdp_port']}/"
			_logger.info(
				f'[LocalBrowserWatchdog] Closing owned stale Chrome process pid={proc.pid} '
				f'for profile {resolved_dir}'
			)
			return await cls._cleanup_process(
				proc,
				browser_record=browser,
				cdp_url=cdp_url,
				require_identity=True,
			)

		# If the owner crashed after launching Chrome but before recording its PID,
		# reclaim only a process carrying the stale owner's exact marker and profile.
		stale_instance_id = metadata.get('instance_id')
		if not stale_instance_id:
			return False
		marker = f'{OWNERSHIP_MARKER_PREFIX}{stale_instance_id}'
		for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
			try:
				process_info = proc.info
				name = (process_info.get('name') or '').lower()
				if not any(browser_name in name for browser_name in cls.BROWSER_PROCESS_NAMES):
					continue
				if process_info.get('cmdline') is None:
					return False
				candidate = {
					'pid': int(proc.pid),
					'start_time': float(proc.create_time()),
					'profile_dir': resolved_dir,
					'instance_id': stale_instance_id,
					'ownership_marker': marker,
					'cdp_port': metadata.get('cdp_port'),
				}
				if not ProfileLease.process_matches_browser(proc, candidate):
					continue
				return await cls._cleanup_process(
					proc,
					browser_record=candidate,
					cdp_url=None,
					require_identity=True,
				)
			except psutil.NoSuchProcess:
				continue
			except (psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError, OSError):
				return False

		return True

	@property
	def browser_pid(self) -> int | None:
		"""Get the browser process ID."""
		if self._subprocess:
			return self._subprocess.pid
		return None

	@staticmethod
	async def get_browser_pid_via_cdp(browser) -> int | None:
		"""Get the browser process ID via CDP SystemInfo.getProcessInfo.

		Args:
			browser: Playwright Browser instance

		Returns:
			Process ID or None if failed
		"""
		try:
			cdp_session = await browser.new_browser_cdp_session()
			result = await cdp_session.send('SystemInfo.getProcessInfo')
			process_info = result.get('processInfo', {})
			pid = process_info.get('id')
			await cdp_session.detach()
			return pid
		except Exception:
			# If we can't get PID via CDP, it's not critical
			return None
