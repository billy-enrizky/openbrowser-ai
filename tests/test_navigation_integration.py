import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit
from uuid import uuid4
from unittest.mock import patch

import psutil
import pytest

from openbrowser.browser import BrowserSession
from openbrowser.browser.navigation import NavigationBlockedError, NavigationBusyError

pytestmark = pytest.mark.integration

_PIXEL = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=')


class _NavigationServer(ThreadingHTTPServer):

	def __init__(self):
		super().__init__(('127.0.0.1', 0), _NavigationHandler)
		self.daemon_threads = True
		self.resource_requested = threading.Event()
		self.release_resource = threading.Event()
		self.requests: list[str] = []


class _NavigationHandler(BaseHTTPRequestHandler):

	def log_message(self, *_args):
		return

	def do_GET(self):
		server = self.server
		assert isinstance(server, _NavigationServer)
		server.requests.append(self.path)
		path = urlsplit(self.path)
		query = parse_qs(path.query)
		if path.path == '/redirect':
			self.send_response(302)
			self.send_header('Location', '/page?' + urlencode({'nonce': query['nonce'][0]}))
			self.end_headers()
			return
		if path.path == '/forbidden-redirect':
			self.send_response(302)
			self.send_header('Location', f'http://localhost:{server.server_port}/page?nonce=blocked')
			self.end_headers()
			return
		if path.path == '/forbidden-delayed-redirect':
			self.send_response(302)
			self.send_header('Location', f'http://localhost:{server.server_port}/delayed?nonce=blocked-before-load')
			self.end_headers()
			return
		if path.path == '/slow.png':
			server.resource_requested.set()
			server.release_resource.wait(timeout=15)
			body, content_type = _PIXEL, 'image/png'
		elif path.path == '/download':
			body, content_type = b'navigation download fixture\n', 'application/octet-stream'
		elif path.path == '/document.pdf':
			from test_mcp_download_e2e import MINIMAL_PDF
			body, content_type = MINIMAL_PDF, 'application/pdf'
		else:
			nonce = query.get('nonce', ['fixture'])[0]
			image = '<img id="slow" src="/slow.png">' if path.path == '/delayed' else ''
			body = (
				f'<!doctype html><html lang="en"><head><title>{nonce}</title>'
				'<link rel="icon" href="data:,"></head><body>'
				f'<h1 id="marker">{nonce}</h1>{image}'
				'<script>addEventListener("load", () => document.documentElement.dataset.loaded = "yes")</script>'
				'</body></html>'
			).encode()
			content_type = 'text/html; charset=utf-8'
		self.send_response(200)
		self.send_header('Content-Type', content_type)
		self.send_header('Content-Length', str(len(body)))
		self.send_header('Cache-Control', 'no-store')
		if path.path == '/download':
			self.send_header('Content-Disposition', 'attachment; filename="navigation-fixture.txt"')
		self.end_headers()
		try:
			self.wfile.write(body)
		except (BrokenPipeError, ConnectionResetError):
			pass


@pytest.fixture
def navigation_site():
	server = _NavigationServer()
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()
	try:
		yield server, f'http://127.0.0.1:{server.server_port}'
	finally:
		server.release_resource.set()
		server.shutdown()
		server.server_close()
		thread.join(timeout=5)
		assert not thread.is_alive()


@pytest.fixture
async def navigation_browser(tmp_path):
	profile = tmp_path / 'profile'
	browser = BrowserSession(
		user_data_dir=str(profile), headless=True, enable_default_extensions=False,
		downloads_path=str(tmp_path / 'downloads'), auto_download_pdfs=False, keep_alive=False,
	)
	identities = []
	try:
		await asyncio.wait_for(browser.start(), timeout=45)
		pid = browser._local_browser_watchdog.browser_pid
		if pid:
			identities.append((pid, psutil.Process(pid).create_time()))
		yield browser
	finally:
		watchdog = browser._local_browser_watchdog
		if watchdog is not None and watchdog.browser_pid:
			try:
				pid = watchdog.browser_pid
				identities.append((pid, psutil.Process(pid).create_time()))
			except psutil.NoSuchProcess:
				pass
		await asyncio.wait_for(browser.kill(), timeout=30)
		for pid, created in set(identities):
			try:
				process = psutil.Process(pid)
				assert process.create_time() != created or not process.is_running()
			except psutil.NoSuchProcess:
				pass


async def _document(browser):
	focus = browser.agent_focus
	assert focus is not None
	response = await focus.cdp_client.send.Runtime.evaluate(
		params={
			'expression': 'JSON.stringify({url:location.href,title:document.title,ready:document.readyState,loaded:document.documentElement.dataset.loaded,marker:document.querySelector("#marker")?.textContent})',
			'returnByValue': True,
		},
		session_id=focus.session_id,
	)
	return json.loads(response['result']['value'])


async def _navigate(browser, *args, **kwargs):
	return await asyncio.wait_for(browser.navigate_to(*args, **kwargs), timeout=15)


async def test_default_navigation_waits_for_delayed_load(navigation_browser, navigation_site):
	browser = navigation_browser
	server, base = navigation_site
	nonce = uuid4().hex

	async def navigate_and_observe():
		await _navigate(browser, f'{base}/delayed?nonce={nonce}')
		return await _document(browser)

	task = asyncio.create_task(navigate_and_observe())
	try:
		assert await asyncio.to_thread(server.resource_requested.wait, 10)
		await asyncio.sleep(1.25)
		server.release_resource.set()
		observed = await asyncio.wait_for(task, timeout=10)
		assert observed['loaded'] == 'yes'
		assert observed['ready'] == 'complete'
		assert observed['marker'] == nonce
	finally:
		server.release_resource.set()
		if not task.done():
			task.cancel()
		await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('milestone', ['commit', 'domcontentloaded', 'load', 'networkidle'])
async def test_declared_navigation_milestones(navigation_browser, navigation_site, milestone):
	browser = navigation_browser
	_, base = navigation_site
	nonce = uuid4().hex
	await _navigate(browser, f'{base}/page?nonce={nonce}', wait_until=milestone, timeout_ms=10000)
	observed = await _document(browser)
	assert observed['url'] == f'{base}/page?nonce={nonce}'
	if milestone != 'commit':
		assert observed['marker'] == nonce
		assert observed['title'] == nonce
	state = browser.lifecycle_snapshot()['navigation']
	assert state['phase'] == 'succeeded'
	assert state['wait_until'] == milestone


async def test_redirect_then_fragment_uses_observed_url(navigation_browser, navigation_site):
	browser = navigation_browser
	_, base = navigation_site
	nonce = uuid4().hex
	await _navigate(browser, f'{base}/redirect?nonce={nonce}')
	url = f'{base}/page?nonce={nonce}'
	assert (await _document(browser))['url'] == url
	assert browser.lifecycle_snapshot()['navigation']['observed_url'] == url
	await _navigate(browser, url + '#fragment', timeout_ms=5000)
	assert (await _document(browser))['url'] == url + '#fragment'
	assert browser.lifecycle_snapshot()['navigation']['observed_url'] == url + '#fragment'


async def test_public_cancellation_aborts_only_its_navigation(navigation_browser, navigation_site):
	browser = navigation_browser
	server, base = navigation_site
	task = asyncio.create_task(_navigate(browser, f'{base}/delayed?nonce=cancel'))
	try:
		assert await asyncio.to_thread(server.resource_requested.wait, 10)
		task.cancel()
		with pytest.raises(asyncio.CancelledError):
			await task
		for _ in range(100):
			if browser.lifecycle_snapshot()['navigation']['phase'] == 'cancelled':
				break
			await asyncio.sleep(0.01)
		assert browser.lifecycle_snapshot()['navigation']['phase'] == 'cancelled'
		server.release_resource.set()
		await _navigate(browser, f'{base}/page?nonce=after-cancel', timeout_ms=5000)
		assert (await _document(browser))['marker'] == 'after-cancel'
		assert sum(path.startswith('/delayed?') for path in server.requests) == 1
	finally:
		server.release_resource.set()
		if not task.done():
			task.cancel()
		await asyncio.gather(task, return_exceptions=True)


async def test_public_overlap_is_refused_without_replay(navigation_browser, navigation_site):
	browser = navigation_browser
	server, base = navigation_site
	task = asyncio.create_task(_navigate(browser, f'{base}/delayed?nonce=first'))
	try:
		assert await asyncio.to_thread(server.resource_requested.wait, 10)
		with pytest.raises(NavigationBusyError):
			await _navigate(browser, f'{base}/page?nonce=must-not-run')
		assert not any('must-not-run' in path for path in server.requests)
		server.release_resource.set()
		await asyncio.wait_for(task, timeout=10)
	finally:
		server.release_resource.set()
		if not task.done():
			task.cancel()
		await asyncio.gather(task, return_exceptions=True)


async def test_download_handoff_is_not_document_readiness(navigation_browser, navigation_site):
	browser = navigation_browser
	_, base = navigation_site
	await _navigate(browser, f'{base}/page?nonce=before-download')
	assert await _navigate(browser, f'{base}/download', timeout_ms=5000) is None
	state = browser.lifecycle_snapshot()['navigation']
	assert state['phase'] == 'download'
	assert (await _document(browser))['marker'] == 'before-download'


async def test_forbidden_input_never_reaches_the_server(navigation_browser, navigation_site):
	browser = navigation_browser
	server, _ = navigation_site
	browser.browser_profile.allowed_domains = ['127.0.0.1']
	with pytest.raises(NavigationBlockedError, match='blocked|policy|disallowed'):
		await _navigate(browser, f'http://localhost:{server.server_port}/page?nonce=forbidden')
	assert not any('nonce=forbidden' in path for path in server.requests)


async def test_forbidden_redirect_fails_and_leaves_about_blank(navigation_browser, navigation_site):
	browser = navigation_browser
	_, base = navigation_site
	browser.browser_profile.allowed_domains = ['127.0.0.1']
	with pytest.raises(NavigationBlockedError, match='blocked|policy|disallowed'):
		await _navigate(browser, f'{base}/forbidden-redirect')
	assert (await _document(browser))['url'] == 'about:blank'
	assert browser.lifecycle_snapshot()['navigation']['phase'] == 'failed'


async def test_forbidden_committed_document_is_contained_while_image_stays_blocked(navigation_browser, navigation_site):
	browser = navigation_browser
	server, base = navigation_site
	browser.browser_profile.allowed_domains = ['127.0.0.1']
	try:
		with pytest.raises(NavigationBlockedError, match='blocked|policy|disallowed'):
			await _navigate(browser, f'{base}/forbidden-delayed-redirect', timeout_ms=2000)
		assert server.resource_requested.is_set()
		assert not server.release_resource.is_set()
		assert (await _document(browser))['url'] == 'about:blank'
		state = browser.lifecycle_snapshot()['navigation']
		assert state['containment']['phase'] == 'contained'
	finally:
		server.release_resource.set()


@pytest.mark.parametrize('ending', ['timeout', 'cancel'])
async def test_forbidden_commit_keeps_document_cleanup_after_primary_end(navigation_browser, navigation_site, ending):
	from openbrowser.browser.navigation import NavigationTimeoutError
	browser = navigation_browser
	server, base = navigation_site
	browser.browser_profile.allowed_domains = ['127.0.0.1']
	root = browser._cdp_client_root
	original = root.send.Runtime.callFunctionOn
	entered, release = asyncio.Event(), asyncio.Event()
	async def held_action(*args, **kwargs):
		if 'this.location.replace' in kwargs.get('params', {}).get('functionDeclaration', ''):
			entered.set()
			await release.wait()
		return await original(*args, **kwargs)
	task = None
	try:
		with patch.object(root.send.Runtime, 'callFunctionOn', held_action):
			task = asyncio.create_task(browser.navigate_to(f'{base}/forbidden-delayed-redirect', timeout_ms=1000 if ending == 'timeout' else 10000))
			await asyncio.wait_for(entered.wait(), 5)
			op = browser._navigation_monitor.active
			if ending == 'cancel':
				task.cancel()
			while not op.terminal:
				op._changed.clear()
				await asyncio.wait_for(op._changed.wait(), 3)
			release.set()
			result, = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
			assert isinstance(result, NavigationTimeoutError if ending == 'timeout' else asyncio.CancelledError)
			await asyncio.wait_for(asyncio.shield(op.policy_task), 3)
			assert op.containment.phase.value == 'contained'
			assert (await _document(browser))['url'] == 'about:blank'
			assert op.phase.value == ('timed_out' if ending == 'timeout' else 'cancelled')
	finally:
		release.set()
		server.release_resource.set()
		if task is not None:
			await asyncio.gather(task, return_exceptions=True)


async def test_kill_retires_sent_policy_by_owned_exit_before_restart(navigation_browser, navigation_site):
	browser = navigation_browser
	server, base = navigation_site
	browser.browser_profile.allowed_domains = ['127.0.0.1']
	root = browser._cdp_client_root
	original = root.send.Runtime.callFunctionOn
	entered, release = asyncio.Event(), asyncio.Event()
	async def delayed_reply(*args, **kwargs):
		result = await original(*args, **kwargs)
		if 'this.location.replace' in kwargs.get('params', {}).get('functionDeclaration', ''):
			entered.set()
			await release.wait()
		return result
	task = None
	try:
		with patch.object(root.send.Runtime, 'callFunctionOn', delayed_reply):
			task = asyncio.create_task(browser.navigate_to(f'{base}/forbidden-delayed-redirect', timeout_ms=10000))
			await asyncio.wait_for(entered.wait(), 5)
			op = browser._navigation_monitor.active
			assert op.containment.owned_browser is not None
			pid, created, _profile, _instance = op.containment.owned_browser
			await asyncio.wait_for(browser.kill(), 20)
			assert op.containment.phase.value == 'revoked'
			try:
				assert psutil.Process(pid).create_time() != created
			except psutil.NoSuchProcess:
				pass
			release.set()
			await asyncio.gather(task, return_exceptions=True)
		server.release_resource.set()
		await asyncio.wait_for(browser.start(), 30)
		await _navigate(browser, f'{base}/page?nonce=after-policy-kill', timeout_ms=5000)
		assert (await _document(browser))['marker'] == 'after-policy-kill'
	finally:
		release.set()
		server.release_resource.set()
		if task is not None:
			await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize('successor', ['same_url_new_loader', 'changed_hash'])
async def test_policy_document_action_cannot_blank_successor_after_host_postcheck(navigation_browser, navigation_site, successor):
	browser = navigation_browser
	_, base = navigation_site
	browser.browser_profile.allowed_domains = ['127.0.0.1']
	root = browser._cdp_client_root
	original = root.send.Runtime.callFunctionOn
	observed = {}
	async def install_successor(*args, **kwargs):
		params = kwargs.get('params', {})
		if 'this.location.replace' in params.get('functionDeclaration', ''):
			assert not observed
			sid = kwargs['session_id']
			before = (await root.send.Page.getFrameTree(session_id=sid))['frameTree']['frame']
			url = params['arguments'][0]['value']
			if successor == 'same_url_new_loader':
				response = await root.send.Page.navigate(params={'url': url}, session_id=sid)
				after = (await root.send.Page.getFrameTree(session_id=sid))['frameTree']['frame']
				assert after['loaderId'] == response['loaderId'] != before['loaderId']
			else:
				await root.send.Runtime.evaluate(params={'expression': 'history.replaceState(null,"",location.href+"#operator")'}, session_id=sid)
				after = (await root.send.Page.getFrameTree(session_id=sid))['frameTree']['frame']
				assert after['loaderId'] == before['loaderId']
			observed.update(loader_id=after['loaderId'], url=after.get('url', '') + after.get('urlFragment', ''))
		return await original(*args, **kwargs)
	with patch.object(root.send.Runtime, 'callFunctionOn', install_successor):
		with pytest.raises(NavigationBlockedError):
			await _navigate(browser, f'{base}/forbidden-redirect', timeout_ms=5000)
	assert observed
	frame = (await root.send.Page.getFrameTree(session_id=browser.agent_focus.session_id))['frameTree']['frame']
	assert frame['loaderId'] == observed['loader_id']
	assert (await _document(browser))['url'] == observed['url'] != 'about:blank'
	assert browser.lifecycle_snapshot()['navigation']['containment']['phase'] == 'skipped'


async def test_policy_owner_contains_commit_while_initial_navigation_reply_is_held(navigation_browser, navigation_site):
	browser = navigation_browser
	server, base = navigation_site
	browser.browser_profile.allowed_domains = ['127.0.0.1']
	root = browser._cdp_client_root
	original = root.send.Page.navigate
	release = asyncio.Event()
	async def held_reply(*args, **kwargs):
		response = await original(*args, **kwargs)
		if kwargs.get('params', {}).get('url') == f'{base}/forbidden-delayed-redirect':
			await release.wait()
		return response
	try:
		with patch.object(root.send.Page, 'navigate', held_reply):
			with pytest.raises(NavigationBlockedError):
				await _navigate(browser, f'{base}/forbidden-delayed-redirect', timeout_ms=5000)
		state = browser.lifecycle_snapshot()['navigation']
		assert state['frame_id'] is None and state['loader_id'] is None
		assert state['containment']['phase'] == 'contained'
		assert (await _document(browser))['url'] == 'about:blank'
	finally:
		release.set()
		server.release_resource.set()


async def test_domcontentloaded_does_not_wait_for_a_blocked_image(navigation_browser, navigation_site):
	browser = navigation_browser
	server, base = navigation_site
	try:
		await _navigate(browser, f'{base}/delayed?nonce=dcl', wait_until='domcontentloaded', timeout_ms=5000)
		observed = await _document(browser)
		assert observed['marker'] == 'dcl'
		assert observed.get('loaded') != 'yes'
		assert not server.release_resource.is_set()
	finally:
		server.release_resource.set()


@pytest.mark.parametrize('stop_method', ['kill', 'stop'])
async def test_shutdown_during_navigation_does_not_poison_restart(navigation_browser, navigation_site, stop_method):
	from openbrowser.browser.navigation import NavigationError

	browser = navigation_browser
	server, base = navigation_site
	generation = browser.lifecycle_snapshot()['generation']
	task = asyncio.create_task(_navigate(browser, f'{base}/delayed?nonce=shutdown'))
	try:
		assert await asyncio.to_thread(server.resource_requested.wait, 10)
		await asyncio.wait_for(getattr(browser, stop_method)(), timeout=20)
		result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=5)
		assert isinstance(result[0], (NavigationError, asyncio.CancelledError))
		assert browser.lifecycle_snapshot()['phase'] == 'stopped'
		assert browser.lifecycle_snapshot()['navigation']['phase'] == 'cancelled'
		server.release_resource.set()
		await asyncio.wait_for(browser.start(), timeout=30)
		await _navigate(browser, f'{base}/page?nonce=restarted', timeout_ms=5000)
		assert browser.lifecycle_snapshot()['generation'] > generation
		assert (await _document(browser))['marker'] == 'restarted'
	finally:
		server.release_resource.set()
		if not task.done():
			task.cancel()
		await asyncio.gather(task, return_exceptions=True)


async def test_pdf_navigation_has_a_bounded_document_or_download_outcome(navigation_browser, navigation_site):
	browser = navigation_browser
	_, base = navigation_site
	await _navigate(browser, f'{base}/page?nonce=before-pdf')
	await _navigate(browser, f'{base}/document.pdf', timeout_ms=10000)
	state = browser.lifecycle_snapshot()['navigation']
	assert state['phase'] in {'succeeded', 'download'}
	if state['phase'] == 'succeeded':
		assert state['observed_url'] == f'{base}/document.pdf'
	else:
		assert state['observed_url'] == f'{base}/page?nonce=before-pdf'
