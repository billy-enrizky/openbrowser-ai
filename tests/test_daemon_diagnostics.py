"""Tests for daemon socket and orphan-process diagnostics."""

import socket
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil

from openbrowser.daemon import diagnostics


def test_socket_candidates_includes_current_socket(tmp_path):
	daemon_dir = tmp_path / 'openbrowser'
	daemon_dir.mkdir()
	(daemon_dir / 'one.sock').write_text('')
	current = daemon_dir / 'current.sock'
	with (
		patch.object(diagnostics, 'DAEMON_DIR', daemon_dir),
		patch.object(diagnostics, 'get_socket_path', return_value=current),
	):
		candidates = diagnostics.socket_candidates()

	assert (daemon_dir / 'one.sock') in candidates
	assert current in candidates


def test_daemon_processes_filters_commands_and_users(tmp_path):
	current = MagicMock()
	current.username.return_value = 'me'
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {'pid': 123, 'create_time': time.time() - 60, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	daemon.username.return_value = 'me'
	daemon.environ.return_value = {'OPENBROWSER_SOCKET': str(tmp_path / 'daemon.sock')}
	other_user = MagicMock()
	other_user.pid = 124
	other_user.info = {'pid': 124, 'create_time': time.time(), 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	other_user.username.return_value = 'someone-else'
	unrelated = MagicMock()
	unrelated.pid = 125
	unrelated.info = {'pid': 125, 'create_time': time.time(), 'cmdline': ['python', 'script.py']}
	unrelated.username.return_value = 'me'

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[daemon, other_user, unrelated]),
	):
		rows = diagnostics.daemon_processes()

	assert len(rows) == 1
	assert rows[0]['pid'] == 123
	assert rows[0]['socket'] == str(tmp_path / 'daemon.sock')


def test_is_daemon_cmdline_requires_an_exact_module_argument():
	assert diagnostics.is_daemon_cmdline(['python', '-m', 'openbrowser.daemon.server'])
	assert not diagnostics.is_daemon_cmdline(['python', '-m', 'openbrowser.daemon.server_extra'])
	assert not diagnostics.is_daemon_cmdline(['python', 'openbrowser.daemon.server'])
	assert not diagnostics.is_daemon_cmdline(['python', '-m'])


def test_daemon_processes_ignores_a_lookalike_module(tmp_path):
	current = MagicMock()
	current.username.return_value = 'me'
	lookalike = MagicMock()
	lookalike.pid = 321
	lookalike.info = {
		'pid': 321,
		'create_time': time.time() - 60,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server_extra'],
	}
	lookalike.username.return_value = 'me'

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[lookalike]),
	):
		rows = diagnostics.daemon_processes()

	assert rows == []


def test_daemon_processes_uses_default_socket_when_env_is_missing(tmp_path):
	current = MagicMock()
	current.username.return_value = 'me'
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {'pid': 123, 'create_time': time.time() - 60, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	daemon.username.return_value = 'me'
	daemon.environ.return_value = {}
	default_socket = tmp_path / 'default.sock'

	with (
		patch.object(diagnostics, 'SOCKET_PATH', default_socket),
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[daemon]),
	):
		rows = diagnostics.daemon_processes()

	assert rows[0]['socket'] == str(default_socket)
	assert rows[0]['socket_known'] is True


def test_daemon_processes_expands_tilde_socket_paths(tmp_path, monkeypatch):
	monkeypatch.setenv('HOME', str(tmp_path))
	socket_path = tmp_path / '.openbrowser' / 'daemon.sock'
	socket_path.parent.mkdir(parents=True)
	socket_path.write_text('')
	current = MagicMock(username=MagicMock(return_value='me'))
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {'pid': 123, 'create_time': time.time() - 60, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	daemon.username.return_value = 'me'
	daemon.environ.return_value = {'OPENBROWSER_SOCKET': '~/.openbrowser/daemon.sock'}

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[daemon]),
		patch.object(diagnostics, '_unix_socket_is_live', return_value=True),
	):
		rows = diagnostics.daemon_processes()

	assert rows[0]['socket'] == str(socket_path)
	assert rows[0]['socket_live'] is True


def test_daemon_processes_treats_relative_socket_paths_as_unknown(monkeypatch):
	current = MagicMock(username=MagicMock(return_value='me'))
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {'pid': 123, 'create_time': time.time() - 60, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	daemon.username.return_value = 'me'
	daemon.environ.return_value = {'OPENBROWSER_SOCKET': 'relative.sock'}

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[daemon]),
	):
		rows = diagnostics.daemon_processes()

	assert rows[0]['socket'] is None
	assert rows[0]['socket_known'] is False
	assert rows[0]['socket_live'] is True


def test_reap_orphan_processes_kills_old_orphan(tmp_path):
	missing_socket = Path(tempfile.mktemp(suffix='.sock', dir='/tmp'))
	orphan = MagicMock()
	orphan.pid = 123
	orphan.info = {'pid': 123, 'create_time': time.time() - 100, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	orphan.username.return_value = 'me'
	orphan.environ.return_value = {'OPENBROWSER_SOCKET': str(missing_socket)}
	process = MagicMock()
	process.wait = MagicMock()
	process.username.return_value = 'me'
	process.create_time.return_value = orphan.info['create_time']
	process.cmdline.return_value = orphan.info['cmdline']

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=process),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[orphan]),
	):
		results = diagnostics.reap_orphan_processes(min_age_seconds=30, timeout=0.1)

	assert results[0]['reaped'] is True, str(results[0])
	process.terminate.assert_called_once()


def test_reap_skips_live_socket(tmp_path):
	socket_path = Path(tempfile.mktemp(suffix='.sock', dir='/tmp'))
	listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	listener.bind(str(socket_path))
	listener.listen(1)
	live = MagicMock()
	live.pid = 123
	live.info = {'pid': 123, 'create_time': time.time() - 100, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	live.username.return_value = 'me'
	live.environ.return_value = {'OPENBROWSER_SOCKET': str(socket_path)}

	try:
		with (
			patch.object(diagnostics.psutil, 'Process', return_value=MagicMock(username=MagicMock(return_value='me'))),
			patch.object(diagnostics.psutil, 'process_iter', return_value=[live]),
		):
			results = diagnostics.reap_orphan_processes(min_age_seconds=30, timeout=0.1)
	finally:
		listener.close()
		socket_path.unlink(missing_ok=True)

	assert results[0]['reaped'] is False
	assert results[0]['reason'] == 'socket is live or unknown'


def test_reap_skips_live_default_socket_without_env():
	socket_path = Path(tempfile.mktemp(suffix='.sock', dir='/tmp'))
	listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	listener.bind(str(socket_path))
	listener.listen(1)
	live = MagicMock()
	live.pid = 123
	live.info = {'pid': 123, 'create_time': time.time() - 100, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	live.username.return_value = 'me'
	live.environ.return_value = {}

	try:
		with (
			patch.object(diagnostics, 'SOCKET_PATH', socket_path),
			patch.object(diagnostics.psutil, 'Process', return_value=MagicMock(username=MagicMock(return_value='me'))),
			patch.object(diagnostics.psutil, 'process_iter', return_value=[live]),
		):
			results = diagnostics.reap_orphan_processes(min_age_seconds=30, timeout=0.1)
	finally:
		listener.close()
		socket_path.unlink(missing_ok=True)

	assert results[0]['reaped'] is False
	assert results[0]['reason'] == 'socket is live or unknown'


def test_reap_skips_process_when_environment_is_unreadable(tmp_path):
	opaque = MagicMock()
	opaque.pid = 123
	opaque.info = {'pid': 123, 'create_time': time.time() - 100, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	opaque.username.return_value = 'me'
	opaque.environ.side_effect = psutil.AccessDenied(pid=123)

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=MagicMock(username=MagicMock(return_value='me'))),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[opaque]),
	):
		results = diagnostics.reap_orphan_processes(min_age_seconds=30, timeout=0.1)

	assert results[0]['reaped'] is False
	assert results[0]['socket_known'] is False


def test_reap_skips_young_process(tmp_path):
	young = MagicMock()
	young.pid = 123
	young.info = {'pid': 123, 'create_time': time.time(), 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	young.username.return_value = 'me'
	young.environ.return_value = {'OPENBROWSER_SOCKET': str(tmp_path / 'missing.sock')}

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=MagicMock(username=MagicMock(return_value='me'))),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[young]),
	):
		results = diagnostics.reap_orphan_processes(min_age_seconds=30, timeout=0.1)

	assert results[0]['reaped'] is False
	assert results[0]['reason'] == 'process is too young'


def test_reap_skips_a_reused_pid(tmp_path):
	orphan = MagicMock()
	orphan.pid = 123
	orphan.info = {'pid': 123, 'create_time': time.time() - 100, 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	orphan.username.return_value = 'me'
	orphan.environ.return_value = {'OPENBROWSER_SOCKET': str(tmp_path / 'missing.sock')}
	reused = MagicMock()
	reused.username.return_value = 'me'
	reused.create_time.return_value = time.time() - 1
	reused.cmdline.return_value = ['python', '-m', 'openbrowser.daemon.server']

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=reused),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[orphan]),
	):
		results = diagnostics.reap_orphan_processes(min_age_seconds=30, timeout=0.1)

	assert results[0]['reaped'] is False
	assert results[0]['reason'] == 'process identity changed'
	reused.terminate.assert_not_called()
