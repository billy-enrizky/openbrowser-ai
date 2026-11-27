"""Tests for daemon socket and orphan-process diagnostics."""

import socket
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest

from openbrowser.daemon import diagnostics


def test_daemon_scan_defers_metadata_until_candidate_matches(tmp_path):
	current = MagicMock(username=MagicMock(return_value='me'))
	ordinary = MagicMock(pid=201)
	ordinary.cmdline.return_value = ['python', 'ordinary.py']
	foreign = MagicMock(pid=202)
	foreign.cmdline.return_value = ['python', '-m', diagnostics.DAEMON_MODULE]
	foreign.username.return_value = 'other-user'
	daemon = MagicMock(pid=203)
	daemon.cmdline.return_value = ['python', '-m', diagnostics.DAEMON_MODULE]
	daemon.username.return_value = 'me'
	daemon.create_time.return_value = 100.0
	daemon.environ.return_value = {'OPENBROWSER_SOCKET': str(tmp_path / 'daemon.sock')}
	processes = [ordinary, foreign, daemon]
	for process in processes:
		process.info = {'cmdline': process.cmdline.return_value, 'create_time': 100.0}
	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=processes) as iterate,
	):
		rows = diagnostics.daemon_processes()
	assert [row['pid'] for row in rows] == [203]
	iterate.assert_called_once_with(['pid'])
	ordinary.username.assert_not_called()
	ordinary.create_time.assert_not_called()
	foreign.create_time.assert_not_called()
	daemon.create_time.assert_called_once_with()


@pytest.mark.parametrize('operation', ['cmdline', 'username', 'create_time'])
@pytest.mark.parametrize(
	'error', [OSError(0, 'native read failed'), psutil.AccessDenied(pid=210), psutil.NoSuchProcess(pid=210)]
)
def test_daemon_scan_contains_each_candidate_read_failure(tmp_path, operation, error):
	current = MagicMock(username=MagicMock(return_value='me'))
	broken = MagicMock(pid=210)
	broken.cmdline.return_value = ['python', '-m', diagnostics.DAEMON_MODULE]
	broken.username.return_value = 'me'
	broken.create_time.return_value = 100.0
	broken.environ.return_value = {'OPENBROWSER_SOCKET': str(tmp_path / 'broken.sock')}
	getattr(broken, operation).side_effect = error
	good = MagicMock(pid=211)
	good.cmdline.return_value = ['python', '-m', diagnostics.DAEMON_MODULE]
	good.username.return_value = 'me'
	good.create_time.return_value = 100.0
	good.environ.return_value = {'OPENBROWSER_SOCKET': str(tmp_path / 'good.sock')}

	def iterate(attrs):
		for process in [broken, good]:
			process.info = {attr: getattr(process, attr)() for attr in attrs if attr != 'pid'}
			yield process

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', side_effect=iterate),
	):
		rows = diagnostics.daemon_processes()
	assert [row['pid'] for row in rows] == [211]


def test_socket_candidates_requires_a_pid_file_for_unlisted_sockets(tmp_path):
	daemon_dir = tmp_path / 'openbrowser'
	daemon_dir.mkdir()
	(daemon_dir / 'one.sock').write_text('')
	(daemon_dir / 'one.pid').write_text('123')
	(daemon_dir / 'unrelated.sock').write_text('')
	current = daemon_dir / 'current.sock'
	with (
		patch.object(diagnostics, 'DAEMON_DIR', daemon_dir),
		patch.object(diagnostics, 'get_socket_path', return_value=current),
	):
		candidates = diagnostics.socket_candidates()

	assert (daemon_dir / 'one.sock') in candidates
	assert (daemon_dir / 'unrelated.sock') not in candidates
	assert current in candidates


def test_daemon_processes_filters_commands_and_users(tmp_path):
	current = MagicMock()
	current.username.return_value = 'me'
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {
		'pid': 123,
		'create_time': time.time() - 60,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	daemon.cmdline.return_value = daemon.info['cmdline']
	daemon.create_time.return_value = daemon.info['create_time']
	daemon.username.return_value = 'me'
	daemon.environ.return_value = {'OPENBROWSER_SOCKET': str(tmp_path / 'daemon.sock')}
	other_user = MagicMock()
	other_user.pid = 124
	other_user.info = {'pid': 124, 'create_time': time.time(), 'cmdline': ['python', '-m', 'openbrowser.daemon.server']}
	other_user.cmdline.return_value = other_user.info['cmdline']
	other_user.create_time.return_value = other_user.info['create_time']
	other_user.username.return_value = 'someone-else'
	unrelated = MagicMock()
	unrelated.pid = 125
	unrelated.info = {'pid': 125, 'create_time': time.time(), 'cmdline': ['python', 'script.py']}
	unrelated.cmdline.return_value = unrelated.info['cmdline']
	unrelated.create_time.return_value = unrelated.info['create_time']
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
	assert diagnostics.is_daemon_cmdline(['/usr/bin/python3.12', '-m', 'openbrowser.daemon.server'])
	assert not diagnostics.is_daemon_cmdline(['python', '-m', 'openbrowser.daemon.server_extra'])
	assert not diagnostics.is_daemon_cmdline(['python', 'openbrowser.daemon.server'])
	assert not diagnostics.is_daemon_cmdline(['python', '-m'])
	assert not diagnostics.is_daemon_cmdline(['python', 'worker.py', '-m', 'openbrowser.daemon.server'])
	assert not diagnostics.is_daemon_cmdline(['python', '-c', 'print(1)', '-m', 'openbrowser.daemon.server'])
	assert not diagnostics.is_daemon_cmdline(['node', '-m', 'openbrowser.daemon.server'])


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
	lookalike.cmdline.return_value = lookalike.info['cmdline']
	lookalike.create_time.return_value = lookalike.info['create_time']
	lookalike.username.return_value = 'me'

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[lookalike]),
	):
		rows = diagnostics.daemon_processes()

	assert rows == []


def test_daemon_processes_treats_missing_environment_as_unknown(tmp_path):
	current = MagicMock()
	current.username.return_value = 'me'
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {
		'pid': 123,
		'create_time': time.time() - 60,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	daemon.cmdline.return_value = daemon.info['cmdline']
	daemon.create_time.return_value = daemon.info['create_time']
	daemon.username.return_value = 'me'
	daemon.environ.return_value = {}
	caller_socket = tmp_path / 'caller.sock'

	with (
		patch.object(diagnostics, 'SOCKET_PATH', caller_socket),
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[daemon]),
	):
		rows = diagnostics.daemon_processes()

	assert rows[0]['socket'] is None
	assert rows[0]['socket_known'] is False
	assert rows[0]['socket_live'] is True


def test_daemon_processes_derives_socket_from_daemon_home(tmp_path):
	current = MagicMock()
	current.username.return_value = 'me'
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {
		'pid': 123,
		'create_time': time.time() - 60,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	daemon.cmdline.return_value = daemon.info['cmdline']
	daemon.create_time.return_value = daemon.info['create_time']
	daemon.username.return_value = 'me'
	daemon.environ.return_value = {'HOME': str(tmp_path)}

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=current),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[daemon]),
	):
		rows = diagnostics.daemon_processes()

	assert rows[0]['socket'] == str(tmp_path / '.openbrowser' / 'daemon.sock')
	assert rows[0]['socket_known'] is True


def test_daemon_processes_expands_tilde_socket_paths(tmp_path, monkeypatch):
	monkeypatch.setenv('HOME', str(tmp_path))
	socket_path = tmp_path / '.openbrowser' / 'daemon.sock'
	socket_path.parent.mkdir(parents=True)
	socket_path.write_text('')
	current = MagicMock(username=MagicMock(return_value='me'))
	daemon = MagicMock()
	daemon.pid = 123
	daemon.info = {
		'pid': 123,
		'create_time': time.time() - 60,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	daemon.cmdline.return_value = daemon.info['cmdline']
	daemon.create_time.return_value = daemon.info['create_time']
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
	daemon.info = {
		'pid': 123,
		'create_time': time.time() - 60,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	daemon.cmdline.return_value = daemon.info['cmdline']
	daemon.create_time.return_value = daemon.info['create_time']
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
	orphan.info = {
		'pid': 123,
		'create_time': time.time() - 100,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	orphan.cmdline.return_value = orphan.info['cmdline']
	orphan.create_time.return_value = orphan.info['create_time']
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
	live.info = {
		'pid': 123,
		'create_time': time.time() - 100,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	live.cmdline.return_value = live.info['cmdline']
	live.create_time.return_value = live.info['create_time']
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


def test_reap_skips_process_with_missing_environment():
	live = MagicMock()
	live.pid = 123
	live.info = {
		'pid': 123,
		'create_time': time.time() - 100,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	live.cmdline.return_value = live.info['cmdline']
	live.create_time.return_value = live.info['create_time']
	live.username.return_value = 'me'
	live.environ.return_value = {}

	with (
		patch.object(diagnostics.psutil, 'Process', return_value=MagicMock(username=MagicMock(return_value='me'))),
		patch.object(diagnostics.psutil, 'process_iter', return_value=[live]),
	):
		results = diagnostics.reap_orphan_processes(min_age_seconds=30, timeout=0.1)

	assert results[0]['reaped'] is False
	assert results[0]['socket_known'] is False
	assert results[0]['reason'] == 'socket is live or unknown'


def test_reap_skips_process_when_environment_is_unreadable(tmp_path):
	opaque = MagicMock()
	opaque.pid = 123
	opaque.info = {
		'pid': 123,
		'create_time': time.time() - 100,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	opaque.cmdline.return_value = opaque.info['cmdline']
	opaque.create_time.return_value = opaque.info['create_time']
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
	young.cmdline.return_value = young.info['cmdline']
	young.create_time.return_value = young.info['create_time']
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
	orphan.info = {
		'pid': 123,
		'create_time': time.time() - 100,
		'cmdline': ['python', '-m', 'openbrowser.daemon.server'],
	}
	orphan.cmdline.return_value = orphan.info['cmdline']
	orphan.create_time.return_value = orphan.info['create_time']
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
