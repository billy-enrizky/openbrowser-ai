"""Tests for ownership-safe browser profile leases."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import psutil
import pytest

from openbrowser.browser.watchdogs.profile_lease import ProfileInUseError, ProfileLease


def _owner_start_time() -> float:
	return psutil.Process(os.getpid()).create_time()


def test_profile_lease_is_exclusive_and_records_owner(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	lease.write_metadata()

	try:
		metadata = ProfileLease.read_metadata(tmp_path / 'profile')
		assert metadata is not None
		assert metadata['instance_id'] == 'instance-a'
		assert metadata['owner_pid'] == os.getpid()
		assert metadata['owner_start_time'] == _owner_start_time()

		other = ProfileLease(tmp_path / 'profile', instance_id='instance-b')
		with pytest.raises(ProfileInUseError, match='Browser profile already in use'):
			other.acquire()
	finally:
		lease.release()

	assert ProfileLease.read_metadata(tmp_path / 'profile') is None


def test_malformed_metadata_is_ignored(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	(profile_dir / ProfileLease.METADATA_FILENAME).write_text('{not-json', encoding='utf-8')

	assert ProfileLease.read_metadata(profile_dir) is None


def test_write_metadata_keeps_browser_identity(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	try:
		lease.write_metadata(
			browser={
				'pid': 123,
				'start_time': 456.5,
				'profile_dir': str((tmp_path / 'profile').resolve()),
				'instance_id': 'instance-a',
				'ownership_marker': '--openbrowser-instance-id=instance-a',
			}
		)
		metadata = ProfileLease.read_metadata(tmp_path / 'profile')
		assert metadata is not None
		assert metadata['browser']['pid'] == 123
		assert metadata['browser']['start_time'] == 456.5
	finally:
		lease.release()


def test_process_identity_rejects_pid_reuse():
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0

	assert ProfileLease.process_matches_identity(process, 123, 20.0) is True
	assert ProfileLease.process_matches_identity(process, 123, 19.0) is False
	assert ProfileLease.process_matches_identity(process, 124, 20.0) is False


def test_browser_identity_requires_profile_and_instance_marker(tmp_path: Path):
	profile_dir = (tmp_path / 'profile').resolve()
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.cmdline.return_value = [
		'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		f'--user-data-dir={profile_dir}',
		'--openbrowser-instance-id=instance-a',
	]
	record = {
		'pid': 123,
		'start_time': 20.0,
		'profile_dir': str(profile_dir),
		'instance_id': 'instance-a',
		'ownership_marker': '--openbrowser-instance-id=instance-a',
	}

	assert ProfileLease.process_matches_browser(process, record) is True

	process.cmdline.return_value[-1] = '--openbrowser-instance-id=instance-b'
	assert ProfileLease.process_matches_browser(process, record) is False


def test_browser_identity_falls_back_to_process_info_cmdline(tmp_path: Path):
	profile_dir = (tmp_path / 'profile').resolve()
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	process.cmdline.side_effect = psutil.AccessDenied(pid=123)
	process.info = {
		'cmdline': [
			'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
			f'--user-data-dir={profile_dir}',
			'--openbrowser-instance-id=instance-a',
		]
	}
	record = {
		'pid': 123,
		'start_time': 20.0,
		'profile_dir': str(profile_dir),
		'instance_id': 'instance-a',
		'ownership_marker': '--openbrowser-instance-id=instance-a',
	}

	assert ProfileLease.process_matches_browser(process, record) is True


def test_malformed_owner_identity_is_treated_as_not_alive():
	assert ProfileLease.process_is_alive(None, None) is False


def test_release_does_not_delete_malformed_metadata(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	lease = ProfileLease(profile_dir, instance_id='instance-a', owner_start_time=20.0)
	lease.acquire()
	lease.write_metadata()
	metadata_path = profile_dir / ProfileLease.METADATA_FILENAME
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['owner_start_time'] = 'not-a-number'
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	lease.release()

	assert metadata_path.exists()


def test_metadata_file_is_not_removed_for_another_owner(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	lease.write_metadata()
	metadata_path = tmp_path / 'profile' / ProfileLease.METADATA_FILENAME
	metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
	metadata['instance_id'] = 'instance-b'
	metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

	lease.release()

	assert metadata_path.exists()
