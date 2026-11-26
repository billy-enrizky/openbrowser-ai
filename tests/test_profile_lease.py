"""Tests for ownership-safe browser profile leases."""

import json
import os
from decimal import Decimal
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


def test_record_browser_normalizes_process_identity(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	process = MagicMock()
	process.pid = '123'
	process.create_time.return_value = Decimal('456.5')

	try:
		record = lease.record_browser(
			process,
			ownership_marker='--openbrowser-instance-id=instance-a',
			executable='/custom/chrome',
			cdp_port=9223,
		)

		assert record['pid'] == 123
		assert record['start_time'] == 456.5
		metadata = ProfileLease.read_metadata(tmp_path / 'profile')
		assert metadata is not None
		assert metadata['browser'] == record
	finally:
		lease.release()


def test_process_identity_rejects_pid_reuse():
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0

	assert ProfileLease.process_matches_identity(process, 123, 20.0) is True
	assert ProfileLease.process_matches_identity(process, 123, 19.0) is False
	assert ProfileLease.process_matches_identity(process, 124, 20.0) is False


def test_process_identity_status_distinguishes_inaccessible_process(monkeypatch):
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	monkeypatch.setattr('psutil.Process', lambda pid: process)

	assert ProfileLease.process_identity_status(123, 20.0) == 'match'
	assert ProfileLease.process_identity_status(123, 19.0) == 'mismatch'

	monkeypatch.setattr('psutil.Process', MagicMock(side_effect=psutil.AccessDenied(pid=123)))
	assert ProfileLease.process_identity_status(123, 20.0) == 'unknown'


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


def test_pristine_profile_proof_for_missing_and_empty_directories(tmp_path: Path):
	missing = ProfileLease(tmp_path / 'missing', instance_id='instance-a')
	missing.acquire()
	try:
		assert missing.profile_pristine is True
	finally:
		missing.release()

	empty_dir = tmp_path / 'empty'
	empty_dir.mkdir()
	empty = ProfileLease(empty_dir, instance_id='instance-b')
	empty.acquire()
	try:
		assert empty.profile_pristine is True
	finally:
		empty.release()


@pytest.mark.parametrize('artifact', ['Preferences', 'storage_state.json'])
def test_pristine_profile_proof_rejects_nonempty_existing_directory(tmp_path: Path, artifact: str):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	(profile_dir / artifact).write_text('{}', encoding='utf-8')

	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


@pytest.mark.parametrize('artifact', [ProfileLease.LOCK_FILENAME, ProfileLease.METADATA_FILENAME])
def test_pristine_profile_proof_rejects_preexisting_owner_artifacts(tmp_path: Path, artifact: str):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	(profile_dir / artifact).write_text('{}', encoding='utf-8')

	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_rejects_pre_lock_insertion(tmp_path: Path, monkeypatch):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	original_lock = ProfileLease._lock

	def insert_before_lock(handle):
		(profile_dir / 'Preferences').write_text('{}', encoding='utf-8')
		original_lock(handle)

	monkeypatch.setattr(ProfileLease, '_lock', staticmethod(insert_before_lock))
	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_rejects_directory_path_replacement(tmp_path: Path, monkeypatch):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	moved_dir = tmp_path / 'profile-moved'
	original_lock = ProfileLease._lock

	def replace_before_lock(handle):
		profile_dir.rename(moved_dir)
		profile_dir.mkdir()
		original_lock(handle)

	monkeypatch.setattr(ProfileLease, '_lock', staticmethod(replace_before_lock))
	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_rejects_lock_path_replacement(tmp_path: Path, monkeypatch):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	lock_path = profile_dir / ProfileLease.LOCK_FILENAME
	original_lock = ProfileLease._lock

	def replace_lock_before_lock(handle):
		lock_path.unlink()
		lock_path.write_bytes(b'')
		original_lock(handle)

	monkeypatch.setattr(ProfileLease, '_lock', staticmethod(replace_lock_before_lock))
	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_rejects_read_uncertainty(tmp_path: Path, monkeypatch):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	real_scandir = os.scandir
	scans = 0

	def fail_second_scan(path):
		nonlocal scans
		scans += 1
		if scans == 2:
			raise OSError('scan denied')
		return real_scandir(path)

	monkeypatch.setattr(os, 'scandir', fail_second_scan)
	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_invalidates_on_same_object_reacquire(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is True
		lease.acquire()
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_stays_false_for_new_owner_reacquire(tmp_path: Path):
	profile_dir = tmp_path / 'profile'
	first = ProfileLease(profile_dir, instance_id='instance-a')
	first.acquire()
	try:
		second = ProfileLease(profile_dir, instance_id='instance-b')
		with pytest.raises(ProfileInUseError, match='Browser profile already in use'):
			second.acquire()
		assert second.profile_pristine is False
	finally:
		first.release()


def test_pristine_profile_proof_invalidates_on_release(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	assert lease.profile_pristine is True
	lease.release()
	assert lease.profile_pristine is False


def test_pristine_profile_proof_invalidates_on_acquire_failure(tmp_path: Path, monkeypatch):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')

	def fail_lock(_handle):
		raise OSError('lock failed')

	monkeypatch.setattr(ProfileLease, '_lock', staticmethod(fail_lock))
	with pytest.raises(OSError, match='lock failed'):
		lease.acquire()
	assert lease.profile_pristine is False


def test_pristine_profile_proof_survives_owner_metadata_write(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	try:
		lease.write_metadata()
		assert lease.profile_pristine is True
	finally:
		lease.release()


def test_pristine_profile_proof_clears_when_browser_is_recorded(tmp_path: Path):
	lease = ProfileLease(tmp_path / 'profile', instance_id='instance-a')
	lease.acquire()
	process = MagicMock()
	process.pid = 123
	process.create_time.return_value = 20.0
	try:
		lease.write_metadata()
		assert lease.profile_pristine is True
		lease.record_browser(
			process,
			ownership_marker='--openbrowser-instance-id=instance-a',
			executable='/custom/chrome',
			cdp_port=9223,
		)
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_rejects_directory_replacement_during_listing(tmp_path: Path, monkeypatch):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	moved_dir = tmp_path / 'profile-moved'
	real_scandir = os.scandir
	scans = 0

	def replace_second_scan(path):
		nonlocal scans
		scans += 1
		if scans == 2:
			profile_dir.rename(moved_dir)
			profile_dir.mkdir()
			(profile_dir / ProfileLease.LOCK_FILENAME).write_bytes(b'')
		return real_scandir(path)

	monkeypatch.setattr(os, 'scandir', replace_second_scan)
	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_rejects_lock_replacement_during_listing(tmp_path: Path, monkeypatch):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	lock_path = profile_dir / ProfileLease.LOCK_FILENAME
	real_scandir = os.scandir
	scans = 0

	def replace_second_scan(path):
		nonlocal scans
		scans += 1
		if scans == 2:
			lock_path.unlink()
			lock_path.write_bytes(b'')
		return real_scandir(path)

	monkeypatch.setattr(os, 'scandir', replace_second_scan)
	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


def test_pristine_profile_proof_rejects_closing_identity_read_failure(tmp_path: Path, monkeypatch):
	profile_dir = tmp_path / 'profile'
	profile_dir.mkdir()
	real_fstat = os.fstat
	calls = 0

	def fail_second_fstat(handle):
		nonlocal calls
		calls += 1
		if calls == 2:
			raise OSError('closing fstat denied')
		return real_fstat(handle)

	monkeypatch.setattr(os, 'fstat', fail_second_fstat)
	lease = ProfileLease(profile_dir, instance_id='instance-a')
	lease.acquire()
	try:
		assert lease.profile_pristine is False
	finally:
		lease.release()


@pytest.mark.parametrize(
	('value', 'expected'),
	[(None, False), ('invalid', False), ('1.5', True), (1.5, True), (2.5, False)],
)
def test_owner_metadata_start_time_handles_missing_and_numeric_values(tmp_path: Path, value, expected):
	lease = ProfileLease(tmp_path / 'profile', instance_id='owner', owner_pid=12345, owner_start_time=1.5)
	metadata = {'instance_id': 'owner', 'owner_pid': 12345, 'owner_start_time': value}
	assert lease._metadata_belongs_to_this_lease(metadata) is expected
