"""Runner state preserves matched browser continuity across warm tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.interactive import engine_session
from benchmarks.interactive.adapters.base import RunContext


@pytest.fixture
def harness(monkeypatch, tmp_path):
    fields = dict(
        lane="matched-shell-1255",
        revision="1255",
        version="159.0.8154.6",
        launcher_sha256="a" * 64,
        engine_image_sha256="a" * 64,
    )
    pin = SimpleNamespace(public_proof=lambda: fields, **fields)
    marker = str(tmp_path / "profile")
    proof = SimpleNamespace(
        profile_marker=marker,
        keychain_verified=True,
        binding=SimpleNamespace(pid=123, created_at=1.0, executable_basename="browser", **fields),
        **fields,
    )
    valid = SimpleNamespace(
        status="valid",
        eligible=True,
        proof=proof,
        keychain_verified=True,
        public_proof=lambda: {"status": "valid", "eligible": True, "pid": 123, "created_at": 1.0},
    )
    failed = SimpleNamespace(
        status="process_replaced",
        eligible=False,
        proof=None,
        keychain_verified=None,
        public_proof=lambda: {"status": "process_replaced", "eligible": False, "pid": None, "created_at": None},
    )
    observations = []
    verifications = []
    monkeypatch.setattr(engine_session, "observe_session", lambda *a: observations.append(a) or valid)
    monkeypatch.setattr(engine_session, "verify_session", lambda *a: verifications.append(a) or valid)
    return SimpleNamespace(
        pin=pin,
        marker=marker,
        valid=valid,
        failed=failed,
        observations=observations,
        verifications=verifications,
        ctx=RunContext(
            task="wizard",
            arm="fake",
            sample=0,
            base_url="http://fixture.invalid",
            root=tmp_path,
            downloads=tmp_path / "downloads",
            profile=Path(marker),
            home=tmp_path / "home",
            tmp=tmp_path / "tmp",
            config=tmp_path / "config",
            session="fake-session",
        ),
        arm=SimpleNamespace(),
    )


def test_unconfigured_session_performs_no_proof_reads(harness):
    session = engine_session.EngineSession(None)
    assert session.observe(harness.arm, harness.ctx) is None
    assert session.capture_warm(harness.arm, harness.ctx) is None
    assert session.verify_warm(harness.arm, harness.ctx) is None
    row = {"ok": True, "browser_version": "legacy"}
    session.stamp(row, None)
    assert row == {"ok": True, "browser_version": "legacy"}
    assert harness.observations == harness.verifications == []


def test_cold_observation_uses_the_actual_adapter_profile(harness):
    session = engine_session.EngineSession(harness.pin)
    arm = SimpleNamespace(_keychain_profile=lambda ctx: harness.marker)
    assert session.observe(arm, harness.ctx) is harness.valid
    assert harness.observations == [(harness.pin, harness.marker)]


def test_mcp_observation_uses_its_context_profile(harness):
    session = engine_session.EngineSession(harness.pin)
    arm = SimpleNamespace(_mock_keychain_profile=lambda ctx: harness.marker)
    assert session.observe(arm, harness.ctx) is harness.valid
    assert harness.observations == [(harness.pin, harness.marker)]


def test_warm_tasks_verify_the_original_warmup_proof_without_recaching(harness):
    session = engine_session.EngineSession(harness.pin)
    assert session.capture_warm(harness.arm, harness.ctx) is harness.valid
    for _ in range(4):
        assert session.verify_warm(harness.arm, harness.ctx) is harness.valid
    assert len(harness.observations) == 1
    assert harness.verifications == [(harness.valid.proof, harness.pin)] * 4
    with pytest.raises(RuntimeError):
        session.capture_warm(harness.arm, harness.ctx)


def test_a_failed_warm_identity_poisons_later_rows_until_new_warmup(harness, monkeypatch):
    session = engine_session.EngineSession(harness.pin)
    session.capture_warm(harness.arm, harness.ctx)
    monkeypatch.setattr(engine_session, "verify_session", lambda *a: harness.failed)
    assert session.verify_warm(harness.arm, harness.ctx) is harness.failed

    def forbidden(*args):
        pytest.fail("failed warm session attempted to recache")

    monkeypatch.setattr(engine_session, "verify_session", forbidden)
    monkeypatch.setattr(engine_session, "observe_session", forbidden)
    assert session.verify_warm(harness.arm, harness.ctx) is harness.failed


def test_adapter_context_marker_change_is_ineligible_even_if_old_browser_lives(harness):
    session = engine_session.EngineSession(harness.pin)
    session.capture_warm(harness.arm, harness.ctx)
    changed = replace(harness.ctx, profile=Path(harness.marker + "-new"))
    observation = session.verify_warm(harness.arm, changed)
    assert observation is not None
    assert observation.eligible is False
    assert observation.status == "marker_changed"
    assert harness.verifications == []
    assert session.verify_warm(harness.arm, harness.ctx) is observation


def test_valid_engine_stamp_preserves_tool_version_and_requires_both_sides(harness):
    session = engine_session.EngineSession(harness.pin)
    session.capture_warm(harness.arm, harness.ctx)
    row = {"ok": True, "browser_version": "server-version", "error": ""}
    session.stamp(row, harness.valid, before=harness.valid)
    assert row["ok"] is True
    assert row["browser_version"] == harness.pin.version
    assert row["reported_browser_version"] == "server-version"
    assert row["engine"] == {
        "required": True,
        "verified": True,
        "pin": harness.pin.public_proof(),
        "warmup": harness.valid.public_proof(),
        "before": harness.valid.public_proof(),
        "after": harness.valid.public_proof(),
    }


@pytest.mark.parametrize("failure_at", ["before", "after"])
def test_failed_engine_stamp_never_repairs_task_or_unknown_metrics(harness, failure_at):
    session = engine_session.EngineSession(harness.pin)
    before = harness.failed if failure_at == "before" else harness.valid
    after = harness.failed if failure_at == "after" else harness.valid
    row = {"ok": True, "browser_version": "unproven", "error": ""}
    session.stamp(row, after, before=before)
    assert row["ok"] is False
    assert row["engine"]["verified"] is False
    assert row["browser_version"] is None
    assert "process_replaced" in row["error"]


def test_valid_engine_never_repairs_a_task_oracle_failure(harness):
    session = engine_session.EngineSession(harness.pin)
    row = {"ok": False, "browser_version": "old", "error": "oracle"}
    session.stamp(row, harness.valid)
    assert row["ok"] is False
    assert row["error"] == "oracle"


def test_missing_warmup_is_unknown_and_never_starts_discovery(harness):
    session = engine_session.EngineSession(harness.pin)
    observation = session.verify_warm(harness.arm, harness.ctx)
    assert observation is not None
    assert observation.eligible is False
    assert observation.keychain_verified is None
    assert harness.observations == harness.verifications == []


def test_warm_stamp_rejects_a_missing_before_proof(harness):
    session = engine_session.EngineSession(harness.pin)
    session.capture_warm(harness.arm, harness.ctx)
    row = {"ok": True, "error": "", "browser_version": "unproven"}
    session.stamp(row, harness.valid)
    assert row["ok"] is False
    assert row["engine"]["verified"] is False


@pytest.mark.parametrize("field", ["version", "launcher_sha256", "engine_image_sha256"])
def test_cold_stamp_requires_observation_to_match_its_pin(harness, field):
    session = engine_session.EngineSession(harness.pin)
    alien = deepcopy(harness.valid)
    setattr(alien.proof, field, "different")
    row = {"ok": True, "error": "", "browser_version": "unproven"}
    session.stamp(row, alien)
    assert row["ok"] is False
    assert row["browser_version"] is None


@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("field", ["pid", "created_at"])
def test_warm_stamp_requires_the_original_warmup_identity(harness, side, field):
    session = engine_session.EngineSession(harness.pin)
    session.capture_warm(harness.arm, harness.ctx)
    alien = deepcopy(harness.valid)
    setattr(alien.proof.binding, field, 456)
    row = {"ok": True, "error": "", "browser_version": "unproven"}
    session.stamp(row, alien if side == "after" else harness.valid, before=alien if side == "before" else harness.valid)
    assert row["ok"] is False
    assert row["engine"]["verified"] is False
