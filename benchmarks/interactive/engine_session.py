"""Attach shared engine proof to native rows and latch failed warm sessions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .adapters.base import RunContext
from .adapters.launch_proof import canonical_profile_marker
from .engine_binding import EngineBindingError, EnginePin
from .measurement import browser_launch_observer
from .session_guard import SessionObservation, observe_session, verify_session


def _profile_marker(arm: Any, ctx: RunContext) -> str:
    for name in ("_keychain_profile", "_mock_keychain_profile"):
        resolver = getattr(arm, name, None)
        if callable(resolver):
            marker = resolver(ctx)
            if not isinstance(marker, str):
                raise ValueError("adapter profile marker is malformed")
            return marker
    return str(ctx.profile)


class EngineSession:
    """Share one proof policy across every native CLI and MCP arm.

    Capture happens after explicit warmup. Verification keeps the initial
    proof, and a failed check poisons this session for its remaining tasks.
    A new instance after explicit warmup is required to renew the proof.
    """

    def __init__(self, pin: EnginePin | None):
        self.pin = pin
        self.initial: SessionObservation | None = None
        self.failure: SessionObservation | None = None
        self._captured = False
        self.launches: list[SessionObservation] = []

    @contextmanager
    def task_scope(self) -> Iterator[None]:
        if self.pin is None:
            yield
        else:
            with browser_launch_observer(self._capture_checkpoint):
                yield

    def _capture_checkpoint(self, marker: str) -> None:
        assert self.pin is not None
        try:
            observation = observe_session(self.pin, marker)
        except Exception:
            observation = SessionObservation("unknown")
        self.launches.append(observation)
        if not self._matches_pin(observation):
            raise EngineBindingError("initial_browser_unverified")

    def observe(self, arm: Any, ctx: RunContext) -> SessionObservation | None:
        if self.pin is None:
            return None
        try:
            return observe_session(self.pin, _profile_marker(arm, ctx))
        except Exception:
            return SessionObservation("unknown")

    def capture_warm(self, arm: Any, ctx: RunContext) -> SessionObservation | None:
        if self.pin is None:
            return None
        if self._captured:
            raise RuntimeError("warmup proof was already captured")
        self._captured = True
        self.initial = self.observe(arm, ctx)
        if self.initial is None or not self.initial.eligible:
            self.failure = self.initial or SessionObservation("unknown")
        return self.initial

    def verify_warm(self, arm: Any, ctx: RunContext) -> SessionObservation | None:
        if self.pin is None:
            return None
        if self.failure is not None:
            return self.failure
        if self.initial is None or self.initial.proof is None:
            self.failure = SessionObservation("unknown")
            return self.failure
        try:
            marker = canonical_profile_marker(_profile_marker(arm, ctx))
            observation = (
                verify_session(self.initial.proof, self.pin)
                if marker == self.initial.proof.profile_marker
                else SessionObservation("marker_changed")
            )
        except Exception:
            observation = SessionObservation("unknown")
        if not observation.eligible:
            self.failure = observation
        return observation

    def _matches_pin(self, observation: SessionObservation | None) -> bool:
        if self.pin is None or observation is None or not observation.eligible or observation.proof is None:
            return False
        proof = observation.proof
        fields = ("lane", "revision", "version", "launcher_sha256", "engine_image_sha256")
        return proof.keychain_verified is True and all(
            getattr(proof, name) == getattr(self.pin, name) == getattr(proof.binding, name) for name in fields
        )

    @staticmethod
    def _identity(observation: SessionObservation | None) -> tuple | None:
        if observation is None or observation.proof is None:
            return None
        proof = observation.proof
        return (proof.profile_marker, proof.binding.pid, proof.binding.created_at, proof.binding.executable_basename)

    @staticmethod
    def _browser_identity(observation: SessionObservation) -> tuple | None:
        if observation.proof is None:
            return None
        return (observation.proof.binding.pid, observation.proof.binding.created_at)

    def stamp(
        self,
        record: dict,
        after: SessionObservation | None,
        *,
        before: SessionObservation | None = None,
    ) -> None:
        if self.pin is None:
            return
        after = after or SessionObservation("unknown")
        valid = self._matches_pin(after) and (before is None or self._matches_pin(before))
        valid = valid and all(self._matches_pin(observation) for observation in self.launches)
        auth_failure = None
        if record.get("task") == "auth" and not self._captured:
            if len(self.launches) != 1:
                auth_failure = "auth_launch_proof_missing"
            elif self._browser_identity(self.launches[0]) == self._browser_identity(after):
                auth_failure = "auth_browser_not_replaced"
            valid = valid and auth_failure is None
        if self._captured:
            valid = (
                valid
                and before is not None
                and self._matches_pin(self.initial)
                and self._identity(before) == self._identity(after) == self._identity(self.initial)
                and self.failure is None
            )
        record["engine"] = {
            "required": True,
            "verified": valid,
            "pin": self.pin.public_proof(),
            "warmup": self.initial.public_proof() if self.initial is not None else None,
            "before": before.public_proof() if before is not None else None,
            "after": after.public_proof(),
            "launches": [observation.public_proof() for observation in self.launches],
        }
        record["reported_browser_version"] = record.get("browser_version")
        record["browser_version"] = self.pin.version if valid else None
        record["ok"] = record.get("ok") is True and valid
        failure = next(
            (item for item in (before, *self.launches, after) if item is not None and not self._matches_pin(item)),
            after,
        )
        code = auth_failure or (failure.status if not failure.eligible else "proof_mismatch")
        record["keychain_verified"] = (
            after.keychain_verified if valid else (failure.keychain_verified if not failure.eligible else None)
        )
        if not valid:
            record["error"] = "; ".join(value for value in (record.get("error", ""), f"matched_engine:{code}") if value)
            if self._captured and self.failure is None:
                self.failure = SessionObservation("unknown")
