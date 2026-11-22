"""Explicit lifecycle for cached DOM views.

One cache entry belongs to one document view of one target. The entry is
stamped with the target and the generation that produced it. Any navigation
request, document commit, within-document navigation, focus change, reset, or
containment advances the generation and makes every older entry unreadable.

The machine stores metadata only. ``BrowserSession`` owns the selector map and
browser-state storage, so existing clearing semantics and attribute names stay
unchanged.

A read after an action is provisional. A provisional read must be confirmed
against a live document identity before the entry is returned. A confirmation
failure is treated as invalid, never as valid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any


class DomCachePhase(StrEnum):
    """Why the cache can or cannot answer a read."""

    EMPTY = "empty"
    READY = "ready"
    PROVISIONAL = "provisional"
    INVALIDATED = "invalidated"


@dataclass(frozen=True)
class DocumentIdentity:
    """The document a live target currently shows."""

    loader_id: str | None
    url: str


@dataclass(frozen=True)
class DomCacheEntry:
    """One published view and the generation that produced it."""

    target_id: str
    global_generation: int
    target_generation: int
    loader_id: str | None
    url: str | None
    selector_count: int
    has_summary: bool
    provisional: bool
    provisional_reason: str | None
    publish_sequence: int = 0


class DomCacheMachine:
    """Pure metadata state machine for cached DOM views.

    No browser call and no page content lives here. The session supplies the
    storage and the live identity read.
    """

    def __init__(self) -> None:
        self._global_generation = 0
        self._target_generations: dict[str, int] = {}
        self._known_identities: dict[str, DocumentIdentity] = {}
        self._main_frames: dict[str, str] = {}
        self._entry: DomCacheEntry | None = None
        self._invalidated_target: str | None = None
        self._invalidated = False
        self._cleared = True
        self._invalidation_reason: str | None = None
        self._invalidation_count = 0
        self._invalidated_at: float | None = None
        self._build_sequence = 0

    # ------------------------------------------------------------------
    # Generations
    # ------------------------------------------------------------------

    def epoch(self, target_id: str) -> tuple[int, int]:
        """Return the generation pair a publish must match."""
        return (self._global_generation, self._target_generations.get(target_id, 0))

    def has_entry(self, target_id: str | None = None) -> bool:
        entry = self._entry
        if entry is None:
            return False
        return target_id is None or entry.target_id == target_id

    def is_current(self, target_id: str) -> bool:
        entry = self._entry
        if entry is None or entry.target_id != target_id:
            return False
        return (entry.global_generation, entry.target_generation) == self.epoch(target_id)

    def summary_is_current(self, target_id: str) -> bool:
        return self.is_current(target_id) and bool(self._entry and self._entry.has_summary)

    def begin_build(self) -> int:
        """Return a monotonic token for one DOM build.

        A build that started before a newer publish must not overwrite it when
        it finishes late.
        """
        self._build_sequence += 1
        return self._build_sequence

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def note_identity(self, target_id: str, identity: DocumentIdentity) -> None:
        self._known_identities[target_id] = identity

    def known_identity(self, target_id: str) -> DocumentIdentity | None:
        return self._known_identities.get(target_id)

    def note_main_frame(self, target_id: str, frame_id: str) -> None:
        self._main_frames[target_id] = frame_id

    def main_frame(self, target_id: str) -> str | None:
        return self._main_frames.get(target_id)

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    def publish(
        self,
        target_id: str,
        *,
        epoch: tuple[int, int],
        loader_id: str | None,
        url: str | None,
        selector_count: int,
        has_summary: bool = False,
        sequence: int | None = None,
    ) -> bool:
        """Publish one view, or refuse it when its generation is stale."""
        if epoch != self.epoch(target_id):
            return False
        if (
            sequence is not None
            and self._entry is not None
            and self._entry.target_id == target_id
            and self._entry.publish_sequence > sequence
        ):
            return False
        self._entry = DomCacheEntry(
            target_id=target_id,
            global_generation=epoch[0],
            target_generation=epoch[1],
            loader_id=loader_id,
            url=url,
            selector_count=selector_count,
            has_summary=has_summary,
            provisional=False,
            provisional_reason=None,
            publish_sequence=sequence or 0,
        )
        self._invalidated = False
        self._cleared = False
        return True

    def adopt(self, target_id: str, *, selector_count: int, has_summary: bool) -> None:
        """Treat state written outside the machine as current for compatibility."""
        identity = self._known_identities.get(target_id)
        self.publish(
            target_id,
            epoch=self.epoch(target_id),
            loader_id=identity.loader_id if identity else None,
            url=identity.url if identity else None,
            selector_count=selector_count,
            has_summary=has_summary,
        )

    def mark_summary(self, target_id: str, *, epoch: tuple[int, int], sequence: int | None = None) -> bool:
        """Attach a summary to the current selector view.

        An older summary must not attach to a newer published map.
        """
        if epoch != self.epoch(target_id) or not self.is_current(target_id):
            return False
        assert self._entry is not None
        if sequence is not None and self._entry.publish_sequence > sequence:
            return False
        self._entry = replace(self._entry, has_summary=True)
        return True

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate(self, target_id: str, reason: str) -> bool:
        """Advance one target generation and drop its entry."""
        self._target_generations[target_id] = self._target_generations.get(target_id, 0) + 1
        discarded = self.has_entry(target_id)
        if discarded:
            self._entry = None
        self._invalidated = True
        self._cleared = False
        self._invalidated_target = target_id
        self._record(reason)
        return discarded

    def invalidate_all(self, reason: str) -> bool:
        """Advance every target generation and drop the single entry."""
        discarded = self._entry is not None
        self._global_generation += 1
        self._entry = None
        self._invalidated = True
        self._cleared = False
        self._invalidated_target = None
        self._record(reason)
        return discarded

    def clear(self, reason: str) -> None:
        """Drop all state for session teardown and report an empty cache."""
        self._global_generation += 1
        self._entry = None
        self._invalidated = False
        self._cleared = True
        self._invalidated_target = None
        self._record(reason)

    def forget(self, target_id: str) -> None:
        """Drop per-target metadata after the target is gone."""
        self._target_generations.pop(target_id, None)
        self._known_identities.pop(target_id, None)
        self._main_frames.pop(target_id, None)
        if self.has_entry(target_id):
            self._entry = None

    def _record(self, reason: str) -> None:
        self._invalidation_reason = reason
        self._invalidation_count += 1
        self._invalidated_at = time.time()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def mark_provisional(self, target_id: str, reason: str) -> None:
        if self._entry is None or self._entry.target_id != target_id:
            return
        if self._entry.provisional:
            return
        self._entry = replace(self._entry, provisional=True, provisional_reason=reason)

    def needs_revalidation(self, target_id: str) -> bool:
        return self.is_current(target_id) and bool(self._entry and self._entry.provisional)

    def confirm(self, target_id: str, identity: DocumentIdentity) -> bool:
        """Compare a live identity with the published one.

        An entry published without an identity adopts the first live identity.
        Any later mismatch is a failure.
        """
        entry = self._entry
        if entry is None or entry.target_id != target_id:
            return False
        self._known_identities[target_id] = identity
        if entry.loader_id is not None and (identity.loader_id is None or identity.loader_id != entry.loader_id):
            return False
        if entry.url is not None and (not identity.url or identity.url != entry.url):
            return False
        self._entry = replace(
            entry,
            loader_id=entry.loader_id or identity.loader_id,
            url=entry.url or identity.url,
            provisional=False,
            provisional_reason=None,
        )
        return True

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def phase(self, target_id: str | None = None) -> DomCachePhase:
        entry = self._entry
        if entry is not None:
            return DomCachePhase.PROVISIONAL if entry.provisional else DomCachePhase.READY
        if self._cleared:
            return DomCachePhase.EMPTY
        if self._invalidated and (self._invalidated_target is None or self._invalidated_target == target_id):
            return DomCachePhase.INVALIDATED
        return DomCachePhase.EMPTY

    def snapshot(self, target_id: str | None = None) -> dict[str, Any]:
        """Return a bounded status view with no page content."""
        entry = self._entry
        effective_target = target_id or (entry.target_id if entry else None)
        epoch = self.epoch(effective_target) if effective_target else (self._global_generation, 0)
        return {
            "phase": self.phase(effective_target).value,
            "global_generation": epoch[0],
            "target_generation": epoch[1],
            "target_id": effective_target,
            "selector_count": entry.selector_count if entry else 0,
            "summary_cached": bool(entry and entry.has_summary),
            "provisional": bool(entry and entry.provisional),
            "invalidation_reason": self._invalidation_reason,
            "invalidation_count": self._invalidation_count,
            "invalidated_at": self._invalidated_at,
        }
