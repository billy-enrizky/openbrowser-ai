"""Typed contracts for the candidate-driven Jev browser agent."""

from dataclasses import dataclass
from typing import Any, Literal


Operation = Literal[
	"CLICK",
	"TYPE_TEXT",
	"SELECT",
	"SCROLL_DOWN",
	"SCROLL_UP",
	"WAIT",
	"DONE",
	"BLOCKED",
]


@dataclass(frozen=True, slots=True)
class TextCandidate:
	key: str
	value: str
	hints: tuple[str, ...]
	source: Literal["task", "page"]


@dataclass(frozen=True, slots=True)
class EditableTarget:
	key: str
	index: int
	label: str
	role: str
	value: str
	element_hash: int
	candidates: tuple[TextCandidate, ...]


@dataclass(frozen=True, slots=True)
class ElementTarget:
	key: str
	index: int
	label: str
	role: str
	element_hash: int


@dataclass(frozen=True, slots=True)
class SelectTarget:
	key: str
	index: int
	text: str
	value: str


@dataclass(frozen=True, slots=True)
class JevObservation:
	state: dict[str, Any]
	questions: dict[str, dict[str, Any]]
	fingerprint: str
	click_targets: dict[str, ElementTarget]
	type_targets: dict[str, EditableTarget]
	select_targets: dict[str, SelectTarget]


@dataclass(frozen=True, slots=True)
class JevAnswer:
	choice: str
	probabilities: dict[str, float]


@dataclass(frozen=True, slots=True)
class JevDecision:
	operation: Operation
	target_key: str | None
	candidate_key: str | None
	answers: dict[str, JevAnswer]


@dataclass(frozen=True, slots=True)
class JevActionRecord:
	operation: str
	target_key: str | None
	candidate_key: str | None
	error: str | None
	fingerprint: str


@dataclass(frozen=True, slots=True)
class JevRunResult:
	status: Literal["done", "blocked", "max_steps", "stopped"]
	history: tuple[JevActionRecord, ...]


@dataclass(frozen=True, slots=True)
class SearchSentence:
	index: int
	text: str


@dataclass(frozen=True, slots=True)
class SearchPassage:
	id: str
	text: str
	sentences: tuple[SearchSentence, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticMatch:
	passage_id: str
	probability: float
	sentence_index: int
	sentence_text: str


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
	scores: tuple[SemanticMatch, ...]
	matches: tuple[SemanticMatch, ...]
	threshold: float = 0.58
	elapsed_ms: int = 0
	usage: object | None = None
