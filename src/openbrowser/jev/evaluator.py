"""Generic direct TypeSafe Jev evaluation with strict choice validation."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from openbrowser.jev.views import JevAnswer


class JevEvaluationError(RuntimeError):
	"""Raised when direct Jev evaluation or its typed response is unsafe."""


class JevEvaluator(Protocol):
	async def evaluate(
		self,
		*,
		state: dict[str, Any],
		questions: dict[str, dict[str, Any]],
	) -> "JevEvaluation":
		"""Return validated answers for code-owned choice questions."""


@dataclass(frozen=True, slots=True)
class JevEvaluation:
	answers: dict[str, JevAnswer]
	usage: object | None = None


def _resolve_api_key(explicit: str | None) -> str | None:
	for value in (explicit, os.environ.get("JEV_API_KEY"), os.environ.get("TYPESAFE_API_KEY")):
		if value and value.strip():
			return value.strip()
	return None


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
	if not isinstance(value, Mapping):
		raise JevEvaluationError(f"Jev response {description} must be an object")
	return value


def _answer_field(answer: Any, field: str) -> Any:
	if isinstance(answer, Mapping):
		return answer.get(field)
	return getattr(answer, field, None)


def _json_safe(value: Any) -> Any:
	if isinstance(value, Mapping):
		return {str(key): _json_safe(item) for key, item in value.items()}
	if isinstance(value, tuple):
		return [_json_safe(item) for item in value]
	if isinstance(value, list):
		return [_json_safe(item) for item in value]
	return value


def _validate_answer(question_id: str, answer: Any, question: Mapping[str, Any]) -> JevAnswer:
	choice = _answer_field(answer, "choice")
	if not isinstance(choice, str):
		raise JevEvaluationError(f"Jev answer {question_id} must select a string choice")
	criteria = _require_mapping(question.get("criteria"), f"criteria for {question_id}")
	if choice not in criteria:
		raise JevEvaluationError(f"Jev answer {question_id} selected an unknown choice: {choice!r}")

	probability_data = _require_mapping(
		_answer_field(answer, "probabilities"), f"probabilities for {question_id}"
	)
	if set(probability_data) != set(criteria):
		raise JevEvaluationError(
			f"Jev answer {question_id} probabilities must match the question choices exactly"
		)

	probabilities: dict[str, float] = {}
	for probability_choice, raw_probability in probability_data.items():
		if isinstance(raw_probability, bool):
			raise JevEvaluationError(f"Jev answer {question_id} has a non-numeric probability")
		try:
			probability = float(raw_probability)
		except (TypeError, ValueError) as exc:
			raise JevEvaluationError(f"Jev answer {question_id} has a non-numeric probability") from exc
		if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
			raise JevEvaluationError(f"Jev answer {question_id} has a probability outside [0, 1]")
		probabilities[str(probability_choice)] = probability

	if abs(sum(probabilities.values()) - 1.0) > 0.02:
		raise JevEvaluationError(f"Jev answer {question_id} probabilities must sum to 1.0")
	if probabilities[choice] < max(probabilities.values()) - 1e-6:
		raise JevEvaluationError(f"Jev answer {question_id} must select its highest-probability choice")

	confidence = _answer_field(answer, "confidence")
	if confidence is not None:
		try:
			confidence_value = float(confidence)
		except (TypeError, ValueError) as exc:
			raise JevEvaluationError(f"Jev answer {question_id} has invalid confidence") from exc
		if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
			raise JevEvaluationError(f"Jev answer {question_id} has invalid confidence")

	return JevAnswer(choice=choice, probabilities=probabilities)


def validate_answers(raw: Any, questions: dict[str, dict[str, Any]]) -> dict[str, JevAnswer]:
	"""Validate one complete TypeSafe response against supplied choice questions."""
	choices_raw = raw.choices if hasattr(raw, "choices") else raw.get("choices") if isinstance(raw, Mapping) else None
	answers = _require_mapping(choices_raw, "choices")
	if set(answers) != set(questions):
		raise JevEvaluationError("Jev answer IDs must exactly match the question IDs")
	return {
		question_id: _validate_answer(question_id, answers[question_id], question)
		for question_id, question in questions.items()
	}


class RemoteJevEvaluator:
	"""Call TypeSafe Jev once for generic code-owned choice questions."""

	def __init__(
		self,
		*,
		api_key: str | None = None,
		model_name: str = "jev-latest",
		client: object | None = None,
	) -> None:
		self.model_name = model_name
		self._api_key = _resolve_api_key(api_key)
		self._client = client if client is not None else self._build_default_client()

	@property
	def api_key(self) -> str | None:
		"""Return the resolved key for diagnostics without logging it."""
		return self._api_key

	def _build_default_client(self) -> object:
		if not self._api_key:
			raise JevEvaluationError(
				"Jev API key is missing. Set JEV_API_KEY or TYPESAFE_API_KEY, or pass api_key explicitly."
			)
		try:
			from typesafe_sdk import AsyncTypeSafeClient
		except (ImportError, ModuleNotFoundError) as exc:
			raise JevEvaluationError(
				"TypeSafe SDK is not installed. Install the optional dependency with: uv sync --extra jev"
			) from exc
		try:
			return AsyncTypeSafeClient(api_key=self._api_key)
		except Exception as exc:
			raise JevEvaluationError(f"Failed to create the TypeSafe Jev client: {type(exc).__name__}") from exc

	def _client_or_create(self) -> object:
		if self._client is None:
			self._client = self._build_default_client()
		return self._client

	@staticmethod
	def _typed_questions(questions: dict[str, dict[str, Any]]) -> dict[str, object]:
		try:
			from typesafe_sdk import Choice
		except (ImportError, ModuleNotFoundError) as exc:
			raise JevEvaluationError(
				"TypeSafe SDK is not installed. Install the optional dependency with: uv sync --extra jev"
			) from exc

		typed_questions: dict[str, object] = {}
		for question_id, question in questions.items():
			if question.get("type") != "choice":
				raise JevEvaluationError(f"Jev question {question_id} must be a Choice question")
			criteria = _require_mapping(question.get("criteria"), f"criteria for {question_id}")
			try:
				typed_questions[question_id] = Choice(
					instructions=_json_safe(question.get("instructions")),
					criteria=_json_safe(criteria),
				)
			except Exception as exc:
				raise JevEvaluationError(f"Could not construct Jev question {question_id}") from exc
		return typed_questions

	async def evaluate(
		self,
		*,
		state: dict[str, Any],
		questions: dict[str, dict[str, Any]],
	) -> JevEvaluation:
		"""Issue exactly one System One request and validate every returned choice."""
		client = self._client_or_create()
		typed_questions = self._typed_questions(questions)
		try:
			system_one = getattr(client, "system_one")
			response = await system_one(state=state, questions=typed_questions, model=self.model_name)
		except JevEvaluationError:
			raise
		except Exception as exc:
			raise JevEvaluationError(f"TypeSafe Jev request failed: {type(exc).__name__}") from exc

		try:
			answers = validate_answers(response, questions)
		except JevEvaluationError:
			raise
		except Exception as exc:
			raise JevEvaluationError(f"Malformed TypeSafe Jev response: {type(exc).__name__}") from exc

		return JevEvaluation(answers=answers, usage=getattr(response, "usage", None))
