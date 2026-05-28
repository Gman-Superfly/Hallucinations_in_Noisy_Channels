"""Generation backend interfaces for HNC experiments."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hnc_lab.schemas import GenerationRequest, GenerationResult, QAItem

__all__ = [
    "BackendFactory",
    "FixtureBackend",
    "GenerationBackend",
    "OpenAICompatibleBackend",
    "build_backend",
]


class GenerationBackend(Protocol):
    """Protocol for text generation backends."""

    name: str

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text for one request."""
        ...


class FixtureBackend:
    """Deterministic backend for pipeline smoke tests.

    This backend uses the known seed answers to simulate a temperature curve.
    It is not evidence for any HNC claim. It exists so the runner, metrics, and
    CSV outputs can be tested without calling an external model.
    """

    name = "fixture"

    def __init__(self, items: list[QAItem]) -> None:
        """Build a fixture backend from known-answer items.

        Args:
            items: QA items used for dry-run simulation.

        Raises:
            AssertionError: If items are empty.
        """
        assert len(items) > 0, "items required"
        self._answers = {item.item_id: item.answers[0] for item in items}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate deterministic fixture output.

        Args:
            request: Generation request.

        Returns:
            Simulated generation result.
        """
        answer = self._answers[request.item_id]
        text = self._select_output(answer, request)
        return GenerationResult(
            request=request,
            text=text,
            backend_name=self.name,
            model_name=self.name,
        )

    def _select_output(self, answer: str, request: GenerationRequest) -> str:
        """Select a simulated output for one request."""
        bucket = _stable_bucket(request.item_id, request.temperature, request.sample_index)
        if request.temperature == 0:
            return answer if bucket >= 20 else "I do not know."
        if request.temperature <= 0.7:
            return answer if bucket >= 10 else "I am not sure."
        if request.temperature <= 1.0:
            return answer if bucket >= 30 else f"{answer} maybe"
        return answer if bucket >= 55 else "Unsupported generic answer."


BackendFactory = GenerationBackend


class OpenAICompatibleBackend:
    """HTTP backend for OpenAI-compatible chat-completion APIs.

    Required environment variables:
        HNC_OPENAI_API_KEY: API key for the backend.
        HNC_OPENAI_MODEL: Model identifier.

    Optional environment variables:
        HNC_OPENAI_BASE_URL: Chat completions URL.
        HNC_OPENAI_TIMEOUT_SECONDS: Request timeout in seconds.
    """

    name = "openai_compatible"

    def __init__(self) -> None:
        """Build an OpenAI-compatible backend from environment variables."""
        self._api_key = os.environ.get("HNC_OPENAI_API_KEY", "").strip()
        self._model = os.environ.get("HNC_OPENAI_MODEL", "").strip()
        self._base_url = os.environ.get(
            "HNC_OPENAI_BASE_URL",
            "https://api.openai.com/v1/chat/completions",
        ).strip()
        timeout_raw = os.environ.get("HNC_OPENAI_TIMEOUT_SECONDS", "60").strip()
        self._timeout_seconds = float(timeout_raw)

        assert self._api_key, "HNC_OPENAI_API_KEY required"
        assert self._model, "HNC_OPENAI_MODEL required"
        assert self._base_url, "HNC_OPENAI_BASE_URL required"
        assert self._timeout_seconds > 0, "HNC_OPENAI_TIMEOUT_SECONDS must be positive"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate text for one request through a chat-completion API.

        Args:
            request: Generation request.

        Returns:
            Raw generation result.

        Raises:
            RuntimeError: If the HTTP call fails or returns an invalid response.
        """
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = Request(
            self._base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self._timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Backend HTTP error {error.code}: {error_body}") from error
        except URLError as error:
            raise RuntimeError(f"Backend connection error: {error}") from error

        data = json.loads(response_body)
        text = _extract_chat_completion_text(data)
        return GenerationResult(
            request=request,
            text=text,
            backend_name=self.name,
            model_name=self._model,
            raw_metadata={"response": data},
        )


def build_backend(name: str, items: list[QAItem]) -> GenerationBackend:
    """Build a generation backend by name.

    Args:
        name: Backend name.
        items: QA items, required by the fixture backend.

    Returns:
        Generation backend.

    Raises:
        ValueError: If the backend is unknown.
    """
    normalized_name = name.strip().lower()
    if normalized_name == "fixture":
        return FixtureBackend(items)
    if normalized_name == "openai_compatible":
        return OpenAICompatibleBackend()
    raise ValueError(f"Unknown backend: {name}")


def _stable_bucket(item_id: str, temperature: float, sample_index: int) -> int:
    """Return a stable pseudo-random integer in [0, 99]."""
    key = f"{item_id}|{temperature:.4f}|{sample_index}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % 100


def _extract_chat_completion_text(data: dict[str, object]) -> str:
    """Extract assistant text from an OpenAI-compatible response."""
    choices = data.get("choices")
    assert isinstance(choices, list), "response missing choices list"
    assert len(choices) > 0, "response choices empty"
    first_choice = choices[0]
    assert isinstance(first_choice, dict), "response choice must be an object"
    message = first_choice.get("message")
    assert isinstance(message, dict), "response choice missing message"
    content = message.get("content")
    assert isinstance(content, str), "response message content must be text"
    return content
