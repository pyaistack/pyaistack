"""Ollama embedding and chat providers."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from ..exceptions import DependencyError, ProviderError
from ..rag.types import Message

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def _load_ollama() -> tuple[
    Any, type[Exception], type[Exception], type[Exception], type[Exception]
]:
    try:
        from httpx import TimeoutException
        from ollama import Client, RequestError, ResponseError
    except ImportError as exc:  # pragma: no cover - exercised only in broken installations
        raise DependencyError(
            "The 'ollama' package is required. Install the project with 'pip install -e .'."
        ) from exc
    return Client, RequestError, ResponseError, ConnectionError, TimeoutException


class _OllamaBase:
    def __init__(
        self,
        *,
        model: str,
        host: str,
        timeout_seconds: float,
        retries: int,
        retry_base_delay_seconds: float,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model cannot be empty")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be greater than zero")
        if type(retries) is not int or retries < 0:
            raise ValueError("retries cannot be negative")
        if (
            type(retry_base_delay_seconds) not in (int, float)
            or not math.isfinite(retry_base_delay_seconds)
            or retry_base_delay_seconds < 0
        ):
            raise ValueError("retry_base_delay_seconds cannot be negative")

        Client, RequestError, ResponseError, ConnectionErrorType, TimeoutException = _load_ollama()
        self._request_error = RequestError
        self._response_error = ResponseError
        self._connection_error = ConnectionErrorType
        self._timeout_error = TimeoutException
        self._client = Client(host=host, timeout=timeout_seconds)
        self._model_name = model
        self._retries = retries
        self._retry_base_delay_seconds = retry_base_delay_seconds

    @property
    def model_name(self) -> str:
        return self._model_name

    def _call(self, operation: str, fn: Callable[[], T]) -> T:
        last_error: Exception | None = None
        attempts = self._retries + 1
        attempts_made = 0

        for attempt in range(attempts):
            attempts_made = attempt + 1
            try:
                return fn()
            except self._request_error as exc:
                raise ProviderError(f"Ollama rejected {operation}") from exc
            except self._response_error as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", -1)
                error_text = str(exc).lower()
                if status_code == 404 and "model" in error_text and "not found" in error_text:
                    raise ProviderError(
                        f'Model "{self.model_name}" is not available.\n'
                        f"Run: ollama pull {self.model_name}"
                    ) from exc
                retryable = status_code in {408, 425, 429} or status_code >= 500
                if not retryable or attempt == attempts - 1:
                    break
            except self._connection_error as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
            except (TimeoutError, self._timeout_error) as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break

            delay = self._retry_base_delay_seconds * (2**attempt)
            LOGGER.warning(
                "Ollama %s failed; retrying in %.2fs (attempt %s/%s)",
                operation,
                delay,
                attempt + 1,
                attempts,
            )
            if delay:
                time.sleep(delay)

        raise ProviderError(
            f"Ollama {operation} failed after {attempts_made} attempt(s): {last_error}"
        ) from last_error


class OllamaEmbeddingProvider(_OllamaBase):
    """Embedding provider backed by Ollama's /api/embed API."""

    def __init__(
        self,
        model: str = "embeddinggemma",
        *,
        host: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
        retries: int = 2,
        retry_base_delay_seconds: float = 0.25,
    ) -> None:
        super().__init__(
            model=model,
            host=host,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = [text for text in texts]
        if not values:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in values):
            raise ProviderError("cannot embed an empty text value")

        response = self._call(
            "embedding request",
            lambda: self._client.embed(model=self.model_name, input=values),
        )
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None:
            try:
                embeddings = response["embeddings"]
            except (KeyError, TypeError) as exc:
                raise ProviderError("Ollama embedding response did not contain embeddings") from exc

        try:
            result = [[float(value) for value in vector] for vector in embeddings]
            if any(not vector or any(not math.isfinite(v) for v in vector) for vector in result):
                raise ValueError("non-finite or empty vector")
        except (ValueError, TypeError, OverflowError) as error:
            raise ProviderError("Ollama returned malformed embeddings") from error
        if len(result) != len(values):
            raise ProviderError(
                f"Ollama returned {len(result)} embeddings for {len(values)} input texts"
            )
        return result


class OllamaChatProvider(_OllamaBase):
    """Chat provider backed by Ollama's /api/chat API."""

    def __init__(
        self,
        model: str = "gemma3:4b",
        *,
        host: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        retries: int = 2,
        retry_base_delay_seconds: float = 0.25,
        temperature: float = 0.1,
    ) -> None:
        if type(temperature) not in (int, float) or not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        self._temperature = temperature
        super().__init__(
            model=model,
            host=host,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
        )

    def chat(self, messages: Sequence[Message]) -> str:
        if not messages:
            raise ProviderError("chat requires at least one message")

        response = self._call(
            "chat request",
            lambda: self._client.chat(
                model=self.model_name,
                messages=list(messages),
                options={"temperature": self._temperature},
            ),
        )

        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None:
            try:
                content = response["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise ProviderError("Ollama chat response did not contain message content") from exc

        if not isinstance(content, str):
            raise ProviderError("Ollama message content must be a string")
        text = content.strip()
        if not text:
            raise ProviderError("Ollama returned an empty response")
        return text
