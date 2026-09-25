from __future__ import annotations

import pytest

from pyaistack.exceptions import ProviderError
from pyaistack.providers.ollama import _OllamaBase


class FakeResponseError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def build_provider() -> _OllamaBase:
    provider = object.__new__(_OllamaBase)
    provider._request_error = RuntimeError
    provider._response_error = FakeResponseError
    provider._connection_error = ConnectionError
    provider._timeout_error = TimeoutError
    provider._model_name = "embeddinggemma"
    provider._retries = 2
    provider._retry_base_delay_seconds = 0
    return provider


def test_missing_ollama_model_has_an_actionable_error_with_debug_cause() -> None:
    provider = build_provider()
    missing_model = FakeResponseError('model "embeddinggemma" not found', 404)

    with pytest.raises(ProviderError, match="Run: ollama pull embeddinggemma") as error:
        provider._call("embedding request", lambda: (_ for _ in ()).throw(missing_model))

    assert error.value.__cause__ is missing_model


def test_non_retryable_ollama_error_reports_actual_attempt_count() -> None:
    provider = build_provider()

    with pytest.raises(ProviderError, match="failed after 1 attempt"):
        provider._call(
            "embedding request",
            lambda: (_ for _ in ()).throw(FakeResponseError("endpoint missing", 404)),
        )
