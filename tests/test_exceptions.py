from __future__ import annotations

from pyaistack.exceptions import ProviderError, format_error


def raise_provider_error() -> None:
    raise ProviderError(
        "Model \"embeddinggemma\" is not available.\nRun: ollama pull embeddinggemma"
    )


def test_format_error_shows_the_application_frame_without_library_frames() -> None:
    try:
        raise_provider_error()
    except ProviderError as error:
        formatted = format_error(error)

    assert formatted.startswith("Traceback (most recent call last):")
    assert 'File "' in formatted
    assert "test_exceptions.py" in formatted
    assert "raise_provider_error()" in formatted
    assert "src/pyaistack" not in formatted
    assert formatted.endswith("Run: ollama pull embeddinggemma")
