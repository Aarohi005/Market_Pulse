"""
LLM provider factory.

Use :func:`create_llm_provider` to instantiate any supported LLM backend
by name.  API keys are resolved from ``kwargs`` first, then from the
standard environment variables.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Supported provider names
_SUPPORTED_PROVIDERS = ("anthropic", "openai", "ollama", "gemini")


def create_llm_provider(provider_name: str, **kwargs: Any):
    """Instantiate and return an LLM provider by name.

    Parameters
    ----------
    provider_name:
        Case-insensitive provider identifier.  One of:

        * ``"anthropic"`` — Anthropic Claude API (requires
          ``ANTHROPIC_API_KEY`` or ``api_key`` kwarg)
        * ``"openai"`` — OpenAI Chat Completions API (requires
          ``OPENAI_API_KEY`` or ``api_key`` kwarg)
        * ``"ollama"`` — Local Ollama server (no API key required;
          accepts ``base_url`` and ``model`` kwargs)

    **kwargs:
        Provider-specific keyword arguments forwarded to the provider
        constructor.  Common keys:

        * ``api_key`` (str) — overrides the environment variable
        * ``model`` (str) — model name / identifier
        * ``base_url`` (str) — base URL (Ollama / OpenAI-compatible)
        * ``timeout`` (int) — request timeout in seconds

    Returns
    -------
    LLMProvider
        A fully initialised provider instance ready to call.

    Raises
    ------
    ValueError
        When *provider_name* is not one of the supported identifiers.
    ImportError
        When the required SDK for the chosen provider is not installed.

    Examples
    --------
    >>> provider = create_llm_provider("anthropic", model="claude-3-5-sonnet-20241022")
    >>> provider = create_llm_provider("openai", model="gpt-4o")
    >>> provider = create_llm_provider("ollama", base_url="http://localhost:11434", model="llama3.1")
    """
    name = provider_name.lower().strip()

    if name == "anthropic":
        return _create_anthropic(**kwargs)
    elif name == "openai":
        return _create_openai(**kwargs)
    elif name == "ollama":
        return _create_ollama(**kwargs)
    elif name == "gemini":
        return _create_gemini(**kwargs)
    else:
        supported = ", ".join(f'"{p}"' for p in _SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unknown LLM provider: {provider_name!r}. "
            f"Supported providers are: {supported}."
        )


# ---------------------------------------------------------------------------
# Private factory helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(
    key_kwarg: Any,
    env_var: str,
    provider_name: str,
) -> str:
    """Resolve an API key from kwarg or environment variable.

    Parameters
    ----------
    key_kwarg:
        Value passed via ``api_key`` kwarg (may be ``None``).
    env_var:
        Name of the environment variable to fall back to.
    provider_name:
        Used in the warning / error message.

    Returns
    -------
    str
        The resolved API key string.

    Raises
    ------
    ValueError
        When neither source yields a non-empty key.
    """
    if key_kwarg:
        return str(key_kwarg)

    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        logger.debug(
            "Using %s API key from environment variable %s.", provider_name, env_var
        )
        return env_value

    raise ValueError(
        f"No API key supplied for provider {provider_name!r}. "
        f"Pass api_key=... or set the {env_var} environment variable."
    )


def _create_anthropic(**kwargs: Any):
    """Create an AnthropicProvider instance.

    Parameters
    ----------
    **kwargs:
        Forwarded to the constructor.  ``api_key`` is resolved from
        kwargs or ``ANTHROPIC_API_KEY`` environment variable.
    """
    try:
        from .providers.anthropic_provider import AnthropicProvider  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' SDK is required for the Anthropic provider. "
            "Install it with: pip install anthropic"
        ) from exc

    api_key = _resolve_api_key(
        kwargs.pop("api_key", None),
        "ANTHROPIC_API_KEY",
        "anthropic",
    )
    model = kwargs.pop("model", "claude-3-5-sonnet-20241022")
    logger.info("Creating AnthropicProvider with model=%s.", model)
    return AnthropicProvider(api_key=api_key, model=model, **kwargs)


def _create_openai(**kwargs: Any):
    """Create an OpenAIProvider instance.

    Parameters
    ----------
    **kwargs:
        Forwarded to the constructor.  ``api_key`` is resolved from
        kwargs or ``OPENAI_API_KEY`` environment variable.
    """
    try:
        from .providers.openai_provider import OpenAIProvider  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'openai' SDK is required for the OpenAI provider. "
            "Install it with: pip install openai"
        ) from exc

    api_key = _resolve_api_key(
        kwargs.pop("api_key", None),
        "OPENAI_API_KEY",
        "openai",
    )
    model = kwargs.pop("model", "gpt-4o")
    logger.info("Creating OpenAIProvider with model=%s.", model)
    return OpenAIProvider(api_key=api_key, model=model, **kwargs)


def _create_ollama(**kwargs: Any):
    """Create an OllamaProvider instance.

    Parameters
    ----------
    **kwargs:
        Forwarded to the constructor.  No API key is needed.
        Relevant kwargs: ``base_url``, ``model``, ``timeout``.
    """
    from .providers.ollama_provider import OllamaProvider

    base_url = kwargs.pop("base_url", "http://localhost:11434")
    model = kwargs.pop("model", "llama3.1")
    logger.info(
        "Creating OllamaProvider with base_url=%s, model=%s.", base_url, model
    )
    return OllamaProvider(base_url=base_url, model=model, **kwargs)


# ---------------------------------------------------------------------------
# Convenience: list supported providers
# ---------------------------------------------------------------------------


def _create_gemini(**kwargs: Any):
    """Create a GeminiProvider instance (LangChain backend).

    Parameters
    ----------
    **kwargs:
        Forwarded to the constructor.  ``api_key`` is resolved from        kwargs or the ``GOOGLE_API_KEY`` environment variable.
        ``model`` defaults to ``"gemini-2.5-flash"``.
    """
    try:
        from .providers.gemini_provider import GeminiProvider  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'langchain-google-genai' package is required for the Gemini provider. "
            "Install it with: pip install langchain-google-genai"
        ) from exc

    # API key is optional — LangChain will fall back to GOOGLE_API_KEY env var
    api_key = kwargs.pop("api_key", None) or os.environ.get("GOOGLE_API_KEY", "").strip() or None
    model = kwargs.pop("model", "gemini-2.5-flash")
    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", 2048)

    logger.info("Creating GeminiProvider with model=%s.", model)
    return GeminiProvider(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def list_supported_providers() -> tuple[str, ...]:
    """
    Return the supported LLM provider names.

    Returns
    -------
    tuple[str, ...]
        Lowercase provider names that can be passed to
        create_llm_provider().
    """
    return _SUPPORTED_PROVIDERS
