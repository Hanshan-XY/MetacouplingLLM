"""
LLM client abstraction using Protocol-based structural subtyping.

Provides a ``LLMClient`` protocol that any compatible client can satisfy,
plus ready-made adapters for OpenAI and Anthropic APIs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Maximum number of retries for transient API errors (rate limits, etc.)
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    usage: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """Protocol that any LLM client must satisfy.

    This uses structural subtyping — any object with a compatible ``chat``
    method will work, no explicit inheritance required.

    Example
    -------
    >>> class MyClient:
    ...     def chat(self, messages, temperature=0.7, max_tokens=None):
    ...         return LLMResponse(content="Hello!")
    >>> isinstance(MyClient(), LLMClient)
    True
    """

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a conversation to the LLM and get a response.

        Parameters
        ----------
        messages:
            Ordered list of messages (system, user, assistant).
        temperature:
            Sampling temperature (0.0 = deterministic, 1.0 = creative).
        max_tokens:
            Maximum tokens in the response. ``None`` uses the model default.

        Returns
        -------
        An ``LLMResponse`` with the assistant's reply and optional usage data.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI Adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    """Adapter wrapping an OpenAI client to satisfy ``LLMClient``.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` instance (or compatible).
    model:
        Model identifier (e.g., ``"gpt-4o"``, ``"gpt-4o-mini"``).
    """

    def __init__(self, client: Any, model: str = "gpt-4o") -> None:
        self._client = client
        self._model = model

    @property
    def raw_client(self) -> Any:
        """Expose the wrapped OpenAI client for advanced integrations."""
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages via the OpenAI Chat Completions API."""
        api_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            retry_kwargs = dict(kwargs)
            retried = False

            # GPT-5-family: temperature may only accept the default (1).
            if self._should_retry_without_temperature(exc, temperature):
                retry_kwargs.pop("temperature", None)
                retried = True

            # GPT-5-family: max_tokens → max_completion_tokens.
            if self._should_retry_with_max_completion_tokens(exc):
                val = retry_kwargs.pop("max_tokens", None)
                if val is not None:
                    retry_kwargs["max_completion_tokens"] = val
                retried = True

            # Model-specific max_tokens cap exceeded: parse the error
            # message and retry with the model's supported limit.
            capped_value = self._parse_max_tokens_cap(exc)
            if capped_value is not None:
                if "max_tokens" in retry_kwargs:
                    retry_kwargs["max_tokens"] = capped_value
                elif "max_completion_tokens" in retry_kwargs:
                    retry_kwargs["max_completion_tokens"] = capped_value
                retried = True

            # Rate limit: retry with exponential backoff.
            if not retried and self._is_rate_limit_error(exc):
                return self._retry_with_backoff(kwargs)

            if not retried:
                raise

            response = self._client.chat.completions.create(**retry_kwargs)

        # Safety: handle empty choices array
        if not response.choices:
            return LLMResponse(content="", usage={})

        choice = response.choices[0]
        usage_data: dict[str, int] = {}
        if response.usage:
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=choice.message.content or "",
            usage=usage_data,
        )

    def _retry_with_backoff(
        self,
        kwargs: dict[str, Any],
    ) -> LLMResponse:
        """Retry the API call with exponential backoff for rate limits."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            delay = _RETRY_BACKOFF_BASE * (2 ** attempt)
            time.sleep(delay)
            try:
                response = self._client.chat.completions.create(**kwargs)
                if not response.choices:
                    return LLMResponse(content="", usage={})
                choice = response.choices[0]
                usage_data: dict[str, int] = {}
                if response.usage:
                    usage_data = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                return LLMResponse(
                    content=choice.message.content or "",
                    usage=usage_data,
                )
            except Exception as exc:
                last_exc = exc
                if not self._is_rate_limit_error(exc):
                    raise
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _parse_max_tokens_cap(exc: Exception) -> int | None:
        """Detect a "max_tokens too large" error and return the model's cap.

        OpenAI returns messages like::

            "max_tokens is too large: 65000. This model supports at most
            16384 completion tokens, whereas you provided 65000."

        This parses the number after "supports at most" and returns it
        so the call can be retried with a compliant value. Returns
        ``None`` if the error doesn't match this pattern.
        """
        import re

        text = str(exc)
        lowered = text.lower()
        if "max_tokens" not in lowered and "max_completion_tokens" not in lowered:
            return None
        if "too large" not in lowered and "supports at most" not in lowered:
            return None
        match = re.search(r"supports at most\s+(\d+)", text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    @staticmethod
    def _should_retry_without_temperature(
        exc: Exception,
        temperature: float,
    ) -> bool:
        """Return True when the API rejected a non-default temperature.

        Some models accept only their default temperature setting. In that
        case, retrying without the parameter is equivalent to using the
        provider default and keeps the adapter compatible.
        """
        if temperature == 1:
            return False

        text = str(exc).lower()
        return "temperature" in text and "default (1)" in text

    @staticmethod
    def _should_retry_with_max_completion_tokens(exc: Exception) -> bool:
        """Return True when the API rejected ``max_tokens``.

        OpenAI's GPT-5-family models require ``max_completion_tokens``
        instead of ``max_tokens``.  Detecting this from the error message
        lets the adapter transparently support both old and new models.
        """
        text = str(exc).lower()
        return "max_tokens" in text and "max_completion_tokens" in text

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """Return True for rate-limit (429) or server (5xx) errors."""
        text = str(exc).lower()
        if "rate" in text and "limit" in text:
            return True
        if "429" in text:
            return True
        # Transient server errors worth retrying
        if any(code in text for code in ("500", "502", "503", "529")):
            return True
        return False


# ---------------------------------------------------------------------------
# Anthropic Adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    """Adapter wrapping an Anthropic client to satisfy ``LLMClient``.

    The Anthropic API handles system prompts differently (as a separate
    parameter rather than a message), so this adapter extracts the system
    message automatically.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` instance (or compatible).
    model:
        Model identifier (e.g., ``"claude-sonnet-4-20250514"``).
    """

    # Models that reject the ``temperature`` parameter (Opus 4.7 removed
    # it along with top_p / top_k). When the configured model is in this
    # set, we omit ``temperature`` from the request upfront. As a safety
    # net for unknown future models, we also catch the
    # "temperature is deprecated" error and retry once without it.
    _MODELS_WITHOUT_TEMPERATURE: frozenset[str] = frozenset({
        "claude-opus-4-7",
    })

    # When ``max_tokens`` exceeds this threshold, switch to streaming via
    # ``client.messages.stream`` + ``.get_final_message``. The Anthropic
    # SDK refuses non-streaming requests whose estimated generation time
    # would exceed the 10-minute non-streaming budget. For Opus 4.7 the
    # practical cutoff is around 21K output tokens; we use 16384 as a
    # safe margin.
    _STREAMING_MAX_TOKENS_THRESHOLD: int = 16384

    def __init__(self, client: Any, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = client
        self._model = model

    @property
    def raw_client(self) -> Any:
        """Expose the wrapped Anthropic client for advanced integrations."""
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages via the Anthropic Messages API."""
        # Extract system messages (combine if multiple are present)
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        system_text = "\n\n".join(system_parts) if system_parts else ""

        # Anthropic requires a positive max_tokens value
        effective_max = max_tokens if max_tokens and max_tokens > 0 else 4096

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": conversation,
            "max_tokens": effective_max,
        }
        # Skip temperature for models that have removed it (Opus 4.7+).
        if self._model not in self._MODELS_WITHOUT_TEMPERATURE:
            kwargs["temperature"] = temperature
        if system_text:
            kwargs["system"] = system_text

        # Retry with exponential backoff for rate-limit / transient errors,
        # with a one-shot retry-without-temperature fallback for unknown
        # models that reject temperature at runtime.
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._issue_request(kwargs)
                break
            except Exception as exc:
                last_exc = exc
                # One-shot fallback: API rejected temperature on a model
                # not in our compile-time deny list. Strip it and retry
                # the same call; only do this once.
                if (
                    "temperature" in kwargs
                    and self._is_temperature_deprecated_error(exc)
                ):
                    kwargs.pop("temperature", None)
                    response = self._issue_request(kwargs)
                    break
                text = str(exc).lower()
                is_transient = (
                    ("rate" in text and "limit" in text)
                    or any(c in text for c in ("429", "500", "502", "503", "529"))
                    or "overloaded" in text
                )
                if not is_transient or attempt >= _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
        else:
            raise last_exc  # type: ignore[misc]

        # Extract text content from response blocks
        content_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                content_text += block.text

        usage_data: dict[str, int] = {}
        if response.usage:
            usage_data = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return LLMResponse(content=content_text, usage=usage_data)

    @staticmethod
    def _is_temperature_deprecated_error(exc: Exception) -> bool:
        """Detect Anthropic's "temperature is deprecated for this model" 400.

        Opus 4.7 (and any future model that drops sampling parameters)
        returns this exact 400 when ``temperature`` is sent. Returns
        ``True`` so the adapter can pop the parameter and retry once.
        """
        text = str(exc).lower()
        return "temperature" in text and "deprecated" in text

    def _issue_request(self, kwargs: dict[str, Any]) -> Any:
        """Issue a single API request, switching to streaming for large
        ``max_tokens`` so the SDK doesn't reject the call as exceeding
        the 10-minute non-streaming budget.

        Returns the same ``Message`` object regardless of which path is
        taken — :meth:`anthropic.MessageStreamManager.get_final_message`
        produces a ``Message`` with the same shape as
        :meth:`anthropic.Anthropic.messages.create`.
        """
        max_tokens = kwargs.get("max_tokens", 4096)
        if max_tokens > self._STREAMING_MAX_TOKENS_THRESHOLD:
            with self._client.messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        return self._client.messages.create(**kwargs)


# ---------------------------------------------------------------------------
# Gemini Adapter
# ---------------------------------------------------------------------------

class GeminiAdapter:
    """Adapter wrapping a Google Gemini client to satisfy ``LLMClient``.

    Built against the new **``google.genai`` SDK** (the unified Google
    GenAI SDK that replaced the deprecated ``google.generativeai``
    package in 2025). Pass a ``google.genai.Client`` instance.

    Gemini's API has a different message shape than OpenAI/Anthropic:
      * messages use a ``parts`` array, not flat strings;
      * the ``"assistant"`` role is called ``"model"``;
      * the system instruction is a separate ``system_instruction``
        field on the request config, not a conversation message.

    This adapter normalizes those differences so callers can pass a
    standard ``list[Message]`` exactly like they would for OpenAI or
    Anthropic.

    Parameters
    ----------
    client:
        A ``google.genai.Client`` instance configured with your API key
        (``Client(api_key=...)``).
    model:
        Model identifier (e.g., ``"gemini-2.5-flash"``,
        ``"gemini-2.5-pro"``).

    Examples
    --------
    >>> from google import genai
    >>> from metacouplingllm import GeminiAdapter, MetacouplingAssistant
    >>> client = genai.Client(api_key="...")
    >>> advisor = MetacouplingAssistant(
    ...     GeminiAdapter(client, model="gemini-2.5-flash")
    ... )
    """

    def __init__(self, client: Any, model: str = "gemini-2.5-flash") -> None:
        self._client = client
        self._model = model

    @property
    def raw_client(self) -> Any:
        """Expose the wrapped ``google.genai.Client`` instance."""
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages via the Gemini ``models.generate_content`` API."""
        # Gemini handles system separately (like Anthropic).
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_parts.append(msg.content)
                continue
            # Gemini calls the assistant role "model".
            role = "model" if msg.role == "assistant" else "user"
            contents.append(
                {"role": role, "parts": [{"text": msg.content}]}
            )

        system_instruction = (
            "\n\n".join(system_parts) if system_parts else None
        )

        # Build the GenerateContentConfig dict-style for cross-version
        # compatibility (the new SDK accepts both dicts and typed
        # GenerateContentConfig objects on `models.generate_content(config=...)`).
        config: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if system_instruction is not None:
            config["system_instruction"] = system_instruction

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        # Extract text. Gemini's `.text` raises if the response was
        # blocked or empty; guard against that.
        try:
            content_text = response.text or ""
        except Exception:
            content_text = ""

        usage_data: dict[str, int] = {}
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            usage_data = {
                "input_tokens": int(getattr(usage_meta, "prompt_token_count", 0)),
                "output_tokens": int(getattr(usage_meta, "candidates_token_count", 0)),
                "total_tokens": int(getattr(usage_meta, "total_token_count", 0)),
            }

        return LLMResponse(content=content_text, usage=usage_data)


# ---------------------------------------------------------------------------
# Grok (xAI) Adapter
# ---------------------------------------------------------------------------

class GrokAdapter:
    """Adapter wrapping an xAI Grok client to satisfy ``LLMClient``.

    Grok's API is OpenAI-protocol-compatible. Pass an ``openai.OpenAI``
    instance configured with ``base_url="https://api.x.ai/v1"``.

    A separate class (rather than reusing ``OpenAIAdapter``) is needed
    so the auto-wiring in ``MetacouplingAssistant`` can route web search
    to ``GrokWebSearchBackend`` (xAI Live Search) instead of
    ``OpenAIWebSearchBackend`` (OpenAI's ``web_search`` tool).

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` instance configured with the xAI base URL.
    model:
        Model identifier (e.g., ``"grok-3"``, ``"grok-2"``).

    Examples
    --------
    >>> from openai import OpenAI
    >>> from metacouplingllm import GrokAdapter, MetacouplingAssistant
    >>> client = OpenAI(api_key="...", base_url="https://api.x.ai/v1")
    >>> advisor = MetacouplingAssistant(GrokAdapter(client, model="grok-3"))
    """

    def __init__(self, client: Any, model: str = "grok-3") -> None:
        self._client = client
        self._model = model

    @property
    def raw_client(self) -> Any:
        """Expose the wrapped xAI client (an ``openai.OpenAI`` instance)."""
        return self._client

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send messages via xAI's chat-completions API (OpenAI-shaped)."""
        api_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": api_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**kwargs)

        if not response.choices:
            return LLMResponse(content="", usage={})

        choice = response.choices[0]
        usage_data: dict[str, int] = {}
        if response.usage:
            usage_data = {
                "prompt_tokens": int(response.usage.prompt_tokens),
                "completion_tokens": int(response.usage.completion_tokens),
                "total_tokens": int(response.usage.total_tokens),
            }
        return LLMResponse(
            content=choice.message.content or "",
            usage=usage_data,
        )
