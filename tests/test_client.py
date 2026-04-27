"""Tests for llm/client.py — Protocol, Message, LLMResponse, and adapters."""

from metacouplingllm.llm.client import (
    AnthropicAdapter,
    LLMClient,
    LLMResponse,
    Message,
    OpenAIAdapter,
)


class TestMessage:
    def test_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_roles(self):
        for role in ("system", "user", "assistant"):
            msg = Message(role=role, content="test")
            assert msg.role == role


class TestLLMResponse:
    def test_creation(self):
        resp = LLMResponse(content="Answer")
        assert resp.content == "Answer"
        assert resp.usage == {}

    def test_with_usage(self):
        resp = LLMResponse(
            content="Answer",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
        assert resp.usage["prompt_tokens"] == 100


class TestLLMClientProtocol:
    def test_custom_client_satisfies_protocol(self):
        class MyClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="ok")

        client = MyClient()
        assert isinstance(client, LLMClient)

    def test_invalid_client_does_not_satisfy(self):
        class BadClient:
            def generate(self, prompt):
                return "hi"

        client = BadClient()
        assert not isinstance(client, LLMClient)


class TestOpenAIAdapter:
    def test_chat_calls_openai_api(self):
        """Test that OpenAIAdapter correctly wraps the OpenAI API."""

        class FakeChoice:
            def __init__(self):
                self.message = type("Msg", (), {"content": "Analyzed!"})()

        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = 20
            total_tokens = 30

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        class FakeOpenAI:
            def __init__(self):
                self.chat = type("Chat", (), {
                    "completions": type("Comp", (), {
                        "create": staticmethod(lambda **kwargs: FakeResponse())
                    })()
                })()

        adapter = OpenAIAdapter(FakeOpenAI(), model="gpt-4o-mini")
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        result = adapter.chat(messages, temperature=0.5)

        assert isinstance(result, LLMResponse)
        assert result.content == "Analyzed!"
        assert result.usage["total_tokens"] == 30

    def test_chat_retries_without_temperature_when_model_requires_default(self):
        """Adapter should retry when a model rejects non-default temperature."""

        class FakeChoice:
            def __init__(self):
                self.message = type("Msg", (), {"content": "Retried!"})()

        class FakeUsage:
            prompt_tokens = 11
            completion_tokens = 7
            total_tokens = 18

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        calls = []

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                if "temperature" in kwargs:
                    raise Exception(
                        "Unsupported value: 'temperature' does not support "
                        "0.1 with this model. Only the default (1) value "
                        "is supported."
                    )
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self):
                self.chat = type("Chat", (), {
                    "completions": FakeCompletions()
                })()

        adapter = OpenAIAdapter(FakeOpenAI(), model="gpt-5")
        messages = [Message(role="user", content="Hello")]

        result = adapter.chat(messages, temperature=0.1)

        assert result.content == "Retried!"
        assert len(calls) == 2
        assert "temperature" in calls[0]
        assert "temperature" not in calls[1]


    def test_chat_retries_max_tokens_to_max_completion_tokens(self):
        """Adapter should retry with max_completion_tokens for GPT-5 family."""

        class FakeChoice:
            def __init__(self):
                self.message = type("Msg", (), {"content": "OK!"})()

        class FakeUsage:
            prompt_tokens = 5
            completion_tokens = 3
            total_tokens = 8

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        calls = []

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                calls.append(dict(kwargs))
                if "max_tokens" in kwargs:
                    raise Exception(
                        "Unsupported parameter: 'max_tokens' is not supported "
                        "with this model. Use 'max_completion_tokens' instead."
                    )
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self):
                self.chat = type("Chat", (), {
                    "completions": FakeCompletions()
                })()

        adapter = OpenAIAdapter(FakeOpenAI(), model="gpt-5.4")
        messages = [Message(role="user", content="Hi")]

        result = adapter.chat(messages, max_tokens=100)

        assert result.content == "OK!"
        assert len(calls) == 2
        assert "max_tokens" in calls[0]
        assert "max_tokens" not in calls[1]
        assert calls[1]["max_completion_tokens"] == 100

    def test_chat_handles_empty_choices(self):
        """Adapter should return empty content for empty choices array."""

        class FakeResponse:
            choices = []
            usage = None

        class FakeOpenAI:
            def __init__(self):
                self.chat = type("Chat", (), {
                    "completions": type("Comp", (), {
                        "create": staticmethod(lambda **kwargs: FakeResponse())
                    })()
                })()

        adapter = OpenAIAdapter(FakeOpenAI(), model="gpt-4o")
        result = adapter.chat([Message(role="user", content="Hi")])

        assert result.content == ""
        assert result.usage == {}

    def test_chat_retries_on_rate_limit(self):
        """Adapter should retry with backoff on rate limit errors."""

        class FakeChoice:
            def __init__(self):
                self.message = type("Msg", (), {"content": "After retry"})()

        class FakeUsage:
            prompt_tokens = 5
            completion_tokens = 3
            total_tokens = 8

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        attempt_count = 0

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count <= 1:
                    raise Exception("Rate limit exceeded (429)")
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self):
                self.chat = type("Chat", (), {
                    "completions": FakeCompletions()
                })()

        # Patch the backoff to be fast
        import metacouplingllm.llm.client as _mod
        old_base = _mod._RETRY_BACKOFF_BASE
        _mod._RETRY_BACKOFF_BASE = 0.01
        try:
            adapter = OpenAIAdapter(FakeOpenAI(), model="gpt-4o")
            result = adapter.chat([Message(role="user", content="Hi")])
            assert result.content == "After retry"
            assert attempt_count == 2
        finally:
            _mod._RETRY_BACKOFF_BASE = old_base

    def test_is_rate_limit_error_detection(self):
        """Test various rate limit / transient error messages."""
        assert OpenAIAdapter._is_rate_limit_error(
            Exception("Rate limit exceeded")
        )
        assert OpenAIAdapter._is_rate_limit_error(
            Exception("Error 429: Too many requests")
        )
        assert OpenAIAdapter._is_rate_limit_error(
            Exception("Error 503: Service unavailable")
        )
        assert not OpenAIAdapter._is_rate_limit_error(
            Exception("Invalid API key")
        )
        assert not OpenAIAdapter._is_rate_limit_error(
            Exception("Model not found")
        )

    def test_parse_max_tokens_cap(self):
        """Test parsing model-specific max_tokens cap errors."""
        # Real OpenAI error for gpt-4o-mini
        err = Exception(
            "Error code: 400 - {'error': {'message': 'max_tokens is "
            "too large: 65000. This model supports at most 16384 "
            "completion tokens, whereas you provided 65000.', "
            "'type': 'invalid_request_error'}}"
        )
        assert OpenAIAdapter._parse_max_tokens_cap(err) == 16384

        # Variant wording
        err2 = Exception(
            "max_tokens too large. This model supports at most 4096 tokens."
        )
        assert OpenAIAdapter._parse_max_tokens_cap(err2) == 4096

        # Non-matching errors return None
        assert OpenAIAdapter._parse_max_tokens_cap(
            Exception("Rate limit exceeded")
        ) is None
        assert OpenAIAdapter._parse_max_tokens_cap(
            Exception("Invalid API key")
        ) is None

    def test_chat_retries_with_capped_max_tokens(self):
        """Adapter should retry with model's max when max_tokens too large."""

        class FakeChoice:
            def __init__(self):
                self.message = type("Msg", (), {"content": "Capped!"})()

        class FakeUsage:
            prompt_tokens = 5
            completion_tokens = 3
            total_tokens = 8

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        calls: list[dict] = []

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                calls.append(dict(kwargs))
                mt = kwargs.get("max_tokens")
                if mt is not None and mt > 16384:
                    raise Exception(
                        "Error code: 400 - max_tokens is too large: "
                        f"{mt}. This model supports at most 16384 "
                        "completion tokens, whereas you provided "
                        f"{mt}."
                    )
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self):
                self.chat = type("Chat", (), {
                    "completions": FakeCompletions()
                })()

        adapter = OpenAIAdapter(FakeOpenAI(), model="gpt-4o-mini")
        result = adapter.chat(
            [Message(role="user", content="Hi")],
            max_tokens=65000,
        )

        assert result.content == "Capped!"
        assert len(calls) == 2
        assert calls[0]["max_tokens"] == 65000
        assert calls[1]["max_tokens"] == 16384

    def test_combined_temperature_and_max_tokens_retry(self):
        """When both temperature and max_tokens are rejected, retry fixes both."""

        class FakeChoice:
            def __init__(self):
                self.message = type("Msg", (), {"content": "Both fixed!"})()

        class FakeUsage:
            prompt_tokens = 5
            completion_tokens = 3
            total_tokens = 8

        class FakeResponse:
            choices = [FakeChoice()]
            usage = FakeUsage()

        calls = []

        class FakeCompletions:
            @staticmethod
            def create(**kwargs):
                calls.append(dict(kwargs))
                if "temperature" in kwargs or "max_tokens" in kwargs:
                    raise Exception(
                        "Unsupported: 'temperature' only supports default (1), "
                        "and 'max_tokens' should be 'max_completion_tokens'"
                    )
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self):
                self.chat = type("Chat", (), {
                    "completions": FakeCompletions()
                })()

        adapter = OpenAIAdapter(FakeOpenAI(), model="gpt-5")
        result = adapter.chat(
            [Message(role="user", content="Hi")],
            temperature=0.5,
            max_tokens=200,
        )

        assert result.content == "Both fixed!"
        assert len(calls) == 2
        assert "temperature" not in calls[1]
        assert "max_tokens" not in calls[1]
        assert calls[1]["max_completion_tokens"] == 200


class TestAnthropicAdapter:
    def test_chat_separates_system_message(self):
        """Test that AnthropicAdapter extracts system into separate param."""

        captured_kwargs = {}

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 15
            output_tokens = 25

        class FakeResponse:
            content = [FakeTextBlock("Framework analysis complete.")]
            usage = FakeUsage()

        class FakeMessages:
            @staticmethod
            def create(**kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        adapter = AnthropicAdapter(FakeAnthropic(), model="claude-sonnet-4-20250514")
        messages = [
            Message(role="system", content="You are a framework expert."),
            Message(role="user", content="Analyze my research."),
        ]
        result = adapter.chat(messages, temperature=0.3, max_tokens=2000)

        assert isinstance(result, LLMResponse)
        assert result.content == "Framework analysis complete."
        assert result.usage["input_tokens"] == 15

        # Verify system was passed separately
        assert captured_kwargs["system"] == "You are a framework expert."
        # Verify only user message in the messages list
        assert len(captured_kwargs["messages"]) == 1
        assert captured_kwargs["messages"][0]["role"] == "user"

    def test_combines_multiple_system_messages(self):
        """Multiple system messages should be joined with double newlines."""

        captured_kwargs = {}

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            content = [FakeTextBlock("OK")]
            usage = FakeUsage()

        class FakeMessages:
            @staticmethod
            def create(**kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        adapter = AnthropicAdapter(FakeAnthropic(), model="claude-sonnet-4-20250514")
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="system", content="Be concise."),
            Message(role="user", content="Hello"),
        ]
        result = adapter.chat(messages)

        assert "You are helpful." in captured_kwargs["system"]
        assert "Be concise." in captured_kwargs["system"]
        assert "\n\n" in captured_kwargs["system"]
        assert len(captured_kwargs["messages"]) == 1

    def test_max_tokens_zero_defaults_to_4096(self):
        """max_tokens=0 should default to 4096."""

        captured_kwargs = {}

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            content = [FakeTextBlock("OK")]
            usage = FakeUsage()

        class FakeMessages:
            @staticmethod
            def create(**kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        adapter = AnthropicAdapter(FakeAnthropic(), model="test")
        adapter.chat(
            [Message(role="user", content="Hi")],
            max_tokens=0,
        )

        assert captured_kwargs["max_tokens"] == 4096

    def test_retries_on_overloaded_error(self):
        """Adapter should retry when Anthropic reports overloaded."""

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            content = [FakeTextBlock("After retry")]
            usage = FakeUsage()

        attempt_count = 0

        class FakeMessages:
            @staticmethod
            def create(**kwargs):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count <= 1:
                    raise Exception("Anthropic API is overloaded (529)")
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        import metacouplingllm.llm.client as _mod
        old_base = _mod._RETRY_BACKOFF_BASE
        _mod._RETRY_BACKOFF_BASE = 0.01
        try:
            adapter = AnthropicAdapter(FakeAnthropic(), model="test")
            result = adapter.chat([Message(role="user", content="Hi")])
            assert result.content == "After retry"
            assert attempt_count == 2
        finally:
            _mod._RETRY_BACKOFF_BASE = old_base

    def test_skips_temperature_for_opus_4_7(self):
        """Opus 4.7 doesn't accept temperature — adapter must omit it."""

        captured_kwargs: dict[str, object] = {}

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            content = [FakeTextBlock("OK")]
            usage = FakeUsage()

        class FakeMessages:
            @staticmethod
            def create(**kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        adapter = AnthropicAdapter(FakeAnthropic(), model="claude-opus-4-7")
        adapter.chat(
            [Message(role="user", content="Hi")],
            temperature=0.7,
        )
        assert "temperature" not in captured_kwargs

    def test_keeps_temperature_for_other_models(self):
        """Older models still accept temperature — adapter must send it."""

        captured_kwargs: dict[str, object] = {}

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            content = [FakeTextBlock("OK")]
            usage = FakeUsage()

        class FakeMessages:
            @staticmethod
            def create(**kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        adapter = AnthropicAdapter(FakeAnthropic(), model="claude-sonnet-4-6")
        adapter.chat(
            [Message(role="user", content="Hi")],
            temperature=0.3,
        )
        assert captured_kwargs["temperature"] == 0.3

    def test_retries_without_temperature_on_deprecated_error(self):
        """If an unknown model rejects temperature at runtime, retry without."""

        captured_calls: list[dict[str, object]] = []

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            content = [FakeTextBlock("Recovered")]
            usage = FakeUsage()

        attempt_count = 0

        class FakeMessages:
            @staticmethod
            def create(**kwargs):
                nonlocal attempt_count
                attempt_count += 1
                captured_calls.append(dict(kwargs))
                if attempt_count == 1:
                    raise Exception(
                        "Error code: 400 - {'message': "
                        "'`temperature` is deprecated for this model.'}"
                    )
                return FakeResponse()

        class FakeAnthropic:
            messages = FakeMessages()

        # Use a model NOT in the deny set to exercise the runtime fallback.
        adapter = AnthropicAdapter(FakeAnthropic(), model="claude-future-x")
        result = adapter.chat(
            [Message(role="user", content="Hi")],
            temperature=0.5,
        )

        assert result.content == "Recovered"
        assert attempt_count == 2
        # First call sent temperature, second call did not.
        assert captured_calls[0].get("temperature") == 0.5
        assert "temperature" not in captured_calls[1]

    def test_is_temperature_deprecated_error_detection(self):
        """Standalone unit test for the deprecation-error detector."""
        assert AnthropicAdapter._is_temperature_deprecated_error(
            Exception("`temperature` is deprecated for this model.")
        )
        assert AnthropicAdapter._is_temperature_deprecated_error(
            Exception("Error 400: temperature is deprecated for this model")
        )
        # Non-matching errors
        assert not AnthropicAdapter._is_temperature_deprecated_error(
            Exception("Rate limit exceeded")
        )
        assert not AnthropicAdapter._is_temperature_deprecated_error(
            Exception("temperature must be between 0 and 1")  # different error
        )

    def test_uses_create_when_max_tokens_below_streaming_threshold(self):
        """Below the streaming threshold, the adapter calls
        client.messages.create (non-streaming) — regression check."""
        captured_kwargs = {}

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 10
            output_tokens = 5

        class FakeResponse:
            content = [FakeTextBlock("OK")]
            usage = FakeUsage()

        class FakeMessages:
            stream_called = False

            @staticmethod
            def create(**kwargs):
                captured_kwargs.update(kwargs)
                return FakeResponse()

            @staticmethod
            def stream(**kwargs):
                FakeMessages.stream_called = True
                raise AssertionError(
                    "stream should not be called for max_tokens below threshold"
                )

        class FakeAnthropic:
            messages = FakeMessages()

        adapter = AnthropicAdapter(FakeAnthropic(), model="test")
        adapter.chat(
            [Message(role="user", content="Hi")],
            max_tokens=8000,  # below 16384 threshold
        )
        assert captured_kwargs["max_tokens"] == 8000
        assert FakeMessages.stream_called is False

    def test_uses_streaming_when_max_tokens_above_streaming_threshold(self):
        """Above the streaming threshold, the adapter switches to
        client.messages.stream + .get_final_message — necessary so the
        Anthropic SDK doesn't reject the call as exceeding the
        10-minute non-streaming budget."""
        captured_kwargs = {}
        stream_entered = False
        stream_exited = False

        class FakeTextBlock:
            def __init__(self, text):
                self.text = text

        class FakeUsage:
            input_tokens = 12
            output_tokens = 7

        class FakeFinalMessage:
            content = [FakeTextBlock("Streamed reply")]
            usage = FakeUsage()

        class FakeStreamCM:
            def __init__(self, kwargs):
                captured_kwargs.update(kwargs)

            def __enter__(self):
                nonlocal stream_entered
                stream_entered = True
                return self

            def __exit__(self, exc_type, exc, tb):
                nonlocal stream_exited
                stream_exited = True
                return False

            def get_final_message(self):
                return FakeFinalMessage()

        class FakeMessages:
            create_called = False

            @staticmethod
            def create(**kwargs):
                FakeMessages.create_called = True
                raise AssertionError(
                    "create should not be called for max_tokens above threshold"
                )

            @staticmethod
            def stream(**kwargs):
                return FakeStreamCM(kwargs)

        class FakeAnthropic:
            messages = FakeMessages()

        adapter = AnthropicAdapter(FakeAnthropic(), model="test")
        result = adapter.chat(
            [Message(role="user", content="Hi")],
            max_tokens=32000,  # above 16384 threshold
        )

        assert result.content == "Streamed reply"
        assert captured_kwargs["max_tokens"] == 32000
        assert stream_entered and stream_exited
        assert FakeMessages.create_called is False
