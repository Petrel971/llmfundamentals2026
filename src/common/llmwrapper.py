import dataclasses
import functools
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
import openai
from dotenv import load_dotenv
from loguru import logger

Provider = Literal["anthropic", "openai"]

# Both providers go through OpenRouter: Claude with the Anthropic SDK, OpenAI with the OpenAI SDK.
OPENROUTER_ANTHROPIC_URL = "https://openrouter.ai/api"
OPENROUTER_OPENAI_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "anthropic/claude-haiku-4.5",
    "openai": "openai/gpt-4.1-mini",
}

# USD per million tokens, from https://openrouter.ai/api/v1/models (checked 2026-10-14).
PRICES_PER_MTOK = {
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "anthropic/claude-sonnet-5.5": {"input": 2.00, "output": 10.00},
    "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "openai/gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}

# Temporary errors worth retrying. A BadRequestError (400) is not here: it would fail again.
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
)

CACHE_DB = Path(__file__).resolve().parents[2] / ".cache" / "llmwrapper_cache.sqlite"  # git-ignored
CACHE_TTL_SECONDS = 30 * 24 * 3600  # entries live one month


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None  # None when the model's price is unknown
    stop_reason: str | None = None  # e.g. "end_turn" or "max_tokens" (Claude), "stop" or "length" (OpenAI)
    latency_seconds: float | None = None  # wall-clock time of ask(), including retries
    from_cache: bool = False


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def retry_with_backoff(max_retries=3, base_delay=1.0):
    """Retry the function on RETRYABLE_ERRORS, waiting base_delay * 2**attempt seconds each time."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as error:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * 2**attempt
                    logger.warning("{} (retry {}/{} in {} s)", type(error).__name__, attempt + 1, max_retries, delay)
                    time.sleep(delay)

        return wrapper

    return decorator


class LLMWrapper:
    """Handles API connection to the chosen LLM."""

    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        max_tokens: int = 1000,
        use_cache: bool = True,
    ):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider: {provider!r}")

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (check your .env file)")

        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.max_tokens = max_tokens
        # max_retries=0: our own decorator does the retries, the SDKs must not add theirs
        self._anthropic = (
            anthropic.Anthropic(base_url=OPENROUTER_ANTHROPIC_URL, auth_token=api_key, max_retries=0)
            if provider == "anthropic"
            else None
        )
        self._openai = (
            openai.OpenAI(base_url=OPENROUTER_OPENAI_URL, api_key=api_key, max_retries=0)
            if provider == "openai"
            else None
        )
        self._db = self._open_cache() if use_cache else None

    def ask(
        self,
        prompt: str,
        system: str | None = None,
        with_streaming: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Send a single prompt and return the text, the token usage, the cost and the latency.

        with_streaming: a function called with each piece of text as soon as it arrives.
        """
        start = time.perf_counter()
        key = self._cache_key(prompt, system)
        response = self._cache_get(key)
        if response:
            logger.debug("Cache hit for {}/{}", self.provider, self.model)
            if with_streaming:
                with_streaming(response.text)
        else:
            logger.debug("Sending request to {}/{}", self.provider, self.model)
            if self.provider == "anthropic":
                response = self._ask_anthropic(prompt, system, with_streaming)
            else:
                response = self._ask_openai(prompt, system, with_streaming)
            self._cache_put(key, response)
        return dataclasses.replace(response, latency_seconds=time.perf_counter() - start)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _ask_anthropic(self, prompt: str, system: str | None, on_text: Callable[[str], None] | None) -> LLMResponse:
        kwargs = {"system": system} if system else {}
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if on_text:
            with self._anthropic.messages.stream(**request) as stream:
                for text in stream.text_stream:
                    on_text(text)
                response = stream.get_final_message()
        else:
            response = self._anthropic.messages.create(**request)
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude refused the request: {response.stop_details}")
        return self._make_response(
            text="".join(b.text for b in response.content if b.type == "text"),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _ask_openai(self, prompt: str, system: str | None, on_text: Callable[[str], None] | None) -> LLMResponse:
        # OpenAI has no "system" argument: the system prompt is the first message
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        request = dict(model=self.model, max_completion_tokens=self.max_tokens, messages=messages)
        if on_text:
            return self._stream_openai(request, on_text)
        response = self._openai.chat.completions.create(**request)
        return self._make_response(
            text=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            stop_reason=response.choices[0].finish_reason,
        )

    def _stream_openai(self, request: dict, on_text: Callable[[str], None]) -> LLMResponse:
        # include_usage: the last chunk carries the token counts (and has no choices)
        stream = self._openai.chat.completions.create(**request, stream=True, stream_options={"include_usage": True})
        pieces, stop_reason, usage = [], None, None
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content
            if text:
                pieces.append(text)
                on_text(text)
            stop_reason = chunk.choices[0].finish_reason or stop_reason
        return self._make_response(
            text="".join(pieces),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=stop_reason,
        )

    def _make_response(self, text: str, input_tokens: int, output_tokens: int, stop_reason: str | None) -> LLMResponse:
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd(self.model, input_tokens, output_tokens),
            stop_reason=stop_reason,
        )

    # --- cache ---

    @staticmethod
    def _open_cache() -> sqlite3.Connection:
        CACHE_DB.parent.mkdir(exist_ok=True)
        db = sqlite3.connect(CACHE_DB)
        db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, created_at REAL)")
        return db

    def _cache_key(self, prompt: str, system: str | None) -> str:
        """SHA-256 of everything that changes the answer: the same inputs always give the same key."""
        inputs = [self.provider, self.model, self.max_tokens, system, prompt]
        return hashlib.sha256(json.dumps(inputs).encode()).hexdigest()

    def _cache_get(self, key: str) -> LLMResponse | None:
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT response FROM cache WHERE key = ? AND created_at >= ?",
            (key, time.time() - CACHE_TTL_SECONDS),
        ).fetchone()
        if row is None:
            return None
        # a cached answer costs nothing
        return LLMResponse(**json.loads(row[0]), cost_usd=0.0, from_cache=True)

    def _cache_put(self, key: str, response: LLMResponse) -> None:
        if self._db is None:
            return
        saved = {
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "stop_reason": response.stop_reason,
        }
        with self._db:  # commits the insert
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(saved), time.time()))


if __name__ == "__main__":
    # allows to quickly check that LLMWrapper works
    load_dotenv()  # reads OPENROUTER_API_KEY from the .env file
    for provider in ("anthropic", "openai"):
        llm = LLMWrapper(provider)
        for attempt in ("first call", "second call (cache)"):
            response = llm.ask("Say hello in French.")
            logger.info("{} {}: {}", provider, attempt, response)
        logger.info("{} streaming:", provider)
        llm_no_cache = LLMWrapper(provider, use_cache=False)
        response = llm_no_cache.ask("Count from 1 to 10 in French.", with_streaming=lambda t: print(t, end="", flush=True))
        print()
        logger.info("{} streaming done: {}", provider, response)
