"""
src/utils/groq_client.py
========================
Centralised Groq API wrapper with live model discovery and auto-fallback.

Key behaviours:
    1. API key resolution  - Colab Secrets → env var → .env file
    2. Live model discovery - calls client.models.list() ONCE at init,
                              caches the set of currently active model IDs
    3. Auto model selection - walks model_priority_list (from config.yaml)
                              and picks the first live model; if a model gets
                              deprecated overnight the next run auto-upgrades
    4. Retry + backoff      - transient failures retried with exp backoff
    5. Never raises         - all failures logged, return None, pipeline continues

Priority list (from config/config.yaml):
    openai/gpt-oss-20b    → primary   (1000 t/s, Groq production)
    openai/gpt-oss-120b   → secondary (500 t/s,  Groq production)
    qwen/qwen3.6-27b      → tertiary  (500 t/s,  Groq preview)
    llama-3.3-70b-versatile  → safety net (deprecated Aug 16 2026)
    llama-3.1-8b-instant     → safety net (deprecated Aug 16 2026)

Usage:
    client = GroqClient(config=config)
    response = client.chat_completion(prompt="Rate this text for toxicity.")
    parsed  = client.parse_json_response(response)
"""

import json
import os
import time
import logging
from typing import Optional, Set, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded fallback priority list
# (used when no config is supplied or config.yaml is missing)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PRIORITY: List[str] = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


# ─────────────────────────────────────────────────────────────────────────────
# API Key Resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_api_key() -> Optional[str]:
    """
    Locate the Groq API key from the first available source.

    Priority:
        1. Google Colab Secrets   (userdata.get('GROQ_API_KEY'))
        2. Environment variable   (os.environ['GROQ_API_KEY'])
        3. .env file              (python-dotenv)
    """
    # 1 ── Google Colab Secrets
    try:
        from google.colab import userdata
        key = userdata.get("GROQ_API_KEY")
        if key:
            logger.debug("API key resolved: Colab Secrets.")
            return key
    except (ImportError, Exception):
        pass

    # 2 ── Environment variable
    key = os.environ.get("GROQ_API_KEY")
    if key:
        logger.debug("API key resolved: environment variable.")
        return key

    # 3 ── .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.environ.get("GROQ_API_KEY")
        if key:
            logger.debug("API key resolved: .env file.")
            return key
    except ImportError:
        pass

    logger.warning(
        "GROQ_API_KEY not found in Colab Secrets, environment, or .env. "
        "LLM evaluators will return safe defaults."
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Groq Client
# ─────────────────────────────────────────────────────────────────────────────

class GroqClient:
    """
    Fail-safe Groq API wrapper with live model discovery and auto-fallback.

    On initialisation:
        - Resolves API key
        - Creates Groq SDK client
        - Calls models.list() to discover which models are currently live
        - Caches live model IDs in self._live_models

    On each chat_completion() call:
        - Resolves the best available model from the priority list
        - If the preferred/requested model is live → use it
        - If not → walk priority list, pick first live model
        - If discovery failed → try in order, let the API surface errors

    This means: if Groq deprecates a model tonight, tomorrow's first run
    automatically upgrades to the next model in the priority list with zero
    code or config changes required.

    Args:
        api_key: Explicit key (for testing/overrides). Omit to auto-resolve.
        config:  Full config dict (loaded from config.yaml). Reads
                 config["groq"]["model_priority_list"].
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.api_key = api_key or _resolve_api_key()
        self._client = None

        # Extract priority list from config, fall back to hardcoded default
        groq_cfg = (config or {}).get("groq", {})
        self._priority: List[str] = groq_cfg.get(
            "model_priority_list", _DEFAULT_PRIORITY
        )
        self._max_tokens: int = groq_cfg.get("max_tokens", 512)
        self._temperature: float = groq_cfg.get("temperature", 0.1)
        self._max_retries: int = groq_cfg.get("max_retries", 3)
        self._retry_delay: float = groq_cfg.get("retry_delay_seconds", 1.0)

        # Will be populated by _discover_live_models()
        # None = discovery failed (try all in order)
        # set  = confirmed live model IDs
        self._live_models: Optional[Set[str]] = None

        self._init_client()
        self._discover_live_models()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_client(self) -> None:
        """Attempt to create a Groq SDK client. Silently fails if unavailable."""
        if not self.api_key:
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            logger.info("GroqClient initialised.")
        except ImportError:
            logger.error("groq package missing. Install: pip install groq")
        except Exception as e:
            logger.error("GroqClient init failed: %s", e)

    def _discover_live_models(self) -> None:
        """
        Fetch currently live models from Groq's models.list() API.
        Called once at init. Result cached in self._live_models.

        If the API call fails (network error, bad key, etc.) self._live_models
        stays None and _resolve_model() falls back to trying in priority order.
        """
        if not self._client:
            return

        try:
            response = self._client.models.list()
            self._live_models = {m.id for m in response.data}
            logger.info(
                "Model discovery: %d live models found on Groq.", len(self._live_models)
            )
            # Log status of each priority model so it's visible in logs
            for mid in self._priority:
                status = "✓ live" if mid in self._live_models else "✗ not found (deprecated?)"
                logger.debug("  Priority check | %-40s %s", mid, status)

        except Exception as exc:
            logger.warning(
                "Model discovery failed: %s. Will attempt models in priority order.", exc
            )
            self._live_models = None   # None = discovery failed

    def _resolve_model(self, preferred: Optional[str] = None) -> str:
        """
        Return the best available model to use for a chat completion call.

        Resolution order:
            1. preferred is live            → return preferred
            2. Walk priority list           → return first live model
            3. Discovery failed (None)      → return preferred or priority[0]
            4. Nothing in priority is live  → return priority[0] (API will error)

        Args:
            preferred: A specific model ID to try first. Pass None for
                       fully automatic selection from the priority list.

        Returns:
            str: Model ID to use in the API call.
        """
        # Discovery failed - try preferred or first in priority
        if self._live_models is None:
            chosen = preferred if preferred else self._priority[0]
            logger.debug("Discovery unavailable - using: %s", chosen)
            return chosen

        # Preferred model is live - use it
        if preferred and preferred in self._live_models:
            return preferred

        # Walk priority list for first live model
        for mid in self._priority:
            if mid in self._live_models:
                if preferred and mid != preferred:
                    logger.info(
                        "Model '%s' unavailable. Auto-selected from priority list: '%s'",
                        preferred, mid
                    )
                return mid

        # Nothing in priority list is live - last resort
        fallback = preferred if preferred else self._priority[0]
        logger.error(
            "No models from priority list are live. Attempting '%s' as last resort. "
            "Check config.yaml model_priority_list.", fallback
        )
        return fallback

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """True when the SDK client is initialised and ready."""
        return self._client is not None

    @property
    def resolved_model(self) -> Optional[str]:
        """The model that will be used by default (first live in priority list)."""
        if not self.is_available:
            return None
        return self._resolve_model()

    def chat_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Optional[str]:
        """
        Send a chat completion request with automatic model resolution and retry.

        Args:
            prompt:       The user-role message.
            model:        Optional preferred model ID. If None or not live,
                          auto-selects best available from priority list.
            system_prompt: Optional system-role message.
            temperature:  Sampling temperature. Defaults to config value.
            max_tokens:   Max tokens in response. Defaults to config value.
            max_retries:  Retry attempts. Defaults to config value.
            retry_delay:  Base retry delay in seconds. Defaults to config value.

        Returns:
            str | None: Raw text from the model, or None on all failures.
        """
        if not self._client:
            logger.warning("chat_completion called but client is unavailable.")
            return None

        # ── Resolve model (auto-selects if preferred is gone/deprecated) ──────
        resolved = self._resolve_model(preferred=model)

        # Use config defaults if not overridden per-call
        temperature = temperature if temperature is not None else self._temperature
        max_tokens  = max_tokens  if max_tokens  is not None else self._max_tokens
        max_retries = max_retries if max_retries is not None else self._max_retries
        retry_delay = retry_delay if retry_delay is not None else self._retry_delay

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # ── Retry loop ────────────────────────────────────────────────────────
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                response = self._client.chat.completions.create(
                    model=resolved,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = round((time.time() - t0) * 1000)
                text = response.choices[0].message.content
                logger.debug(
                    "Groq OK | model=%s | tokens=%s | latency=%dms",
                    resolved, response.usage.total_tokens, latency_ms,
                )
                return text

            except Exception as exc:
                logger.warning(
                    "Groq attempt %d/%d failed (model=%s): %s - %s",
                    attempt, max_retries, resolved, type(exc).__name__, exc
                )
                if attempt < max_retries:
                    sleep = retry_delay * (2 ** (attempt - 1))
                    logger.debug("Retrying in %.1fs...", sleep)
                    time.sleep(sleep)

        logger.error("All %d Groq attempts failed. Returning None.", max_retries)
        return None

    def parse_json_response(self, response: Optional[str]) -> Optional[dict]:
        """
        Safely parse JSON from a Groq response string.

        Handles raw JSON and markdown-fenced JSON (```json ... ```).

        Returns:
            dict | None
        """
        if not response:
            return None

        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            cleaned = "\n".join(inner).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed: %s | Preview: %.150s", exc, response)
            return None
