"""
ollama_client.py
────────────────
Handles all communication with the local Ollama instance for the
two-pass brand analysis pipeline.

Responsibilities:
    - Pass 1a: Send individual text chunks to Gemma 4B and parse
               the response into validated ChunkSignals instances.
    - Pass 1b: Send images to Gemma vision and parse color signals.
               (stubbed here; implemented in vision_client.py)
    - Retry logic for malformed JSON responses, which occur with
               non-negligible frequency in small models.
    - JSON repair for common model output issues (markdown fences,
               trailing commas, truncated responses).

The client is intentionally stateless. All context management
(e.g. rolling summaries, conversation history) is handled upstream
in the pipeline orchestrator, not here.

Ollama API reference:
    https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import logging
import re
import time
from typing import Any

import httpx
from pydantic import ValidationError

from criteria_loader import CriteriaLoader, DimensionKey
from schemas import ChunkSignals

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL   = "http://localhost:11434"
DEFAULT_MODEL     = "gemma3:4b"
DEFAULT_TIMEOUT   = 300.0  # seconds; larger models on CPU can be slow — raise if needed
MAX_RETRIES       = 3
RETRY_DELAY       = 2.0    # seconds between retries


# ── JSON repair utilities ─────────────────────────────────────────────────
#
# Small models frequently wrap JSON in markdown fences or add minor syntax
# errors. These helpers attempt lightweight repairs before falling back to
# a retry, avoiding unnecessary round-trips to the model.

def _strip_markdown_fences(text: str) -> str:
    """
    Removes markdown code fences that models sometimes wrap JSON in.
    Handles ```json ... ``` and ``` ... ``` variants.
    """
    text = text.strip()
    # Match optional language tag after opening fence
    pattern = r"^```(?:json)?\s*(.*?)\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    """
    Extracts the first complete JSON object from a string.

    Models sometimes produce a JSON object followed by explanatory text.
    This finds the outermost { } pair and discards everything after it.
    Returns the original string if no valid extraction is possible.
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text  # Unbalanced braces — return as-is and let the parser fail


def _repair_json(raw: str) -> str:
    """
    Applies a sequence of lightweight repairs to raw model output.

    Repair steps (in order):
        1. Strip markdown fences
        2. Extract first JSON object (discard trailing prose)

    Does not attempt structural repairs (e.g. missing commas, truncation).
    If these repairs are insufficient, the caller should retry the request.
    """
    raw = _strip_markdown_fences(raw)
    raw = _extract_first_json_object(raw)
    return raw


# ── Schema serialisation ──────────────────────────────────────────────────

# Explicit schema template for Pass 1a text chunk extraction.
# Hardcoded rather than generated via model introspection to ensure
# the model receives a clear, unambiguous structure.
# image_signal is intentionally excluded — it is populated in
# Pass 1b (VisionClient) and must not be inferred from text.
CHUNK_SIGNALS_SCHEMA_HINT = json.dumps({
    "chunk_id":          0,
    "source_page":       "<str>",
    "emotional_impact":  ["<gew_key>", "<gew_key>"],
    "kansei_perception": {"<pair_key>": "<pole_or_neutral>"},
    "visual_setting":    ["<iptc_key>"],
    "target_audience":   ["<sinus_key>"],
    "confidence":        0.0,
}, indent=2)


# ── Ollama client ─────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin client for the Ollama /api/generate endpoint.

    Handles prompt construction, request dispatch, response repair,
    and Pydantic validation for Pass 1a (text chunk analysis).

    Args:
        base_url:    Base URL of the running Ollama instance.
        model:       Ollama model tag to use for text analysis.
        criteria_dir: Path to the directory containing criteria JSON files.
        timeout:     Per-request timeout in seconds.
        max_retries: Number of retry attempts on validation or parse failure.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        criteria_dir: str = "./criteria",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries
        self.loader      = CriteriaLoader(criteria_dir=criteria_dir)

        # Reuse a single httpx client across all requests for connection pooling
        self._http = httpx.Client(timeout=self.timeout)

    # ── Low-level request ─────────────────────────────────────────────────

    def _post(self, prompt: str, system: str) -> str:
        """
        Sends a single generate request to Ollama and returns the raw
        response string.

        Uses stream=false so the full response is returned in one payload,
        avoiding the complexity of reassembling streamed chunks.

        Raises:
            httpx.HTTPError: On network-level failures.
            ValueError: If Ollama returns a non-200 status.
        """
        payload = {
            "model":  self.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                # Lower temperature for more deterministic structured output.
                # Stop tokens are intentionally omitted: Gemma wraps JSON in
                # markdown fences which would trigger a stop before any content
                # is written. Fences are stripped downstream by _repair_json().
                "temperature": 0.2,
            },
        }

        response = self._http.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )

        if response.status_code != 200:
            raise ValueError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )

        data = response.json()
        return data.get("response", "")

    # ── Pass 1a: Text chunk analysis ──────────────────────────────────────

    def _build_extraction_prompt(
        self,
        chunk: dict[str, Any],
        dimensions: list[DimensionKey] | None = None,
    ) -> tuple[str, str]:
        """
        Constructs the system and user prompts for a single chunk analysis.

        The system prompt carries the criteria and task instruction.
        The user prompt carries the chunk content and output schema hint.

        Returns:
            Tuple of (system_prompt, user_prompt).
        """
        system_prompt = self.loader.build_system_prompt(
            dimensions=dimensions,
            task="extraction",
        )

        user_prompt = (
            f"Analyse the following brand text chunk and extract visual signals.\n\n"
            f"CHUNK METADATA:\n"
            f"  chunk_id: {chunk['chunk_id']}\n"
            f"  source_page: {chunk.get('source_page', 'unknown')}\n"
            f"  heading: {chunk.get('heading', '')}\n\n"
            f"CHUNK TEXT:\n{chunk['text']}\n\n"
            f"REQUIRED OUTPUT SCHEMA (return exactly this structure):\n"
            f"{CHUNK_SIGNALS_SCHEMA_HINT}\n\n"
            f"Return only a single valid JSON object. No explanation."
        )

        return system_prompt, user_prompt

    def analyse_chunk(
        self,
        chunk: dict[str, Any],
        dimensions: list[DimensionKey] | None = None,
    ) -> ChunkSignals | None:
        """
        Analyses a single text chunk and returns a validated ChunkSignals instance.

        Retries up to max_retries times on JSON parse or Pydantic validation
        failure. Returns None if all attempts fail, allowing the pipeline to
        continue with partial results rather than halting entirely.

        Args:
            chunk:      A chunk dict from corpus.json with keys:
                        chunk_id, text, heading, source_page (added by caller).
            dimensions: Criteria dimensions to include. Defaults to all four.

        Returns:
            Validated ChunkSignals instance, or None on persistent failure.
        """
        system_prompt, user_prompt = self._build_extraction_prompt(
            chunk, dimensions
        )

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._post(user_prompt, system_prompt)
                repaired = _repair_json(raw)
                data = json.loads(repaired)

                # Ensure provenance fields match the source chunk, not model output
                data["chunk_id"]    = chunk["chunk_id"]
                data["source_page"] = chunk.get("source_page", "unknown")
                # Ensure image_signal is always a valid dict — the model must not
                # populate this field in Pass 1a (it is set by VisionClient in Pass 1b).
                # Overwrite any non-dict value with an empty ImageSignal dict.
                if not isinstance(data.get("image_signal"), dict):
                    data["image_signal"] = {}

                signals = ChunkSignals.model_validate(data)
                logger.debug(
                    "chunk %d validated OK (attempt %d)", chunk["chunk_id"], attempt
                )
                return signals

            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "chunk %d attempt %d/%d failed: %s",
                    chunk["chunk_id"], attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(RETRY_DELAY)

        logger.error(
            "chunk %d failed after %d attempts. Last error: %s",
            chunk["chunk_id"], self.max_retries, last_error,
        )
        return None

    # ── Batch processing ──────────────────────────────────────────────────

    def analyse_corpus_page(
        self,
        page_label: str,
        chunks: list[dict[str, Any]],
        dimensions: list[DimensionKey] | None = None,
        min_char_count: int = 60,
    ) -> tuple[list[ChunkSignals], list[dict[str, Any]]]:
        """
        Analyses all text chunks from a single corpus page.

        Skips chunks below min_char_count to avoid sending navigation
        labels, button text, or other low-signal fragments to the model.

        Args:
            page_label:     The corpus page key (e.g. 'about_us').
            chunks:         List of chunk dicts from corpus.json.
            dimensions:     Criteria dimensions to include.
            min_char_count: Minimum character count to process a chunk.

        Returns:
            Tuple of:
                - List of validated ChunkSignals (processed chunks).
                - List of skip-record dicts documenting every skipped chunk
                  with its reason, heading, text preview, and char count.
        """
        results: list[ChunkSignals] = []
        skipped_records: list[dict[str, Any]] = []

        for chunk in chunks:
            # Inject source_page since corpus.json stores it at page level
            chunk_with_page = {**chunk, "source_page": page_label}
            char_count = chunk.get("char_count", len(chunk.get("text", "")))

            if char_count < min_char_count:
                skipped_records.append({
                    "chunk_id":    chunk["chunk_id"],
                    "source_page": page_label,
                    "skip_reason": "too_short",
                    "char_count":  char_count,
                    "heading":     chunk.get("heading", ""),
                    "text":        chunk.get("text", ""),
                })
                logger.debug(
                    "skipping chunk %d (too short: %d chars)",
                    chunk["chunk_id"], char_count,
                )
                continue

            signals = self.analyse_chunk(chunk_with_page, dimensions)
            if signals is not None:
                results.append(signals)
            else:
                # All retries exhausted — record as failed rather than silently dropping
                skipped_records.append({
                    "chunk_id":    chunk["chunk_id"],
                    "source_page": page_label,
                    "skip_reason": "validation_failed",
                    "char_count":  char_count,
                    "heading":     chunk.get("heading", ""),
                    "text":        chunk.get("text", ""),
                })

        logger.info(
            "page '%s': %d/%d chunks analysed, %d skipped",
            page_label, len(results), len(chunks), len(skipped_records),
        )
        return results, skipped_records

    # ── Health check ──────────────────────────────────────────────────────

    def ping(self) -> bool:
        """
        Checks whether the Ollama instance is reachable and the configured
        model is available.

        Returns True if healthy, False otherwise. Does not raise.
        """
        try:
            response = self._http.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            tags = [m["name"] for m in response.json().get("models", [])]
            available = any(self.model in tag for tag in tags)
            if not available:
                logger.warning(
                    "model '%s' not found in Ollama. Available: %s",
                    self.model, tags,
                )
            return available
        except Exception as exc:
            logger.error("Ollama ping failed: %s", exc)
            return False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._http.close()
