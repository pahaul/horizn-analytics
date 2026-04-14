"""
aggregator.py
─────────────
Pass 2 of the brand analysis pipeline.

Takes the full collection of ChunkSignals produced in Pass 1 and
synthesises them into a single BrandProfile instance.

Two-step process:

    Step 1 – Statistical pre-aggregation (Python):
        Deterministic frequency counting and confidence weighting.
        Handles: Kansei, IPTC settings, Sinus audience, color palette,
        and all ImageSignal fields (shot_size, camera_angle, lighting,
        people_presence, composition, product_focus, narrative_style).
        GEW emotional_impact is intentionally excluded — advertising copy
        is positive by construction and adds no discriminating signal.

    Step 2 – LLM synthesis call:
        Pre-aggregated signals are passed to the model. The model is asked
        to produce brand_adjectives, brand_values, and image_prompt_components.
        image_prompt_components are derived exclusively from the pre-aggregated
        visual signals — the model is explicitly instructed not to invent
        visual language from text signals.

Separation of concerns:
    Python handles counting and weighting (reliable).
    The LLM handles language, interpretation, and articulation (what it is good at).
"""

import json
import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import ValidationError

from schemas import (
    BrandProfile, ChunkSignals, ColorSignal, ImagePromptComponents, ImageSignal
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL           = "http://localhost:11434"
DEFAULT_MODEL             = "gemma3:4b"
DEFAULT_TIMEOUT           = 90.0
MAX_RETRIES               = 3
RETRY_DELAY               = 2.0
HIGH_CONFIDENCE_THRESHOLD = 0.6


# ── Step 1: Statistical pre-aggregation ──────────────────────────────────

def _weighted_frequency(
    values: list[str],
    weights: list[float],
) -> list[tuple[str, float]]:
    """
    Returns values sorted by confidence-weighted frequency, descending.

    Each occurrence contributes its chunk confidence score rather than 1,
    so a value in a high-confidence chunk outweighs the same value in
    multiple low-confidence chunks.
    """
    scores: dict[str, float] = {}
    for value, weight in zip(values, weights):
        scores[value] = scores.get(value, 0.0) + weight
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _aggregate_kansei(
    chunks: list[ChunkSignals],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """
    Consolidates Kansei bipolar assessments across all chunks.

    For each pair key, tallies confidence-weighted votes per pole.
    The winning pole must exceed a minimum score of 0.3 to avoid
    labelling weakly-signalled pairs as non-neutral.

    Returns:
        Tuple of (consolidated profile, raw vote tallies for auditability).
    """
    pair_scores: dict[str, dict[str, float]] = {}
    raw_votes_full: dict[str, dict[str, float]] = {}

    for chunk in chunks:
        for pair_key, pole in chunk.kansei_perception.items():
            if pair_key not in raw_votes_full:
                raw_votes_full[pair_key] = {}
            raw_votes_full[pair_key][pole] = (
                raw_votes_full[pair_key].get(pole, 0.0) + chunk.confidence
            )
            # Neutral is an abstention, not a vote for a pole.
            if pole != "neutral":
                if pair_key not in pair_scores:
                    pair_scores[pair_key] = {}
                pair_scores[pair_key][pole] = (
                    pair_scores[pair_key].get(pole, 0.0) + chunk.confidence
                )

    consolidated: dict[str, str] = {}
    for pair_key, pole_scores in pair_scores.items():
        if not pole_scores:
            consolidated[pair_key] = "neutral"
            continue
        top_pole = max(pole_scores, key=lambda p: pole_scores[p])
        consolidated[pair_key] = (
            top_pole if pole_scores[top_pole] >= 0.3 else "neutral"
        )

    raw_votes = {
        pair_key: {pole: round(score, 3) for pole, score in poles.items()}
        for pair_key, poles in raw_votes_full.items()
    }
    return consolidated, raw_votes


def _normalise_color_role(role: str) -> str:
    """
    Normalises a color role string to a single valid value.

    Handles pipe-separated multi-role strings (e.g. "primary | accent")
    by taking the first token. Unknown values fall back to "neutral".
    """
    VALID_ROLES = {"primary", "secondary", "accent", "background", "text", "neutral"}
    first = role.split("|")[0].strip().lower()
    return first if first in VALID_ROLES else "neutral"


def _aggregate_colors(chunks: list[ChunkSignals]) -> ColorSignal:
    """
    Merges color palettes from all image signals into a single ColorSignal.

    For each hex code, the role assigned by the highest-confidence chunk
    is used. Top 5 hex codes by confidence are retained.
    Returns an empty ColorSignal if no vision analysis was run.
    """
    hex_with_confidence: dict[str, float] = {}
    role_by_hex: dict[str, tuple[str, float]] = {}

    for chunk in chunks:
        palette = chunk.image_signal.color_palette
        for hex_code in palette.hex_codes:
            current = hex_with_confidence.get(hex_code, 0.0)
            if chunk.confidence > current:
                hex_with_confidence[hex_code] = chunk.confidence

        for hex_code, role in palette.roles.items():
            _, existing_conf = role_by_hex.get(hex_code, ("", 0.0))
            if chunk.confidence > existing_conf:
                role_by_hex[hex_code] = (_normalise_color_role(role), chunk.confidence)

    sorted_hex = sorted(
        hex_with_confidence.items(), key=lambda x: x[1], reverse=True
    )[:5]
    top_hex = [h for h, _ in sorted_hex]
    roles   = {h: role_by_hex[h][0] for h in top_hex if h in role_by_hex}

    return ColorSignal(hex_codes=top_hex, roles=roles)


def _aggregate_image_signals(
    chunks: list[ChunkSignals],
) -> dict[str, Any]:
    """
    Aggregates all ImageSignal fields across vision-analysed chunks.

    Uses confidence-weighted frequency for mandatory fields.
    Optional fields (composition, product_focus, narrative_style) are
    aggregated only from chunks where the field was populated.

    Returns a plain dict ready to be included in the pre-aggregated
    signals passed to the LLM synthesis call.
    """
    # Only include chunks that actually have image signal data
    vision_chunks = [c for c in chunks if c.image_signal.shot_size]

    if not vision_chunks:
        return {}

    def top_values(field: str, n: int) -> list[str]:
        """Confidence-weighted top-n values for a mandatory ImageSignal field."""
        values  = [getattr(c.image_signal, field) for c in vision_chunks
                   if getattr(c.image_signal, field)]
        weights = [c.confidence for c in vision_chunks
                   if getattr(c.image_signal, field)]
        return [v for v, _ in _weighted_frequency(values, weights)[:n]]

    def top_optional(field: str, n: int) -> list[str]:
        """Confidence-weighted top-n values for an optional ImageSignal field."""
        values  = [getattr(c.image_signal, field) for c in vision_chunks
                   if getattr(c.image_signal, field) is not None]
        weights = [c.confidence for c in vision_chunks
                   if getattr(c.image_signal, field) is not None]
        return [v for v, _ in _weighted_frequency(values, weights)[:n]]

    return {
        "images_analysed":          len(vision_chunks),
        "dominant_shot_sizes":      top_values("shot_size", 3),
        "dominant_camera_angles":   top_values("camera_angle", 2),
        "dominant_depth_of_field":  top_values("depth_of_field", 1),
        "dominant_lighting":        top_values("lighting", 2),
        "dominant_people_presence": top_values("people_presence", 2),
        "dominant_settings":        top_values("setting", 3),
        "dominant_compositions":    top_optional("composition", 2),
        "dominant_product_focus":   top_optional("product_focus", 2),
        "dominant_narrative_styles": top_optional("narrative_style", 2),
    }


def pre_aggregate(chunks: list[ChunkSignals]) -> dict[str, Any]:
    """
    Performs deterministic statistical pre-aggregation on ChunkSignals.

    Returns a plain dict containing aggregated signals from both text
    analysis (Pass 1a) and vision analysis (Pass 1b). This dict is
    serialised and passed to the LLM synthesis call.
    """
    if not chunks:
        return {}

    high_conf    = [c for c in chunks if c.confidence >= HIGH_CONFIDENCE_THRESHOLD]
    working_set  = high_conf if high_conf else chunks

    # -- Visual setting (IPTC) — from text analysis -------------------------
    setting_values  = [s for c in working_set for s in c.visual_setting]
    setting_weights = [c.confidence for c in working_set for _ in c.visual_setting]
    primary_settings = [
        v for v, _ in _weighted_frequency(setting_values, setting_weights)[:3]
    ]

    # -- Target audience (Sinus) --------------------------------------------
    audience_values  = [a for c in working_set for a in c.target_audience]
    audience_weights = [c.confidence for c in working_set for _ in c.target_audience]
    primary_audience = [
        v for v, _ in _weighted_frequency(audience_values, audience_weights)[:2]
    ]

    # -- Kansei profile ------------------------------------------------------
    kansei_profile, kansei_raw_votes = _aggregate_kansei(working_set)

    # -- Color palette (from image signals) ----------------------------------
    color_palette = _aggregate_colors(chunks)

    # -- Image signal aggregation (from vision analysis) --------------------
    image_signals = _aggregate_image_signals(chunks)

    # -- Merge settings: prefer image-derived settings if available ---------
    # Image settings are grounded in what the brand actually shows.
    # Text settings reflect what the brand writes about, which may differ.
    if image_signals.get("dominant_settings"):
        combined_settings = image_signals["dominant_settings"]
    else:
        combined_settings = primary_settings

    # -- Profile confidence --------------------------------------------------
    mean_confidence = sum(c.confidence for c in chunks) / len(chunks)

    # -- Low-signal warnings -------------------------------------------------
    warnings: list[str] = []
    if not color_palette.hex_codes:
        warnings.append(
            "color palette underdetermined — no vision analysis results found"
        )
    if not image_signals:
        warnings.append(
            "no image signals available — image_prompt_components will be empty"
        )
    if not primary_audience:
        warnings.append(
            "target audience could not be determined from available content"
        )
    if len(high_conf) < 3:
        warnings.append(
            f"only {len(high_conf)} high-confidence chunks available — "
            "profile reliability may be low"
        )

    return {
        "kansei_profile":            kansei_profile,
        "primary_settings":          combined_settings,
        "primary_audience":          primary_audience,
        "color_palette":             color_palette.model_dump(),
        "image_signals":             image_signals,
        "mean_confidence":           round(mean_confidence, 3),
        "high_confidence_chunks":    len(high_conf),
        "total_chunks":              len(chunks),
        "low_signal_warnings":       warnings,
        "kansei_raw_votes":          kansei_raw_votes,
    }


# ── Step 2: LLM synthesis call ────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """You are a brand visual strategist.

You will receive a JSON summary of aggregated brand signals extracted from
a company website. Your task is to synthesise these signals into three outputs:

1. brand_adjectives: up to 6 adjectives the brand uses to describe itself.
   Extract from brand copy — do not invent.

2. brand_values: up to 5 explicit values or commitments from brand copy.
   Must be grounded in evidence — do not invent.

3. image_prompt_components: structured components for AI image generation.
   IMPORTANT: derive these EXCLUSIVELY from the image_signals section of the
   input. Do not invent visual language from text signals.
   If image_signals is empty or missing, return empty strings for all fields.

   Fields to populate:
   - subject: primary subject, grounded in people_presence + product_focus signals
   - environment: setting, grounded in dominant_settings signals
   - lighting: grounded in dominant_lighting signals
   - composition: grounded in dominant_shot_sizes + dominant_camera_angles +
     dominant_depth_of_field + dominant_compositions signals
   - color_palette: translate hex codes and roles into descriptive language
   - mood_keywords: up to 5, grounded in dominant_narrative_styles + Kansei profile
   - negative_prompt: up to 5 visual elements to avoid, from brand positioning

Respond ONLY as a valid JSON object with exactly these keys:
{
  "brand_adjectives": [...],
  "brand_values": [...],
  "image_prompt_components": {
    "subject": "...",
    "environment": "...",
    "lighting": "...",
    "composition": "...",
    "color_palette": "...",
    "mood_keywords": [...],
    "negative_prompt": [...]
  }
}

No explanation. No markdown. No preamble."""


def _build_synthesis_prompt(
    pre_agg: dict[str, Any],
    source_url: str,
) -> str:
    """Builds the user prompt for the Pass 2 synthesis call."""
    signals_json = json.dumps(pre_agg, indent=2)
    return (
        f"Website: {source_url}\n\n"
        f"Aggregated brand signals:\n{signals_json}\n\n"
        f"Synthesise brand_adjectives, brand_values, and "
        f"image_prompt_components. For image_prompt_components, "
        f"use only the image_signals section."
    )


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_first_json_object(text: str) -> str:
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
    return text


# ── Aggregator ────────────────────────────────────────────────────────────

class Aggregator:
    """
    Synthesises a BrandProfile from a collection of ChunkSignals.

    Combines deterministic Python pre-aggregation (Step 1) with a
    focused LLM synthesis call (Step 2).
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        self.base_url    = base_url.rstrip("/")
        self.model       = model
        self.timeout     = timeout
        self.max_retries = max_retries
        self._http       = httpx.Client(timeout=self.timeout)

    def _post(self, prompt: str) -> str:
        """Sends a synthesis request to Ollama and returns the raw response."""
        payload = {
            "model":   self.model,
            "system":  SYNTHESIS_SYSTEM_PROMPT,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": 0.3},
        }
        response = self._http.post(f"{self.base_url}/api/generate", json=payload)
        if response.status_code != 200:
            raise ValueError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:200]}"
            )
        return response.json().get("response", "")

    def synthesise(
        self,
        chunks: list[ChunkSignals],
        source_url: str,
    ) -> BrandProfile:
        """
        Produces a BrandProfile from the full set of ChunkSignals.

        Runs pre-aggregation deterministically, then calls the LLM for
        synthesis. If all retries fail, returns a BrandProfile populated
        with pre-aggregated data only, leaving LLM fields empty.
        """
        pre_agg = pre_aggregate(chunks)
        prompt  = _build_synthesis_prompt(pre_agg, source_url)

        llm_data: dict[str, Any] = {}
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw      = self._post(prompt)
                repaired = _strip_markdown_fences(raw)
                repaired = _extract_first_json_object(repaired)
                llm_data = json.loads(repaired)
                logger.debug("synthesis call validated OK (attempt %d)", attempt)
                break
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "synthesis attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(RETRY_DELAY)
        else:
            logger.error(
                "synthesis failed after %d attempts: %s",
                self.max_retries, last_error,
            )
            pre_agg.setdefault("low_signal_warnings", []).append(
                "LLM synthesis failed — brand_adjectives, brand_values, "
                "and image_prompt_components are empty"
            )

        # -- image_prompt_components -----------------------------------------
        image_components_data = llm_data.get("image_prompt_components", {})
        try:
            image_components = ImagePromptComponents.model_validate(
                image_components_data
            )
        except ValidationError:
            image_components = ImagePromptComponents()

        # -- Visual signal summaries from pre-aggregation --------------------
        img = pre_agg.get("image_signals", {})

        profile = BrandProfile(
            source_url               = source_url,
            generated_at             = datetime.now(timezone.utc).isoformat(),
            chunks_analysed          = pre_agg.get("total_chunks", len(chunks)),
            high_confidence_chunks   = pre_agg.get("high_confidence_chunks", 0),
            kansei_profile           = pre_agg.get("kansei_profile", {}),
            primary_settings         = pre_agg.get("primary_settings", []),
            primary_audience         = pre_agg.get("primary_audience", []),
            color_palette            = ColorSignal.model_validate(
                                           pre_agg.get("color_palette", {})
                                       ),
            dominant_shot_sizes      = img.get("dominant_shot_sizes", []),
            dominant_lighting        = img.get("dominant_lighting", []),
            dominant_people_presence = img.get("dominant_people_presence", []),
            dominant_narrative_styles = img.get("dominant_narrative_styles", []),
            brand_adjectives         = llm_data.get("brand_adjectives", []),
            brand_values             = llm_data.get("brand_values", []),
            image_prompt_components  = image_components,
            profile_confidence       = pre_agg.get("mean_confidence", 0.0),
            low_signal_warnings      = pre_agg.get("low_signal_warnings", []),
            kansei_raw_votes         = pre_agg.get("kansei_raw_votes", {}),
        )

        logger.info(
            "brand profile built: %d chunks, confidence %.2f, "
            "%d vision chunks, %d warnings",
            profile.chunks_analysed,
            profile.profile_confidence,
            img.get("images_analysed", 0),
            len(profile.low_signal_warnings),
        )
        return profile

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._http.close()
