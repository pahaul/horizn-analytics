"""
schemas.py
──────────
Pydantic models for the two-pass brand analysis pipeline.

Pass 1 – ChunkSignals:
    One instance per text chunk from corpus.json.
    Pass 1a (text): OllamaClient populates Kansei, IPTC, Sinus, GEW, confidence.
    Pass 1b (vision): VisionClient populates image_signal with visual analysis
    of brand images. GEW is intentionally excluded from vision analysis —
    advertising images are constructed to be positive by default and emotion
    classification adds no signal.

Pass 2 – BrandProfile:
    One instance per scraped website (i.e. one per pipeline run).
    Aggregated from all ChunkSignals by a second LLM call.
    image_prompt_components is derived exclusively from aggregated ImageSignal
    data — never from text signals. This ensures visual output reflects
    what the brand actually shows, not what it says.

Design principles:
    - Controlled vocabularies at the chunk level minimise hallucination in
      small models. Every categorical field lists its valid values explicitly.
    - List length constraints use Annotated[list[str], Field(max_length=N)],
      the correct Pydantic v2 pattern for sequence validation.
    - Mandatory vs optional ImageSignal fields: fields that are always
      unambiguously answerable from any image are required (no default).
      Fields that may be genuinely ambiguous are optional (default=None or "").
    - GEW emotional_impact is retained on ChunkSignals for text analysis
      but removed from BrandProfile — it is not used downstream.
    - ColorSignal is retained as a lightweight sub-model within ImageSignal
      for the color palette fields, and independently in BrandProfile for
      the aggregated palette output consumed by build_context.py.
"""

from typing import Annotated, Optional
from pydantic import BaseModel, Field


# ── Reusable constrained list types ──────────────────────────────────────

StrList2 = Annotated[list[str], Field(max_length=2)]
StrList3 = Annotated[list[str], Field(max_length=3)]
StrList5 = Annotated[list[str], Field(max_length=5)]
StrList6 = Annotated[list[str], Field(max_length=6)]


# ── ColorSignal ───────────────────────────────────────────────────────────
#
# Retained as a standalone model for two reasons:
#   1. It is embedded inside ImageSignal for the vision pass.
#   2. BrandProfile.color_palette is a ColorSignal (aggregated hex codes only,
#      without the full compositional fields).

class ColorSignal(BaseModel):
    """Dominant colors extracted from a single image."""

    hex_codes: Annotated[list[str], Field(max_length=5)] = Field(
        default=[],
        description=(
            "Dominant brand-relevant colors as hex codes (#RRGGBB). Max 5. "
            "Focus on intentional brand colors, not incidental environmental tones."
        ),
    )
    roles: dict[str, str] = Field(
        default={},
        description=(
            "Maps each hex code to its role. "
            "Valid roles: primary, secondary, accent, background, text, neutral."
        ),
    )


# ── ImageSignal ───────────────────────────────────────────────────────────
#
# Replaces the old ColorSignal as the output of Pass 1b (vision analysis).
# Captures compositional, photographic, and content-level signals from
# brand images using controlled vocabularies.
#
# Field split:
#   MANDATORY — always classifiable from any brand image, no ambiguity:
#     shot_size, camera_angle, depth_of_field, people_presence,
#     lighting, setting, color_palette
#
#   OPTIONAL — may be genuinely unclear or inapplicable:
#     composition  (multiple principles can co-exist; leave empty if unclear)
#     product_focus (leave empty if no product is identifiable)
#     narrative_style (leave empty for highly abstract imagery)

class ImageSignal(BaseModel):
    """
    Visual analysis of a single brand image (Pass 1b output).

    All mandatory fields must be populated. Optional fields may be left
    empty when the image does not provide a clear signal for that dimension.
    """

    # -- Color ---------------------------------------------------------------

    color_palette: ColorSignal = Field(
        default_factory=ColorSignal,
        description=(
            "Dominant colors extracted from this image. "
            "Max 5 hex codes with roles."
        ),
    )

    # -- Shot & camera (MANDATORY) -------------------------------------------

    shot_size: str = Field(
        default="",
        description=(
            "Distance between camera and primary subject. "
            "One of: extreme_close_up, close_up, medium_close_up, medium, "
            "medium_wide, wide, extreme_wide."
        ),
    )
    camera_angle: str = Field(
        default="",
        description=(
            "Vertical angle of the camera relative to the subject. "
            "One of: eye_level, high_angle, low_angle, dutch_tilt, overhead."
        ),
    )
    depth_of_field: str = Field(
        default="",
        description=(
            "Apparent depth of field in the image. "
            "One of: shallow (background blurred), moderate, deep (all in focus)."
        ),
    )

    # -- Lighting (MANDATORY) ------------------------------------------------

    lighting: str = Field(
        default="",
        description=(
            "Dominant lighting character. "
            "One of: natural_soft, natural_harsh, studio_soft, studio_dramatic, "
            "backlit, low_key, high_key."
        ),
    )

    # -- Content (MANDATORY) -------------------------------------------------

    people_presence: str = Field(
        default="",
        description=(
            "Whether and how people appear in the image. "
            "One of: none, hands_only, partial_body, full_body, "
            "face_close_up, group."
        ),
    )
    setting: str = Field(
        default="",
        description=(
            "Physical environment of the image. "
            "Use IPTC setting keys: indoor_retail, outdoor_travel, "
            "indoor_workspace, outdoor_urban, indoor_home."
        ),
    )

    # -- Composition (OPTIONAL) ----------------------------------------------

    composition: Optional[str] = Field(
        default=None,
        description=(
            "Dominant compositional principle, if clearly identifiable. "
            "One of: rule_of_thirds, centered, symmetrical, leading_lines, "
            "negative_space, frame_within_frame. "
            "Leave empty if no single principle dominates."
        ),
    )

    # -- Product & narrative (OPTIONAL) --------------------------------------

    product_focus: Optional[str] = Field(
        default=None,
        description=(
            "Relationship between product and image content. "
            "One of: product_only, product_in_use, lifestyle_with_product, "
            "lifestyle_without_product. "
            "Leave empty if no product is identifiable."
        ),
    )
    narrative_style: Optional[str] = Field(
        default=None,
        description=(
            "Visual storytelling approach. "
            "One of: documentary, staged, abstract, flat_lay. "
            "Leave empty for highly ambiguous imagery."
        ),
    )


# ── Pass 1: Chunk-level signal extraction ────────────────────────────────

class ChunkSignals(BaseModel):
    """
    Brand signals extracted from a single text chunk (Pass 1a output),
    optionally enriched with image signals from Pass 1b.

    One instance is produced per chunk in corpus.json. The full collection
    of ChunkSignals for a website is passed as input to Pass 2.
    """

    # -- Provenance ----------------------------------------------------------

    chunk_id: int = Field(
        description="Chunk index as defined in corpus.json."
    )
    source_page: str = Field(
        description="Page label from corpus.json (e.g. 'about_us', 'homepage')."
    )

    # -- GEW: Emotional impact (text analysis only) --------------------------
    # Retained for text-level brand tone analysis.
    # Not aggregated into BrandProfile — advertising copy emotional framing
    # is always positive by construction and adds no discriminating signal.

    emotional_impact: StrList2 = Field(
        default=[],
        description=(
            "Max 2 GEW emotion family names describing the intended emotional "
            "response to this text content. Use exact keys from criteria_gew.json."
        ),
    )

    # -- Kansei: Product perception ------------------------------------------

    kansei_perception: dict[str, str] = Field(
        default={},
        description=(
            "Kansei bipolar assessment. Keys are pair identifiers from "
            "criteria_kansei.json (e.g. 'modern_traditional'), values are the "
            "dominant pole label (e.g. 'modern') or 'neutral'. Max 4 pairs."
        ),
    )

    # -- IPTC: Visual setting ------------------------------------------------

    visual_setting: StrList2 = Field(
        default=[],
        description=(
            "Max 2 IPTC setting keys describing the physical environment "
            "implied or depicted. Use exact keys from criteria_setting_iptc.json."
        ),
    )

    # -- Sinus: Target audience ----------------------------------------------

    target_audience: StrList2 = Field(
        default=[],
        description=(
            "Max 2 Sinus milieu keys describing the implied target audience. "
            "Use exact keys from criteria_audience_sinus.json."
        ),
    )

    # -- Image signal --------------------------------------------------------
    # Populated by Pass 1b (VisionClient). Empty for text-only chunks.

    image_signal: ImageSignal = Field(
        default_factory=ImageSignal,
        description=(
            "Visual analysis of the brand image associated with this chunk. "
            "Populated by Pass 1b. Empty for pure text chunks."
        ),
    )

    # -- Quality signal ------------------------------------------------------

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Relevance score for visual brand analysis. "
            "0.0 = no useful signal (e.g. shipping info, cookie notice). "
            "1.0 = highly relevant (e.g. brand manifesto, visual identity description). "
            "Used by Pass 2 to weight contributions across chunks."
        ),
    )


# ── Pass 2: Aggregated brand profile ─────────────────────────────────────

class ImagePromptComponents(BaseModel):
    """
    Structured building blocks for constructing image generation prompts.

    Derived by Pass 2 exclusively from aggregated ImageSignal data.
    Fields are directly composable into a prompt string for image generation
    models such as Stable Diffusion, Midjourney, or FLUX.
    """

    subject: str = Field(
        default="",
        description=(
            "The primary subject of brand imagery, grounded in people_presence "
            "and product_focus signals from the vision analysis. "
            "Example: 'young professional carrying a minimal backpack in motion'"
        ),
    )
    environment: str = Field(
        default="",
        description=(
            "The dominant setting, derived from aggregated IPTC setting signals. "
            "Example: 'urban street, morning light, European city'"
        ),
    )
    lighting: str = Field(
        default="",
        description=(
            "Lighting character, derived from aggregated lighting signals. "
            "Example: 'soft natural daylight, slightly overcast, no harsh shadows'"
        ),
    )
    composition: str = Field(
        default="",
        description=(
            "Compositional and camera style, derived from shot_size, "
            "camera_angle, depth_of_field, and composition signals. "
            "Example: 'eye-level medium shot, shallow depth of field, rule of thirds'"
        ),
    )
    color_palette: str = Field(
        default="",
        description=(
            "Color description for the image, derived from the aggregated "
            "color palette hex codes and roles. "
            "Example: 'muted steel blue and off-white, minimal contrast, clean background'"
        ),
    )
    mood_keywords: StrList5 = Field(
        default=[],
        description=(
            "Up to 5 concise mood or style descriptors, derived from "
            "narrative_style and Kansei signals. "
            "Example: ['documentary', 'understated', 'purposeful']"
        ),
    )
    negative_prompt: StrList5 = Field(
        default=[],
        description=(
            "Visual elements to exclude, derived from brand positioning. "
            "Example: ['stock photo aesthetic', 'oversaturated', 'loud branding']"
        ),
    )


class BrandProfile(BaseModel):
    """
    Unified visual brand profile for a single website (Pass 2 output).

    Aggregated from all ChunkSignals produced in Pass 1. Stored as
    brand_profile.json and consumed by the chat interface and image
    prompt builder.
    """

    # -- Metadata ------------------------------------------------------------

    source_url: str = Field(
        description="The root URL of the scraped website."
    )
    generated_at: str = Field(
        description="ISO 8601 timestamp of profile generation."
    )
    chunks_analysed: int = Field(
        description="Total number of ChunkSignals used as input for this profile."
    )
    high_confidence_chunks: int = Field(
        description="Number of chunks with confidence >= 0.6."
    )

    # -- Aggregated signals --------------------------------------------------

    kansei_profile: dict[str, str] = Field(
        default={},
        description=(
            "Consolidated Kansei assessment across all chunks. "
            "Dominant pole per pair, weighted by chunk confidence."
        ),
    )
    primary_settings: StrList3 = Field(
        default=[],
        description=(
            "Up to 3 most frequently occurring IPTC setting keys, "
            "aggregated from both text chunks and image signals."
        ),
    )
    primary_audience: StrList2 = Field(
        default=[],
        description="Up to 2 most strongly signalled Sinus milieu keys.",
    )
    color_palette: ColorSignal = Field(
        default_factory=ColorSignal,
        description="Consolidated color palette from all vision-analysed images.",
    )

    # -- Visual signal summary -----------------------------------------------
    # Aggregated from ImageSignal fields across all vision-analysed chunks.
    # These are frequency distributions, not single values, to preserve
    # the range of visual styles the brand uses.

    dominant_shot_sizes: StrList3 = Field(
        default=[],
        description="Up to 3 most common shot sizes across brand images.",
    )
    dominant_lighting: StrList2 = Field(
        default=[],
        description="Up to 2 most common lighting styles across brand images.",
    )
    dominant_people_presence: StrList2 = Field(
        default=[],
        description="Up to 2 most common people presence categories across brand images.",
    )
    dominant_narrative_styles: StrList2 = Field(
        default=[],
        description="Up to 2 most common narrative styles (optional field — may be empty).",
    )

    # -- Brand voice summary -------------------------------------------------

    brand_adjectives: StrList6 = Field(
        default=[],
        description=(
            "Up to 6 adjectives the brand consistently uses to describe itself "
            "or its products. Extracted directly from copy, not inferred."
        ),
    )
    brand_values: StrList5 = Field(
        default=[],
        description=(
            "Up to 5 explicit brand values or commitments. "
            "Grounded in copy evidence."
        ),
    )

    # -- Image generation output ---------------------------------------------

    image_prompt_components: ImagePromptComponents = Field(
        default_factory=ImagePromptComponents,
        description=(
            "Structured components for building image generation prompts. "
            "Derived exclusively from aggregated ImageSignal data. "
            "The primary consumer-facing output of the pipeline."
        ),
    )

    # -- Confidence summary --------------------------------------------------

    profile_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Overall confidence in this profile. "
            "Derived from the mean confidence of contributing chunks."
        ),
    )
    low_signal_warnings: list[str] = Field(
        default=[],
        description="Human-readable warnings about gaps or weaknesses in the analysis.",
    )

    # -- Diagnostic metadata -------------------------------------------------

    kansei_raw_votes: dict[str, dict[str, float]] = Field(
        default={},
        description=(
            "Raw confidence-weighted vote tallies per Kansei pair. "
            "Included for auditability."
        ),
    )
