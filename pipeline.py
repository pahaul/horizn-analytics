"""
pipeline.py
───────────
Top-level orchestrator for the Horizn Analytics brand analysis pipeline.

This module is the single entry point that connects all pipeline stages.
Given a corpus.json file produced by the Phase 1 scraper, it runs:

    Pass 1a — Text analysis:
        Each text chunk is sent to Gemma 4B via OllamaClient.
        Output: one ChunkSignals instance per chunk.

    Pass 1b — Vision analysis (optional):
        Each relevant image URL is sent to Gemma vision.
        Output: ColorSignal data merged into the ChunkSignals collection.
        Skipped if no vision model is available or images are absent.

    Pass 2 — Synthesis:
        All ChunkSignals are aggregated into a single BrandProfile
        via the Aggregator (statistical pre-aggregation + LLM synthesis).
        Output: brand_profile.json

Designed to be called:
    - Directly from a Jupyter notebook (run_pipeline())
    - From the FastAPI endpoint (Phase 3)
    - From the CLI for local testing

Usage (notebook / CLI):
    from pipeline import run_pipeline

    profile = run_pipeline(
        corpus_path="./output/corpus.json",
        output_path="./output/brand_profile.json",
        source_url="https://horizn-studios.com",
    )
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from aggregator import Aggregator
from ollama_client import OllamaClient
from schemas import BrandProfile, ChunkSignals

# Configure a simple console logger for pipeline progress.
# FastAPI and notebook callers can reconfigure this as needed.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Model configuration ──────────────────────────────────────────────────
#
# TEXT_MODEL  — used for Pass 1a (chunk analysis) and Pass 2 (synthesis).
#               Gemma 3 4B is fast and reliable for structured JSON output.
# VISION_MODEL — used for Pass 1b (image analysis) only.
#               Use a model with strong colour and composition perception.
# Both must be available in Ollama (check with: ollama list).
TEXT_MODEL   = "gemma4:31b-cloud"
VISION_MODEL = "gemma4:31b-cloud"  # Use a vision-capable model for Pass 1b

# Convenience alias — used for the output filename.
DEFAULT_MODEL = TEXT_MODEL


# ── Corpus loading ────────────────────────────────────────────────────────

def load_corpus(corpus_path: str | Path) -> dict[str, Any]:
    """
    Loads and validates the structure of corpus.json.

    Expects the format produced by the Phase 1 scraper:
        {
            "page_label": {
                "meta": {...},
                "chunks": [{"chunk_id": 0, "text": "...", ...}],
                "images": [...]
            }
        }

    Raises:
        FileNotFoundError: If the corpus file does not exist.
        ValueError: If the file is not valid JSON or missing expected keys.
    """
    path = Path(corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            corpus = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"corpus.json is not valid JSON: {exc}") from exc

    if not isinstance(corpus, dict):
        raise ValueError("corpus.json must be a JSON object at the top level.")

    logger.info(
        "corpus loaded: %d pages — %s",
        len(corpus),
        list(corpus.keys()),
    )
    return corpus


def _collect_chunks(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flattens all chunks from all pages into a single list.

    Injects the page label as 'source_page' into each chunk dict,
    since corpus.json stores it at the page level rather than per-chunk.

    Returns a flat list of chunk dicts ready for OllamaClient.
    """
    all_chunks: list[dict[str, Any]] = []
    for page_label, page_data in corpus.items():
        for chunk in page_data.get("chunks", []):
            all_chunks.append({**chunk, "source_page": page_label})
    return all_chunks


def _collect_images(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flattens all image entries from all pages into a single list.

    Injects source_page into each image dict so downstream sampling
    in VisionClient can group and cap by page. No filtering is applied
    here — filter_relevant_images() in Phase 1 already removed junk.
    """
    all_images: list[dict[str, Any]] = []
    for page_label, page_data in corpus.items():
        for image in page_data.get("images", []):
            all_images.append({**image, "source_page": page_label})
    return all_images


# ── Pass 1a: Text analysis ────────────────────────────────────────────────

def run_text_analysis(
    corpus: dict[str, Any],
    client: OllamaClient,
) -> tuple[list[ChunkSignals], list[dict[str, Any]]]:
    """
    Runs Pass 1a: sends all text chunks to Gemma and collects ChunkSignals.

    Processes pages in corpus order. Each page's chunks are sent
    individually; results are accumulated into flat lists.

    Returns:
        Tuple of:
            - All successfully validated ChunkSignals.
            - All skip records (too_short or validation_failed) across
              all pages, for inclusion in the decision log.
    """
    all_signals: list[ChunkSignals] = []
    all_skipped: list[dict[str, Any]] = []

    for page_label, page_data in corpus.items():
        chunks = page_data.get("chunks", [])
        if not chunks:
            logger.info("page '%s': no chunks, skipping", page_label)
            continue

        logger.info(
            "pass 1a — analysing page '%s' (%d chunks)", page_label, len(chunks)
        )
        page_signals, page_skipped = client.analyse_corpus_page(
            page_label=page_label,
            chunks=chunks,
        )
        all_signals.extend(page_signals)
        all_skipped.extend(page_skipped)

    logger.info(
        "pass 1a complete: %d ChunkSignals collected, %d skipped across %d pages",
        len(all_signals),
        len(all_skipped),
        len(corpus),
    )
    return all_signals, all_skipped


# ── Pass 1b: Vision analysis ──────────────────────────────────────────────

def run_vision_analysis(
    corpus: dict[str, Any],
    signals: list[ChunkSignals],
    ollama_base_url: str = "http://localhost:11434",
    model: str = VISION_MODEL,
    max_images: int = 20,
) -> tuple[list[ChunkSignals], list]:
    """
    Runs Pass 1b: enriches ChunkSignals with color data from image analysis.

    For each image that survived Phase 1 filtering:

        1. Downloads the image and base64-encodes it.
        2. Sends it to the Gemma vision endpoint via VisionClient.
        3. Parses the response into a ColorSignal instance.
        4. Merges ColorSignals into the existing ChunkSignals collection,
           grouped by source_page, via merge_color_signals_into_chunks().

    VisionClient handles deduplication and per-page sampling internally.
    If no images are present, or if VisionClient fails on all images,
    the signals list is returned unchanged and the aggregator emits a
    low_signal_warning in the BrandProfile.

    Args:
        corpus:          Loaded corpus.json dict.
        signals:         ChunkSignals from Pass 1a; mutated in-place with
                         color data and returned.
        ollama_base_url: Base URL of the Ollama instance.
        model:           Ollama model tag (must support vision).
        max_images:      Cap on the number of images to process.

    Returns:
        Tuple of:
            - The same ChunkSignals list, now with color fields populated
              where vision analysis succeeded.
            - List of (source_page, image_url, ColorSignal) triples from
              VisionClient, for inclusion in the decision log.
    """
    from vision_client import VisionClient, merge_image_signals_into_chunks

    images = _collect_images(corpus)

    if not images:
        logger.info("pass 1b — no images in corpus, skipping.")
        return signals, []

    with VisionClient(
        base_url=ollama_base_url,
        model=model,
        max_images=max_images,
    ) as vision_client:
        image_results = vision_client.analyse_images(images)

    signals = merge_image_signals_into_chunks(signals, image_results)
    return signals, image_results


# ── Pass 2: Synthesis ─────────────────────────────────────────────────────

def run_synthesis(
    signals: list[ChunkSignals],
    source_url: str,
    aggregator: Aggregator,
) -> BrandProfile:
    """
    Runs Pass 2: synthesises all ChunkSignals into a BrandProfile.

    Delegates to Aggregator.synthesise(), which handles both the
    statistical pre-aggregation and the LLM synthesis call.
    """
    logger.info(
        "pass 2 — synthesising %d signals into brand profile", len(signals)
    )
    profile = aggregator.synthesise(signals, source_url)
    logger.info("pass 2 complete — profile confidence: %.2f", profile.profile_confidence)
    return profile


# ── Output ────────────────────────────────────────────────────────────────

def _berlin_timestamp() -> str:
    """Returns the current time as an ISO 8601 string in Europe/Berlin timezone."""
    return datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y%m%dT%H%M%S")


def save_profile(
    profile: BrandProfile,
    output_path: str | Path,
    model: str = DEFAULT_MODEL,
    signals: list | None = None,
    skipped: list | None = None,
    vision_results: list | None = None,
) -> tuple[Path, Path | None]:
    """
    Serialises the BrandProfile and a decision log to disk.

    Filenames are timestamped in Europe/Berlin time to avoid overwriting
    previous runs. Both files are written to the same directory.

    Args:
        profile:        The BrandProfile to serialise.
        output_path:    Base output path (e.g. ./output/brand_profile.json).
                        The timestamp is inserted before the .json extension.
        signals:        ChunkSignals from Pass 1a. Each entry is logged with
                        its full text, heading, and all LLM decisions so that
                        every choice can be audited directly.
        skipped:        Skip records from analyse_corpus_page. Logged with
                        skip reason (too_short / validation_failed), heading,
                        and full text so skipped chunks can be reviewed.
        vision_results: List of (source_page, image_url, ImageSignal) triples
                        from VisionClient. Logged as a dedicated "vision_analysis"
                        section for audit and review.

    Returns:
        Tuple of (profile_path, log_path). log_path is None if neither
        signals nor skipped records were provided.
    """
    base = Path(output_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    ts         = _berlin_timestamp()
    model_slug = model.replace(":", "-").replace("/", "-")

    # -- Brand profile -------------------------------------------------------
    profile_path = base.parent / f"{base.stem}_{model_slug}_{ts}{base.suffix}"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2, ensure_ascii=False)
    logger.info("brand profile saved → %s", profile_path)

    # -- Decision log --------------------------------------------------------
    log_path: Path | None = None
    if signals or skipped:
        log_path = base.parent / f"decision_log_{ts}.json"

        processed_entries = []
        for s in (signals or []):
            processed_entries.append({
                "status":            "processed",
                "chunk_id":          s.chunk_id,
                "source_page":       s.source_page,
                "confidence":        s.confidence,
                "heading":           "",   # not stored on ChunkSignals; see corpus
                "emotional_impact":  s.emotional_impact,
                "kansei_perception": s.kansei_perception,
                "visual_setting":    s.visual_setting,
                "target_audience":   s.target_audience,
                "color_hex_codes":   s.image_signal.color_palette.hex_codes,
            })

        skipped_entries = []
        for rec in (skipped or []):
            skipped_entries.append({
                "status":      "skipped",
                "chunk_id":    rec["chunk_id"],
                "source_page": rec["source_page"],
                "skip_reason": rec["skip_reason"],
                "char_count":  rec["char_count"],
                "heading":     rec.get("heading", ""),
                "text":        rec.get("text", ""),
            })

        # -- Vision analysis log entries ------------------------------------
        vision_entries = []
        for item in (vision_results or []):
            source_page, image_url, image_signal = item
            vision_entries.append({
                "source_page":      source_page,
                "image_url":        image_url,
                "hex_codes":        image_signal.color_palette.hex_codes,
                "roles":            image_signal.color_palette.roles,
                "shot_size":        image_signal.shot_size,
                "camera_angle":     image_signal.camera_angle,
                "depth_of_field":   image_signal.depth_of_field,
                "lighting":         image_signal.lighting,
                "people_presence":  image_signal.people_presence,
                "setting":          image_signal.setting,
                "composition":      image_signal.composition,
                "product_focus":    image_signal.product_focus,
                "narrative_style":  image_signal.narrative_style,
            })

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "pipeline_run":          ts,
                "source_url":            profile.source_url,
                "model":                 "gemma3:4b",
                "chunks_processed":      len(processed_entries),
                "chunks_skipped":        len(skipped_entries),
                # Raw Kansei vote tallies before the neutral threshold filter.
                # Shows actual pole margins — useful for tuning the threshold.
                "kansei_raw_votes":      profile.kansei_raw_votes,
                "processed":             processed_entries,
                "skipped":               skipped_entries,
                # One entry per analysed image: source_page, image_url, hex_codes, roles.
                # Used by color_review.ipynb to display the correct image next to
                # the colors Gemma extracted from it.
                "vision_analysis":       vision_entries,
            }, f, indent=2, ensure_ascii=False)
        logger.info("decision log saved → %s", log_path)

    return profile_path, log_path


# ── Main entry point ──────────────────────────────────────────────────────

def run_pipeline(
    corpus_path: str | Path = "./output/corpus.json",
    output_path: str | Path = "./output/brand_profile.json",
    source_url: str = "",
    criteria_dir: str = "./criteria",
    ollama_base_url: str = "http://localhost:11434",
    text_model: str = TEXT_MODEL,
    vision_model: str = VISION_MODEL,
    skip_vision: bool = False,
) -> BrandProfile:
    """
    Runs the full brand analysis pipeline end to end.

    Args:
        corpus_path:     Path to corpus.json produced by the Phase 1 scraper.
        output_path:     Destination path for the output brand_profile.json.
        source_url:      Root URL of the scraped website. Used for metadata.
                         Inferred from corpus if omitted.
        criteria_dir:    Directory containing the four criteria JSON files.
        ollama_base_url: Base URL of the running Ollama instance.
        text_model:      Ollama model for Pass 1a (text) and Pass 2 (synthesis).
        vision_model:    Ollama model for Pass 1b (vision analysis).
        skip_vision:     If True, Pass 1b is skipped entirely.

    Returns:
        The validated BrandProfile instance (also written to output_path).

    Raises:
        FileNotFoundError: If corpus_path does not exist.
        RuntimeError: If Ollama is unreachable or the model is unavailable.
    """
    # -- Load corpus ---------------------------------------------------------
    corpus = load_corpus(corpus_path)

    # -- Infer source URL from corpus metadata if not provided ---------------
    if not source_url:
        first_page = next(iter(corpus.values()), {})
        source_url = first_page.get("meta", {}).get("url", "unknown")
        # Strip path to get root URL
        from urllib.parse import urlparse
        parsed = urlparse(source_url)
        source_url = f"{parsed.scheme}://{parsed.netloc}"
        logger.info("source_url inferred from corpus: %s", source_url)

    # -- Initialise clients --------------------------------------------------
    text_client = OllamaClient(
        base_url=ollama_base_url,
        model=text_model,
        criteria_dir=criteria_dir,
    )
    aggregator = Aggregator(
        base_url=ollama_base_url,
        model=text_model,
    )

    # -- Health check --------------------------------------------------------
    logger.info("checking Ollama availability...")
    if not text_client.ping():
        raise RuntimeError(
            f"Ollama is not reachable at {ollama_base_url} "
            f"or model '{model}' is not available. "
            f"Run: ollama pull {text_model}"
        )
    logger.info("Ollama OK — text model '%s', vision model '%s'", text_model, vision_model)

    # -- Pass 1a: Text analysis ----------------------------------------------
    signals, skipped = run_text_analysis(corpus, text_client)

    if not signals:
        raise RuntimeError(
            "Pass 1a produced no ChunkSignals. "
            "Check corpus.json structure and Ollama logs."
        )

    # -- Pass 1b: Vision analysis --------------------------------------------
    vision_results: list = []
    if not skip_vision:
        signals, vision_results = run_vision_analysis(
            corpus,
            signals,
            ollama_base_url=ollama_base_url,
            model=vision_model,
        )

    # -- Pass 2: Synthesis ---------------------------------------------------
    profile = run_synthesis(signals, source_url, aggregator)

    # -- Save output ---------------------------------------------------------
    save_profile(profile, output_path, model=text_model, signals=signals,
                 skipped=skipped, vision_results=vision_results)

    return profile


# ── CLI entry point ───────────────────────────────────────────────────────
#
# Allows running the pipeline directly:
#   python pipeline.py --corpus ./output/corpus.json --url https://horizn-studios.com

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the Horizn Analytics brand analysis pipeline."
    )
    parser.add_argument(
        "--corpus",
        default="./output/corpus.json",
        help="Path to corpus.json (default: ./output/corpus.json)",
    )
    parser.add_argument(
        "--output",
        default="./output/brand_profile.json",
        help="Output path for brand_profile.json",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Root URL of the scraped website (inferred from corpus if omitted)",
    )
    parser.add_argument(
        "--criteria",
        default="./criteria",
        help="Path to criteria JSON directory (default: ./criteria)",
    )
    parser.add_argument(
        "--text-model",
        default=TEXT_MODEL,
        help=f"Model for text analysis and synthesis (default: {TEXT_MODEL})",
    )
    parser.add_argument(
        "--vision-model",
        default=VISION_MODEL,
        help=f"Model for vision analysis (default: {VISION_MODEL})",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip Pass 1b vision analysis",
    )

    args = parser.parse_args()

    try:
        profile = run_pipeline(
            corpus_path  = args.corpus,
            output_path  = args.output,
            source_url   = args.url,
            criteria_dir = args.criteria,
            text_model   = args.text_model,
            vision_model = args.vision_model,
            skip_vision  = args.skip_vision,
        )
        print(f"\nPipeline complete.")
        print(f"  Chunks analysed : {profile.chunks_analysed}")
        print(f"  Confidence      : {profile.profile_confidence:.2f}")
        print(f"  Warnings        : {len(profile.low_signal_warnings)}")
        print(f"  Output          : {args.output}")

    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
