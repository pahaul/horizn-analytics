"""
vision_client.py
────────────────
Pass 1b of the Horizn Analytics brand analysis pipeline.

Responsibilities:
    - Download images from corpus.json URLs and base64-encode them.
    - Send each image to the Ollama vision endpoint (Gemma 3 4B or compatible).
    - Parse the model response into a validated ImageSignal instance.
    - Merge ImageSignal data into the existing ChunkSignals collection
      (grouped by source_page).

ImageSignal replaces the old ColorSignal and captures a full set of
photographic and compositional signals alongside the color palette:
    - Color palette (hex codes + roles)
    - Shot size, camera angle, depth of field
    - Lighting style
    - People presence, IPTC setting
    - Composition principle (optional)
    - Product focus, narrative style (optional)

Design principles carried forward:
    - No tool calling — unreliable in small models by design.
    - Stateless calls — no conversation history; all context in prompt.
    - JSON repair pattern — markdown fences stripped before parsing.
    - No stop tokens — Gemma stops prematurely on backtick sequences.
    - Retry logic on parse/validation failure.
    - English comments and docstrings throughout.
"""

import base64
import json
import logging
import re
import time
from collections import defaultdict
from typing import Any

import httpx
from pydantic import ValidationError

from schemas import ChunkSignals, ImageSignal, ColorSignal

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL      = "http://localhost:11434"
DEFAULT_MODEL        = "gemma3:4b"
DEFAULT_TIMEOUT      = 300.0  # Vision calls are slower than text — raise if needed
MAX_RETRIES          = 3
RETRY_DELAY          = 2.0

MAX_IMAGES_PER_PAGE  = 3
MAX_IMAGES           = 20


# ── JSON repair utilities ─────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    pattern = r"^```(?:json)?\s*(.*?)\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


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


def _repair_json(raw: str) -> str:
    raw = _strip_markdown_fences(raw)
    raw = _extract_first_json_object(raw)
    return raw


# ── Prompt ────────────────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = (
    "You are a visual brand analyst specialising in photography and brand imagery. "
    "Your task is to analyse brand images and classify them using controlled "
    "vocabularies. Be precise and literal — classify what you actually see, "
    "not what you assume. "
    "Respond only with a single valid JSON object. No explanation, no prose."
)

# Schema hint shown to the model in the user prompt.
# Each field lists its exact valid values to constrain the model's output.
IMAGE_SIGNAL_SCHEMA_HINT = json.dumps({
    "color_palette": {
        "hex_codes": ["#RRGGBB"],
        "roles": {"#RRGGBB": "<one of: primary, secondary, accent, background, text, neutral>"}
    },
    "shot_size": "<one of: extreme_close_up, close_up, medium_close_up, medium, medium_wide, wide, extreme_wide>",
    "camera_angle": "<one of: eye_level, high_angle, low_angle, dutch_tilt, overhead>",
    "depth_of_field": "<one of: shallow, moderate, deep>",
    "lighting": "<one of: natural_soft, natural_harsh, studio_soft, studio_dramatic, backlit, low_key, high_key>",
    "people_presence": "<one of: none, hands_only, partial_body, full_body, face_close_up, group>",
    "setting": "<one of: indoor_retail, outdoor_travel, indoor_workspace, outdoor_urban, indoor_home>",
    "composition": "<one of: rule_of_thirds, centered, symmetrical, leading_lines, negative_space, frame_within_frame — or omit if unclear>",
    "product_focus": "<one of: product_only, product_in_use, lifestyle_with_product, lifestyle_without_product — or omit if no product>",
    "narrative_style": "<one of: documentary, staged, abstract, flat_lay — or omit if unclear>",
}, indent=2)


# ── Relevance filter ─────────────────────────────────────────────────────
#
# A lightweight pre-filter pass before full ImageSignal analysis.
# The model answers a single yes/no question: is this image showing a
# product in a real-life context (location, activity, person interacting
# with it) rather than isolated on a plain background?
#
# Keeps: lifestyle shots, campaign images, editorial photography,
#        products carried/used by people, products at locations.
# Removes: plain-background product shots, logos, icons, UI elements,
#          material close-ups, award badges, empty product shots.

RELEVANCE_SYSTEM_PROMPT = (
    "You are a brand image classifier. "
    "Answer only with a single word: YES or NO. No explanation."
)

RELEVANCE_USER_PROMPT = (
    "Does this image show a product in a real-life context — "
    "for example at a location, carried or used by a person, or in an "
    "environmental setting? "
    "Answer NO if the image shows: a product on a plain or white background, "
    "a logo, an icon, an award badge, a material close-up, or any UI element. "
    "Answer YES or NO only."
)


def _build_vision_prompt(alt_text: str) -> str:
    """
    Constructs the user-facing prompt for a single vision analysis call.

    Mandatory fields (shot_size, camera_angle, depth_of_field, lighting,
    people_presence, setting, color_palette) must always be populated.
    Optional fields (composition, product_focus, narrative_style) may be
    omitted if the image does not provide a clear signal.
    """
    return (
        f"Analyse the brand image provided.\n\n"
        f"IMAGE CONTEXT:\n"
        f"  alt_text: {alt_text or '(none)'}\n\n"
        f"TASK:\n"
        f"  Classify the image using the schema below.\n\n"
        f"MANDATORY FIELDS (always fill these):\n"
        f"  color_palette — 2 to 5 dominant brand colors with roles\n"
        f"  shot_size     — camera distance from subject\n"
        f"  camera_angle  — vertical angle of the camera\n"
        f"  depth_of_field — shallow = background blurred, deep = all sharp\n"
        f"  lighting      — dominant lighting character\n"
        f"  people_presence — whether/how people appear\n"
        f"  setting       — physical environment\n\n"
        f"OPTIONAL FIELDS (omit key entirely if unclear):\n"
        f"  composition   — dominant compositional principle\n"
        f"  product_focus — relationship of product to image content\n"
        f"  narrative_style — visual storytelling approach\n\n"
        f"REQUIRED OUTPUT SCHEMA:\n"
        f"{IMAGE_SIGNAL_SCHEMA_HINT}\n\n"
        f"Return only a single valid JSON object. No explanation."
    )


# ── VisionClient ──────────────────────────────────────────────────────────

class VisionClient:
    """
    Handles Pass 1b: image URL → ImageSignal via Ollama vision.

    Downloads each image, base64-encodes it, sends it to the Ollama
    vision endpoint, and parses the response into an ImageSignal instance.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        max_images_per_page: int = MAX_IMAGES_PER_PAGE,
        max_images: int = MAX_IMAGES,
    ):
        self.base_url            = base_url.rstrip("/")
        self.model               = model
        self.max_retries         = max_retries
        self.max_images_per_page = max_images_per_page
        self.max_images          = max_images
        self._http               = httpx.Client(timeout=timeout)

    # ── Image acquisition ─────────────────────────────────────────────────

    def _download_image_b64(self, url: str) -> str:
        """
        Downloads an image and returns it as a base64-encoded string.

        Raises httpx.HTTPError on download failure.
        """
        response = self._http.get(url)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    # ── Relevance pre-filter ─────────────────────────────────────────────

    def _is_relevant_image(self, image: dict[str, Any]) -> bool:
        """
        Returns True if the image shows a product in a real-life context.

        Sends a single yes/no question to the vision model. Images that show
        only an isolated product on a plain background, logos, icons, or UI
        elements are rejected before the full ImageSignal analysis runs.

        Returns True on any error so that uncertain images are not silently
        dropped — they will be caught by the full analysis retry logic.
        """
        url = image.get("url", "")
        try:
            image_b64 = self._download_image_b64(url)
            payload = {
                "model":   self.model,
                "prompt":  RELEVANCE_USER_PROMPT,
                "system":  RELEVANCE_SYSTEM_PROMPT,
                "images":  [image_b64],
                "stream":  False,
                "options": {"temperature": 0.0},
            }
            response = self._http.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip().upper()
            relevant = answer.startswith("YES")
            logger.debug(
                "relevance filter: %s — %s",
                "KEEP" if relevant else "SKIP",
                url[:70],
            )
            return relevant
        except Exception as exc:
            logger.warning(
                "relevance check failed for '%s', keeping image: %s",
                url[:60], exc,
            )
            return True  # fail open — don't silently drop on error

    # ── Single image analysis ─────────────────────────────────────────────

    def analyse_image(self, image: dict[str, Any]) -> ImageSignal | None:
        """
        Analyses a single image and returns an ImageSignal, or None on failure.

        Sends the image to the Ollama vision endpoint with the structured
        prompt. Retries up to max_retries times on parse or validation error.

        Args:
            image: Image dict from corpus.json, must include 'url' and
                   optionally 'alt_text'.

        Returns:
            Validated ImageSignal instance, or None if all retries fail.
        """
        url      = image.get("url", "")
        alt_text = image.get("alt_text", "")
        prompt   = _build_vision_prompt(alt_text)

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                image_b64 = self._download_image_b64(url)

                payload = {
                    "model":  self.model,
                    "prompt": prompt,
                    "system": VISION_SYSTEM_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.1},
                }

                response = self._http.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()

                raw      = response.json().get("response", "")
                repaired = _repair_json(raw)
                data     = json.loads(repaired)

                # Normalise color_palette sub-object if present
                if "color_palette" not in data:
                    data["color_palette"] = {}

                signal = ImageSignal.model_validate(data)

                logger.debug(
                    "image '%s' validated OK (attempt %d)",
                    url[:60], attempt,
                )
                return signal

            except (httpx.HTTPError, ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "image '%s' attempt %d/%d failed: %s",
                    url[:60], attempt, self.max_retries, exc,
                )
                if attempt < self.max_retries:
                    time.sleep(RETRY_DELAY)

        logger.error(
            "image '%s' failed after %d attempts. Last error: %s",
            url[:60], self.max_retries, last_error,
        )
        return None

    # ── Sampling ─────────────────────────────────────────────────────────

    # URL fragments that identify e-commerce product packshots —
    # front/back/side/3-4 views and detail shots with no environmental context.
    # These are filtered before sampling to avoid skewing the visual analysis
    # with plain white-background product photography.
    # Packshot patterns matched as whole path segments — i.e. the pattern
    # must be followed by a dot, underscore, or end of string to avoid
    # false positives like "_Backpack" matching "_Back".
    PACKSHOT_PATTERNS = ("_Front", "_Back", "_3_4", "_Side", "_Detail")

    def _is_packshot(self, url: str) -> bool:
        """
        Returns True if the URL contains a packshot pattern as a whole segment.

        Matches _Front, _Back, _Side, _Detail, _3_4 only when followed by
        a dot, underscore, or end of string — preventing false positives
        like '_Backpack' matching '_Back'.
        """
        import re
        for pattern in self.PACKSHOT_PATTERNS:
            if re.search(re.escape(pattern) + r'(?:[._]|$)', url):
                return True
        return False

    def _sample_images(
        self,
        images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Selects a representative subset of images for vision analysis.

        Strategy:
            1. Deduplicate by URL — Shopify CDN images often repeat across pages.
            2. Filter packshots — front/back/side/3-4/detail views on white
               backgrounds add no environmental or compositional signal.
            3. Group by source_page.
            4. Take at most max_images_per_page per page.
            5. Apply global max_images cap.
        """
        seen_urls: set[str] = set()
        deduplicated: list[dict[str, Any]] = []
        for img in images:
            if img["url"] not in seen_urls:
                seen_urls.add(img["url"])
                deduplicated.append(img)

        duplicates_removed = len(images) - len(deduplicated)
        if duplicates_removed:
            logger.info(
                "pass 1b — deduplicated %d images (%d duplicates removed)",
                len(deduplicated), duplicates_removed,
            )

        # Filter packshots
        filtered = [img for img in deduplicated if not self._is_packshot(img["url"])]
        packshots_removed = len(deduplicated) - len(filtered)
        if packshots_removed:
            logger.info(
                "pass 1b — removed %d packshot images (front/back/side/3-4/detail)",
                packshots_removed,
            )
        deduplicated = filtered

        by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for img in deduplicated:
            by_page[img.get("source_page", "unknown")].append(img)

        sampled: list[dict[str, Any]] = []
        for page_label, page_images in by_page.items():
            page_sample = page_images[: self.max_images_per_page]
            logger.debug(
                "page '%s': %d images available, %d sampled",
                page_label, len(page_images), len(page_sample),
            )
            sampled.extend(page_sample)

        if len(sampled) > self.max_images:
            logger.info(
                "applying global cap: %d → %d images",
                len(sampled), self.max_images,
            )
            sampled = sampled[: self.max_images]

        return sampled

    # ── Batch analysis ────────────────────────────────────────────────────

    def analyse_images(
        self,
        images: list[dict[str, Any]],
    ) -> list[tuple[str, str, ImageSignal]]:
        """
        Analyses a sampled subset of images.

        Returns:
            List of (source_page, image_url, ImageSignal) triples for all
            successfully analysed images.
        """
        logger.info(
            "pass 1b — %d images across all pages (before dedup/sampling)",
            len(images),
        )

        selected = self._sample_images(images)

        logger.info(
            "pass 1b — analysing %d images (max %d/page, global cap %d)",
            len(selected), self.max_images_per_page, self.max_images,
        )

        # -- Relevance pre-filter -------------------------------------------
        # Each image gets a fast yes/no check before the full analysis.
        # This removes logos, plain-background product shots, UI elements
        # and other non-contextual images without wasting a full analysis call.
        relevant: list[dict[str, Any]] = []
        for i, image in enumerate(selected, 1):
            url = image.get("url", "")
            logger.info(
                "  relevance [%d/%d] %s",
                i, len(selected), url[:70],
            )
            if self._is_relevant_image(image):
                relevant.append(image)

        logger.info(
            "pass 1b — relevance filter: %d/%d images kept",
            len(relevant), len(selected),
        )

        # -- Full ImageSignal analysis --------------------------------------
        results: list[tuple[str, str, ImageSignal]] = []
        for i, image in enumerate(relevant, 1):
            source_page = image.get("source_page", "unknown")
            url         = image.get("url", "")
            logger.info(
                "  analyse [%d/%d] %s — %s",
                i, len(relevant),
                source_page,
                url[:70],
            )
            signal = self.analyse_image(image)
            if signal is not None:
                results.append((source_page, url, signal))

        logger.info(
            "pass 1b complete: %d/%d relevant images produced ImageSignals",
            len(results), len(relevant),
        )
        return results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._http.close()


# ── Merge helper ──────────────────────────────────────────────────────────

def merge_image_signals_into_chunks(
    signals: list[ChunkSignals],
    image_results: list[tuple[str, str, ImageSignal]],
) -> list[ChunkSignals]:
    """
    Merges ImageSignal data into the ChunkSignals collection.

    Attaches each ImageSignal to the first ChunkSignal from the same
    source_page that still has an empty image_signal field. One ImageSignal
    per chunk, distributed in corpus order.

    Args:
        signals:       Flat list of ChunkSignals from Pass 1a.
        image_results: List of (source_page, image_url, ImageSignal) triples
                       from VisionClient.

    Returns:
        The same list of ChunkSignals, mutated in-place with image signals.
    """
    if not image_results:
        return signals

    from collections import deque
    page_queue: dict[str, deque[ImageSignal]] = defaultdict(deque)
    for source_page, _url, image_signal in image_results:
        page_queue[source_page].append(image_signal)

    for chunk_signal in signals:
        page = chunk_signal.source_page
        # An empty ImageSignal has no shot_size set
        if page in page_queue and page_queue[page] and not chunk_signal.image_signal.shot_size:
            chunk_signal.image_signal = page_queue[page].popleft()
            logger.debug(
                "attached ImageSignal to chunk %d (page '%s')",
                chunk_signal.chunk_id, page,
            )

    leftover = sum(len(q) for q in page_queue.values())
    if leftover:
        logger.info(
            "%d ImageSignal(s) had no matching empty ChunkSignal — "
            "aggregator pools across all chunks.",
            leftover,
        )

    return signals
