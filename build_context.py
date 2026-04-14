"""
build_context.py
────────────────
Builds a structured XML context document from corpus.json + brand_profile.json.

The output (brand_context.xml) replaces briefing.md as the knowledge base
for the chat model. XML is preferred over plain Markdown because LLMs treat
tags as semantic boundaries, making source attribution and structured
retrieval more reliable.

Every data point is tagged with:
  - source_type: how it was produced (scraped_text | llm_analysis | llm_aggregation | vision_analysis)
  - source_page:  which page of the website it came from (where applicable)
  - method:       which analytical framework produced it (where applicable)

This transparency allows the model to accurately tell users not just
*what* the data says, but *how* it was derived.

Usage:
    python build_context.py

Output:
    ./output/brand_context.xml
"""

import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────

CORPUS_PATH  = Path("./output/corpus.json")
PROFILE_GLOB = "./output/brand_profile_*.json"
OUTPUT_PATH  = Path("./output/brand_context.xml")

# ── Human-readable translations of internal labels ────────────────────────
#
# These prevent the model from ever surfacing raw taxonomy keys to users.
# Update when criteria files change.

SINUS_LABELS = {
    "adaptive_pragmatic": (
        "Adaptive Pragmatists: pragmatic, flexible, performance-oriented; "
        "seek straightforward solutions that work in everyday life"
    ),
    "performer": (
        "Performers: ambitious, efficiency-driven, tech-savvy; "
        "combine achievement with lifestyle and value status and quality"
    ),
    "liberal_intellectual": (
        "Liberal Intellectuals: highly educated, culturally open, "
        "sustainability-conscious; value authenticity and depth"
    ),
    "socioecological": (
        "Socio-Ecologicals: values-driven, sustainability-oriented, "
        "critical of consumerism; seek products with genuine purpose"
    ),
}

IPTC_LABELS = {
    "indoor_retail":     "Retail / shop environment (indoor)",
    "outdoor_travel":    "Travel / on the move (outdoor)",
    "indoor_workspace":  "Office / workspace (indoor)",
    "outdoor_urban":     "Urban exterior (outdoor)",
    "indoor_home":       "Home / domestic setting (indoor)",
}

KANSEI_EXPLANATIONS = {
    "modern_traditional":     ("modern",       "traditional",  "design language and zeitgeist"),
    "minimal_ornate":         ("minimal",       "ornate",       "design complexity"),
    "functional_decorative":  ("functional",    "decorative",   "design priority"),
    "bold_subtle":            ("bold",          "subtle",       "visual presence"),
    "urban_natural":          ("urban",         "natural",      "lifestyle association"),
    "premium_accessible":     ("premium",       "accessible",   "price positioning"),
    "timeless_trendy":        ("timeless",      "trendy",       "trend cycle dependency"),
    "individual_collective":  ("individual",    "collective",   "identity appeal"),
    "dynamic_static":         ("dynamic",       "static",       "energy and motion character"),
    "transparent_mysterious": ("transparent",   "mysterious",   "communication style"),
    "rough_refined":          ("rough",         "refined",      "material feel"),
    "serious_playful":        ("serious",       "playful",      "tonality"),
}

GEW_EXPLANATIONS = {
    "joy":      "Joy — positive activation, enthusiasm",
    "relief":   "Relief — release of tension, sense of trust",
    "interest": "Interest — cognitive curiosity, sustained attention",
    "pride":    "Pride — sense of agency, identification",
    "awe":      "Awe — fascination, sense of scale or significance",
}

SKIP_HEADINGS = {"Customer Reviews"}
MIN_CHUNK_CHARS = 40

PAGE_META = {
    "about_us": {
        "label": "About Us",
        "description": "Company history, mission, values, partnerships, quality promise",
    },
    "homepage": {
        "label": "Homepage",
        "description": "Core brand messages, current campaigns, community positioning",
    },
    "more_about_quality": {
        "label": "Quality & Materials",
        "description": "Material innovations, sustainability technologies, production standards",
    },
    "more_about_our_efforts_and_impact": {
        "label": "Sustainability & Impact",
        "description": "Social initiatives, environmental programmes, community efforts",
    },
    "more_about_our_partners": {
        "label": "Partnerships",
        "description": "Collaborations with brands, artists and organisations",
    },
    "spotlight_sofo_backpack_rolltop_x": {
        "label": "Product Page: SoFo Rolltop Backpack X",
        "description": "Product details, material specifications, variants, customer reviews",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────

def esc(text: str) -> str:
    """Escapes special XML characters in text content."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def load_latest_profile(glob_pattern: str) -> dict:
    import glob as _glob
    files = sorted(_glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No brand_profile files found matching: {glob_pattern}\n"
            "Run the pipeline first: python pipeline.py"
        )
    path = files[-1]
    print(f"Using profile: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── XML builder ───────────────────────────────────────────────────────────

def build_xml(corpus: dict, profile: dict) -> str:
    lines: list[str] = []

    source_url  = profile.get("source_url", "https://horizn-studios.com")
    generated   = profile.get("generated_at", "")[:10]
    confidence  = profile.get("profile_confidence", 0.0)

    lines += [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<brand_knowledge>",
        "",
        "  <!--",
        "    Horizn Studios — structured knowledge base for brand intelligence chat",
        f"    Website:          {source_url}",
        f"    Analysed on:      {generated}",
        f"    Profile confidence: {confidence:.1%}",
        "",
        "    Every node carries attributes that specify:",
        "      source_type — how the information was produced",
        "      source_page — which page of the website it came from (where applicable)",
        "      method      — which analytical framework was used (where applicable)",
        "",
        "    source_type values:",
        "      scraped_text      Text extracted directly from the website",
        "      llm_analysis      Per-chunk analysis by Gemma 3 4B",
        "      llm_aggregation   Aggregation of all chunk signals into a single profile",
        "      vision_analysis   Colour extraction from product images via Gemma 3 4B Vision",
        "  -->",
        "",
    ]

    # ── 1. Methodology ────────────────────────────────────────────────────
    lines += [
        '  <methodology source_type="documentation">',
        "    <!--",
        "      Explains how the brand profile was produced.",
        "      Enables the model to answer questions about the analysis methodology.",
        "    -->",
        "",
        "    <overview>",
        "      The Horizn Studios brand profile was produced in a two-pass LLM pipeline:",
        "      1. Pass 1 — Chunk analysis: every text section of the website was",
        "         analysed individually by Gemma 3 4B and tagged with brand signals.",
        "      2. Pass 2 — Aggregation: all chunk signals were statistically and",
        "         via LLM condensed into a single brand profile.",
        "      In addition, product images were analysed for colour palettes",
        "      using Gemma 3 4B Vision.",
        "    </overview>",
        "",
        '    <framework name="GEW" full_name="Geneva Emotion Wheel">',
        "      Measures the emotional impact of brand communication on the viewer.",
        "      The model classifies up to 2 emotion families per text chunk.",
        "      Academic basis: Russell &amp; Scherer, University of Geneva.",
        "      Purpose: which emotions should the brand evoke in its audience?",
        "    </framework>",
        "",
        '    <framework name="Kansei Engineering">',
        "      Bipolar adjective pairs for measuring product perception.",
        "      Up to 4 axes are scored per chunk (e.g. modern vs. traditional).",
        "      Aggregation weights chunks by their confidence score.",
        "      Purpose: how is the brand perceived? Which poles dominate?",
        "    </framework>",
        "",
        '    <framework name="IPTC" full_name="International Press Telecommunications Council">',
        "      Taxonomy for classifying visual settings and environments.",
        "      Industry standard in media and photography.",
        "      Purpose: in which contexts does the brand present itself visually?",
        "    </framework>",
        "",
        '    <framework name="Sinus-Milieus">',
        "      Sociographic target audience model (Sinus Institute, Germany).",
        "      Segments society by values and lifestyles.",
        "      Purpose: which social groups does the brand address?",
        "    </framework>",
        "",
        "    <confidence_note>",
        f"      Profile confidence: {confidence:.1%} (mean of all chunk confidence scores).",
        "      Values above 0.75 are considered reliable.",
        "      Kansei axes with a high neutral share are less conclusive —",
        "      see kansei_profile raw_votes for margins.",
        "    </confidence_note>",
        "",
        "  </methodology>",
        "",
    ]

    # ── 2. Brand Identity ─────────────────────────────────────────────────
    lines += [
        '  <brand_identity source_type="llm_aggregation" method="GEW+Kansei+Sinus">',
        "    <!--",
        "      Aggregated brand profile. Each element is the result of analysing",
        "      all pages, weighted by chunk confidence score.",
        "    -->",
        "",
        f'    <source_url>{esc(source_url)}</source_url>',
        f'    <analysed_at>{esc(generated)}</analysed_at>',
        f'    <chunks_analysed>{profile.get("chunks_analysed", 0)}</chunks_analysed>',
        "",
    ]

    # Brand adjectives
    adjectives = profile.get("brand_adjectives", [])
    lines += [
        '    <brand_adjectives source_type="llm_aggregation" method="direct extraction from brand copy">',
        "      <!-- Adjectives the brand consistently uses to describe itself or its products -->",
    ]
    for adj in adjectives:
        lines.append(f"      <adjective>{esc(adj)}</adjective>")
    lines += ["    </brand_adjectives>", ""]

    # Brand values
    values = profile.get("brand_values", [])
    lines += [
        '    <brand_values source_type="llm_aggregation" method="direct extraction from brand copy">',
        "      <!-- Explicit brand values and commitments, grounded in website copy -->",
    ]
    for val in values:
        lines.append(f"      <value>{esc(val)}</value>")
    lines += ["    </brand_values>", ""]

    # Dominant emotions (GEW)
    emotions = profile.get("dominant_emotions", [])
    lines += [
        '    <emotional_impact source_type="llm_aggregation" method="GEW (Geneva Emotion Wheel)">',
        "      <!-- Emotions the brand intends to evoke in its audience, per GEW analysis -->",
    ]
    for em in emotions:
        explanation = GEW_EXPLANATIONS.get(em, em)
        lines.append(f'      <emotion key="{esc(em)}">{esc(explanation)}</emotion>')
    lines += ["    </emotional_impact>", ""]

    # Kansei profile
    kansei = profile.get("kansei_profile", {})
    raw_votes = profile.get("kansei_raw_votes", {})
    lines += [
        '    <kansei_profile source_type="llm_aggregation" method="Kansei Engineering">',
        "      <!--",
        "        Bipolar perception axes. Each axis shows the dominant pole.",
        "        'neutral' means no clear pole — the brand is not distinctly",
        "        positioned on this axis.",
        "        raw_votes shows confidence-weighted vote tallies for transparency.",
        "      -->",
    ]
    for key, dominant in kansei.items():
        if key not in KANSEI_EXPLANATIONS:
            continue
        pole_a, pole_b, dimension = KANSEI_EXPLANATIONS[key]
        votes = raw_votes.get(key, {})
        vote_str = ", ".join(f"{k}: {v}" for k, v in votes.items())
        lines += [
            f'      <axis key="{esc(key)}" dimension="{esc(dimension)}" '
            f'pole_a="{esc(pole_a)}" pole_b="{esc(pole_b)}" dominant="{esc(dominant)}">',
            f"        <raw_votes>{esc(vote_str)}</raw_votes>",
            f"      </axis>",
        ]
    lines += ["    </kansei_profile>", ""]

    # Target audience (Sinus)
    audience = profile.get("primary_audience", [])
    lines += [
        '    <target_audience source_type="llm_aggregation" method="Sinus-Milieus">',
        "      <!-- Primary audience segments per Sinus-Milieu classification -->",
    ]
    for seg in audience:
        description = SINUS_LABELS.get(seg, seg)
        lines.append(f'      <segment key="{esc(seg)}">{esc(description)}</segment>')
    lines += ["    </target_audience>", ""]

    # Primary settings (IPTC)
    settings = profile.get("primary_settings", [])
    lines += [
        '    <primary_settings source_type="llm_aggregation" method="IPTC taxonomy">',
        "      <!-- Most frequently occurring visual settings in brand communication -->",
    ]
    for s in settings:
        label = IPTC_LABELS.get(s, s)
        lines.append(f'      <setting key="{esc(s)}">{esc(label)}</setting>')
    lines += ["    </primary_settings>", ""]

    lines += ["  </brand_identity>", ""]

    # ── 3. Visual Identity ────────────────────────────────────────────────
    palette = profile.get("color_palette", {})
    img     = profile.get("image_prompt_components", {})

    lines += [
        '  <visual_identity source_type="vision_analysis" '
        'method="Gemma 3 4B Vision — colour extraction from product images">',
        "    <!--",
        "      Colour palette derived from analysing up to 20 product images.",
        "      Note: Gemma describes colours semantically, not by pixel sampling.",
        "      Hex values are approximations (~delta 40-130 vs. PIL measurement).",
        "      Treat them as directional brand colour indicators, not exact values.",
        "    -->",
        "",
        "    <color_palette>",
    ]
    for hx in palette.get("hex_codes", []):
        role = palette.get("roles", {}).get(hx, "unknown")
        lines.append(f'      <color hex="{esc(hx)}" role="{esc(role)}" />')
    lines += ["    </color_palette>", ""]

    lines += [
        '    <image_prompt_components source_type="llm_aggregation" '
        'method="synthesis of Kansei + GEW + colour palette">',
        "      <!-- Structured building blocks for AI image generation (Midjourney, FLUX, SD) -->",
    ]
    if img.get("subject"):
        lines.append(f"      <subject>{esc(img['subject'])}</subject>")
    if img.get("environment"):
        lines.append(f"      <environment>{esc(img['environment'])}</environment>")
    if img.get("lighting"):
        lines.append(f"      <lighting>{esc(img['lighting'])}</lighting>")
    if img.get("composition"):
        lines.append(f"      <composition>{esc(img['composition'])}</composition>")
    if img.get("color_palette"):
        lines.append(f"      <color_description>{esc(img['color_palette'])}</color_description>")
    if img.get("mood_keywords"):
        lines.append(f"      <mood_keywords>{esc(', '.join(img['mood_keywords']))}</mood_keywords>")
    if img.get("negative_prompt"):
        lines.append(f"      <negative_prompt>{esc(', '.join(img['negative_prompt']))}</negative_prompt>")
    lines += [
        "    </image_prompt_components>",
        "",
        "  </visual_identity>",
        "",
    ]

    # ── 4. Page content ───────────────────────────────────────────────────
    lines += [
        '  <website_content source_type="scraped_text">',
        "    <!--",
        "      Original text from the Horizn Studios website.",
        "      Scraped: 2026-04-09.",
        "      Each page includes its URL and internal key for source attribution.",
        "      Customer reviews excluded (not brand-authored text).",
        "      Chunks under 40 characters excluded (navigation artefacts).",
        "    -->",
        "",
    ]

    for page_key, page_data in corpus.items():
        meta  = PAGE_META.get(page_key, {"label": page_key, "description": ""})
        url   = page_data.get("meta", {}).get("url", "")
        chunks = [
            c for c in page_data.get("chunks", [])
            if c.get("heading", "") not in SKIP_HEADINGS
            and len(c.get("text", "").strip()) >= MIN_CHUNK_CHARS
        ]
        if not chunks:
            continue

        lines += [
            f'    <page key="{esc(page_key)}" label="{esc(meta["label"])}">',
            f'      <url>{esc(url)}</url>',
            f'      <description>{esc(meta["description"])}</description>',
            "",
        ]

        seen: set[str] = set()
        for chunk in chunks:
            heading = chunk.get("heading", "").strip()
            text    = chunk.get("text", "").strip()
            dedup_key = text[:80]
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            if heading:
                lines.append(f'      <chunk heading="{esc(heading)}">')
            else:
                lines.append("      <chunk>")
            lines.append(f"        {esc(text)}")
            lines.append("      </chunk>")

        lines += ["    </page>", ""]

    lines += [
        "  </website_content>",
        "",
        "</brand_knowledge>",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}\n"
            "Run the Phase 1 scraper first."
        )

    print(f"Loading corpus: {CORPUS_PATH}")
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    profile = load_latest_profile(PROFILE_GLOB)

    xml = build_xml(corpus, profile)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(xml)

    char_count   = len(xml)
    token_approx = char_count // 4

    print(f"\nContext document saved: {OUTPUT_PATH}")
    print(f"  Characters:     {char_count:,}")
    print(f"  Approx. tokens: ~{token_approx:,}")
    print()
    print("Next step: run brand_chat.py to test the model against this context.")


if __name__ == "__main__":
    main()
