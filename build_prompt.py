"""
build_prompt.py
Reads the latest brand_profile_gemma4-31b-cloud_*.json and
decision_log_*.json from ./output/ and assembles two FLUX prompts:
  1. Naive baseline (hardcoded)
  2. Structured brand prompt (deterministic template — no LLM call)

BFL prompt structure (docs.bfl.ml/guides/prompting_summary):
  Subject → Action → Environment → Lighting → Composition → Color palette → Mood
"""

import json
import glob
import os
import random
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./output"

# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def find_latest(pattern: str) -> str | None:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, pattern)))
    return files[-1] if files else None


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pool extraction from decision log
# ---------------------------------------------------------------------------

def extract_color_pool(log: dict) -> dict:
    """role -> [hex, ...], duplicates kept for frequency weighting."""
    pool: dict = defaultdict(list)
    for entry in log.get("vision_analysis", []):
        for hex_code, role in entry.get("roles", {}).items():
            pool[role].append(hex_code)
    return dict(pool)


def extract_signal_pool(log: dict) -> dict:
    """field -> [value, ...], duplicates kept for frequency weighting."""
    FIELDS = ["shot_size", "camera_angle", "depth_of_field", "lighting",
              "people_presence", "setting", "composition", "narrative_style"]
    pool: dict = defaultdict(list)
    for entry in log.get("vision_analysis", []):
        for field in FIELDS:
            val = entry.get(field)
            if val:
                pool[field].append(val)
    return dict(pool)


# ---------------------------------------------------------------------------
# Label maps (taxonomy key -> natural language phrase)
# ---------------------------------------------------------------------------

LABELS = {
    # setting
    "outdoor_urban":    "an outdoor urban plaza",
    "outdoor_travel":   "an outdoor travel location",
    "indoor_home":      "a minimal indoor home interior",
    "indoor_retail":    "an indoor retail space",
    "indoor_workspace": "a clean indoor workspace",
    # people presence
    "partial_body":     "shown in a partial body shot from the waist down",
    "full_body":        "standing in a full body shot",
    "none":             "with no person present — product only",
    "group":            "with a small group of people",
    # shot size
    "medium":           "medium shot",
    "medium_wide":      "medium wide shot",
    "medium_close_up":  "medium close-up shot",
    "wide":             "wide shot",
    "close_up":         "close-up shot",
    # camera angle
    "eye_level":        "shot at eye level",
    "low_angle":        "shot from a low angle looking up",
    "high_angle":       "shot from a slight high angle",
    # depth of field
    "shallow":          "shallow depth of field with a softly blurred background",
    "moderate":         "moderate depth of field",
    "deep":             "deep depth of field with a sharp background",
    # lighting
    "natural_soft":     "soft diffused natural daylight",
    "natural_harsh":    "direct harsh natural sunlight with defined shadows",
    "studio_soft":      "soft controlled studio lighting",
    # composition
    "centered":         "centered composition",
    "rule_of_thirds":   "rule-of-thirds composition",
    # narrative
    "staged":           "staged editorial campaign photography",
    "documentary":      "documentary candid photography",
}

def l(key: str) -> str:
    return LABELS.get(key, key.replace("_", " "))


# ---------------------------------------------------------------------------
# Slot drawing
# ---------------------------------------------------------------------------

def draw_slots(color_pool: dict, signal_pool: dict) -> dict:
    def pick_color(role: str) -> str:
        c = color_pool.get(role, [])
        return random.choice(c) if c else "—"

    def pick_signal(field: str) -> str:
        c = signal_pool.get(field, [])
        return random.choice(c) if c else "—"

    outfit_pool = color_pool.get("neutral", []) + color_pool.get("secondary", [])

    return {
        "setting":          pick_signal("setting"),
        "people_presence":  pick_signal("people_presence"),
        "shot_size":        pick_signal("shot_size"),
        "camera_angle":     pick_signal("camera_angle"),
        "depth_of_field":   pick_signal("depth_of_field"),
        "lighting":         pick_signal("lighting"),
        "composition":      pick_signal("composition"),
        "narrative_style":  pick_signal("narrative_style"),
        "suitcase_color":   pick_color("primary"),
        "outfit_color":     random.choice(outfit_pool) if outfit_pool else "—",
        "background_color": pick_color("background"),
        "floor_color":      pick_color("neutral"),
        "accent_color":     pick_color("accent"),
    }


# ---------------------------------------------------------------------------
# Prompt assembly — BFL structure:
# Subject -> Action -> Environment -> Lighting -> Composition -> Colors -> Mood
# ---------------------------------------------------------------------------

def assemble_prompt(slots: dict) -> str:
    s = slots

    # Subject + action
    if s["people_presence"] == "none":
        subject = (
            f"A {l(s['shot_size'])} of a premium Horizn Studios hard-shell suitcase "
            f"in {s['suitcase_color']}, standing alone"
        )
    else:
        subject = (
            f"A person {l(s['people_presence'])}, "
            f"pulling a premium Horizn Studios hard-shell suitcase in {s['suitcase_color']}, "
            f"wearing a muted monochromatic outfit in {s['outfit_color']}"
        )

    # Environment
    environment = (
        f"set in {l(s['setting'])} with floor surfaces in {s['floor_color']} "
        f"and background in {s['background_color']}"
    )

    # Lighting
    lighting = l(s["lighting"])

    # Composition
    composition = (
        f"{l(s['shot_size'])}, {l(s['camera_angle'])}, "
        f"{l(s['composition'])}, {l(s['depth_of_field'])}"
    )

    # Accents
    accents = f"accent details in {s['accent_color']} on hardware and zippers"

    # Mood + style
    mood = (
        f"premium, refined, and timeless mood, "
        f"{l(s['narrative_style'])}"
    )

    prompt = (
        f"Editorial lifestyle campaign photo for a premium luggage brand. "
        f"{subject}, {environment}. "
        f"{lighting}, {composition}. "
        f"{accents}. "
        f"{mood}."
    )

    return prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load profile
    profile_path = find_latest("brand_profile_gemma4-31b-cloud_*.json")
    if not profile_path:
        raise FileNotFoundError("No brand_profile_gemma4-31b-cloud_*.json found in ./output/")
    profile = load_json(profile_path)

    # Load decision log
    log_path = find_latest("decision_log_*.json")
    if log_path:
        log         = load_json(log_path)
        color_pool  = extract_color_pool(log)
        signal_pool = extract_signal_pool(log)
    else:
        raw        = profile.get("color_palette", {})
        color_pool = defaultdict(list)
        for h, r in raw.get("roles", {}).items():
            color_pool[r].append(h)
        color_pool  = dict(color_pool)
        signal_pool = {}
        print("  [warn] No decision log found — using brand_profile palette as fallback")

    # Draw slots
    slots = draw_slots(color_pool, signal_pool)

    # Header
    print()
    print_separator("═")
    print("  HORIZN STUDIOS — FLUX PROMPT GENERATOR")
    print_separator("═")
    print(f"  Profile:    {os.path.basename(profile_path)}")
    print(f"  Confidence: {profile.get('profile_confidence', 'n/a')}")
    if log_path:
        print(f"  Log:        {os.path.basename(log_path)}")
    print_separator("═")

    # Slots
    print()
    print("  SLOTS DRAWN")
    print_separator()
    for k, v in slots.items():
        print(f"    {k:<20} {v}")

    # Prompt 1 — naive
    print()
    print_separator("═")
    print("  PROMPT 1 — NAIVE BASELINE")
    print_separator()
    print("Generate a photo for Horizn Studios, a premium travel brand.")

    # Prompt 2 — structured
    print()
    print("  PROMPT 2 — STRUCTURED BRAND PROMPT")
    print_separator()
    prompt = assemble_prompt(slots)
    print(prompt)
    print()
    print_separator("═")
    print()


if __name__ == "__main__":
    main()
