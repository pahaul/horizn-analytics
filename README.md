Horizn Analytics
A pipeline that scrapes a brand's website, extracts visual and emotional signals using structured analytical frameworks, and assembles them into structured AI image generation prompts.
Built as a proof of concept using Horizn Studios as the target brand.
→ Read the article

Pipeline
scraping.ipynb         →  corpus.json
python pipeline.py     →  brand_profile.json + decision_log.json
python build_context.py →  brand_context.xml
python build_prompt.py  →  structured image prompt
python brand_chat.py    →  interactive brand Q&A (CLI)
Image generation (Nano Banana 2 via ComfyUI) is done manually using the prompt output.
Stack

Python 3.11, Ollama, Gemma 31B (cloud)
Playwright, BeautifulSoup
FastAPI (optional wrapper)
Nano Banana 2 via ComfyUI for image generation

Usage
bash# 1. Run scraping notebook
#    scraping.ipynb → ./output/corpus.json

# 2. Run analysis pipeline
python pipeline.py

# 3. Generate image prompt
python build_prompt.py

# 4. Build brand context for chat
python build_context.py

# 5. Brand chat CLI
python brand_chat.py
Structure
scraping.ipynb       — scraper (Playwright + BeautifulSoup)
pipeline.py          — orchestrator
schemas.py           — Pydantic models
ollama_client.py     — text analysis (Pass 1a + Pass 2)
vision_client.py     — vision analysis (Pass 1b)
aggregator.py        — signal aggregation
build_context.py     — builds brand_context.xml
brand_chat.py        — CLI chat interface
build_prompt.py      — prompt generator
criteria/            — analytical framework definitions
output/              — pipeline outputs (gitignored)
Output example
brand_profile.json — Kansei profile
────────────────────────────────────────────
modern_traditional     modern        36.5
premium_accessible     premium       38.3
rough_refined          refined       23.4
profile_confidence     0.844
