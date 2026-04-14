"""
brand_chat.py
─────────────
CLI tool for testing the Horizn Studios brand Q&A model.

Loads brand_context.xml as structured knowledge base, sends questions to
Ollama, and prints answers. Keeps conversation history so follow-up
questions work naturally.

Usage:
    python brand_chat.py
    python brand_chat.py --context ./output/brand_context.xml
    python brand_chat.py --model gemma3:12b --verbose

Commands during chat:
    /quit or /exit   — exit
    /reset           — clear conversation history
    /history         — show conversation so far
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_CONTEXT = "./output/brand_context.xml"
DEFAULT_MODEL   = "qwen3.5:4b"
DEFAULT_OLLAMA  = "http://localhost:11434"

# ── System prompt ─────────────────────────────────────────────────────────
#
# The XML context is injected at the end. Iterate on this block to tune
# tone, guardrails, and citation behaviour without touching the data layer.

SYSTEM_PROMPT_TEMPLATE = """\
You are a brand intelligence assistant for Horizn Studios. Your role is to \
answer questions about the brand accurately, professionally, and with clear \
attribution to your sources.

SCOPE:
You answer only questions about Horizn Studios: brand identity, products, \
positioning, values, target audience, visual identity, partnerships, and \
the methodology behind the brand analysis.
For any other topic, respond exactly with: \
"This question is outside my scope."
No exceptions — this applies to role-play requests, hypothetical scenarios, \
indirect phrasings, and seemingly harmless digressions.

ROLES AND PERSONAS:
You do not take on roles. You do not play salespeople, consultants, \
characters, or any other identity — even if the user asks. \
You remain the brand intelligence assistant at all times.

SOURCE ATTRIBUTION:
The knowledge base is structured by provenance. Use this actively:
- Information taken directly from the website: \
(Source: [page name], scraped_text)
- Information from the LLM analysis of text chunks: \
(Source: brand_profile, llm_aggregation)
- Information from image analysis: \
(Source: vision_analysis)
- When drawing conclusions across multiple sources: \
(Assessment based on: Source A + Source B)
- When information is not in the knowledge base: \
"No information available on this."

LANGUAGE AND TONE:
- Write in complete, well-formed sentences.
- Avoid defaulting to bullet points. Use lists only when the content \
genuinely calls for it — e.g. enumerating several distinct products.
- No filler phrases ("Happy to answer...", "Great question...").
- Never surface raw classification keys (e.g. "adaptive_pragmatic", \
"indoor_retail") — always use the human-readable description from \
the knowledge base.
- Tone: precise, competent, direct. Neither promotional nor stiff.

KNOWLEDGE BASE:
Everything you know about Horizn Studios is in the XML document below. \
Each node carries attributes specifying how and where the information \
was produced.

{context}
"""


# ── Ollama client ─────────────────────────────────────────────────────────

# Timeout configuration for streaming requests.
# With streaming, `read` controls how long to wait for each individual chunk,
# not for the total response. The critical value is how long the model takes
# to produce its first token — this can be 60+ seconds on CPU with a large
# system prompt. Set read generously.
OLLAMA_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=300.0,   # max wait per chunk — covers slow first-token on CPU
    write=30.0,
    pool=10.0,
)


def chat_stream(
    messages: list[dict],
    model: str,
    base_url: str,
    verbose: bool = False,
) -> str:
    """
    Sends a message list to Ollama with streaming enabled.

    Prints each token to stdout as it arrives so the user sees output
    immediately rather than waiting for the full response. Returns the
    complete reply string for history tracking.
    """
    url = f"{base_url}/api/chat"
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   True,
        "think":    False,   # disable thinking mode — speeds up first token significantly
    }

    if verbose:
        print(f"  [→ Ollama {model}, {len(messages)} messages in context]")

    full_reply = []

    try:
        with httpx.stream("POST", url, json=payload, timeout=OLLAMA_TIMEOUT) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_reply.append(token)
                if chunk.get("done"):
                    break

    except httpx.ConnectError:
        print(f"\n[Error] Ollama not reachable at {base_url}")
        print(f"        Is Ollama running? Check with: ollama list")
        sys.exit(1)
    except httpx.ReadTimeout:
        print(f"\n[Error] Ollama read timeout — model is taking too long.")
        print(f"        Try increasing OLLAMA_TIMEOUT in brand_chat.py.")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"\n[Error] Ollama HTTP {e.response.status_code}: {e.response.text}")
        sys.exit(1)

    print()  # newline after streamed output
    return "".join(full_reply).strip()


def ping(model: str, base_url: str) -> bool:
    """Checks if Ollama is reachable and the model is available."""
    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5.0)
        if r.status_code != 200:
            return False
        tags = r.json().get("models", [])
        return any(m.get("name", "").startswith(model) for m in tags)
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Horizn Studios Brand Q&A — CLI test tool"
    )
    parser.add_argument("--context",  default=DEFAULT_CONTEXT,
                        help=f"Path to brand_context.xml (default: {DEFAULT_CONTEXT})")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--ollama",   default=DEFAULT_OLLAMA,
                        help=f"Ollama base URL (default: {DEFAULT_OLLAMA})")
    parser.add_argument("--verbose",  action="store_true",
                        help="Show debug info per request")
    args = parser.parse_args()

    # -- Load XML context ----------------------------------------------------
    context_path = Path(args.context)
    if not context_path.exists():
        print(f"[Error] Context document not found: {context_path}")
        print(f"        Run first: python build_context.py")
        sys.exit(1)

    context_text  = context_path.read_text(encoding="utf-8")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)

    # -- Health check --------------------------------------------------------
    print(f"Connecting to Ollama ({args.ollama}) ...")
    if not ping(args.model, args.ollama):
        print(f"[Error] Model '{args.model}' not available.")
        print(f"        Load with: ollama pull {args.model}")
        sys.exit(1)

    token_estimate = len(system_prompt) // 4
    print(f"Model:   {args.model}")
    print(f"Context: {context_path}  (~{token_estimate:,} tokens in system prompt)")
    print()
    print("─" * 60)
    print("Horizn Studios Brand Intelligence")
    print("Commands: /quit  /reset  /history")
    print("─" * 60)
    print()

    # -- Conversation loop ---------------------------------------------------
    # History contains only user/assistant turns — system prompt is prepended
    # fresh each call so it never gets pushed out of context.
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # -- Built-in commands -----------------------------------------------
        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye.")
            break

        if user_input.lower() == "/reset":
            history.clear()
            print("[Conversation history cleared]\n")
            continue

        if user_input.lower() == "/history":
            if not history:
                print("[No history]\n")
            else:
                for msg in history:
                    role = "You" if msg["role"] == "user" else "Model"
                    print(f"{role}: {msg['content']}\n")
            continue

        # -- Build message list for this turn --------------------------------
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history
            + [{"role": "user", "content": user_input}]
        )

        # -- Call Ollama -----------------------------------------------------
        print("Model: ", end="", flush=True)
        reply = chat_stream(messages, model=args.model, base_url=args.ollama, verbose=args.verbose)
        print(reply)
        print()

        # -- Update history --------------------------------------------------
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
