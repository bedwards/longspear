#!/usr/bin/env python3
"""Longspear TUI — Terminal conversation interface.

A rich terminal UI for chatting with personas and monitoring the system.
Talks to the Longspear API at localhost:28000.

Usage:
    python scripts/tui.py                    # Interactive mode
    python scripts/tui.py chat hcr           # Quick chat with HCR
    python scripts/tui.py debate              # Start a debate
    python scripts/tui.py monitor             # System monitor
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error

API = "http://localhost:28000"

# ── ANSI Colors ───────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Persona colors
HCR_COLOR = "\033[38;5;141m"   # Purple
NBJ_COLOR = "\033[38;5;208m"   # Orange
USER_COLOR = "\033[38;5;75m"   # Blue
SYS_COLOR = "\033[38;5;244m"   # Gray
OK_COLOR = "\033[38;5;78m"     # Green
ERR_COLOR = "\033[38;5;196m"   # Red
TITLE_COLOR = "\033[38;5;183m" # Light purple

PERSONA_COLORS = {
    "heather_cox_richardson": HCR_COLOR,
    "nate_b_jones": NBJ_COLOR,
}
PERSONA_NAMES = {
    "heather_cox_richardson": "Heather Cox Richardson",
    "nate_b_jones": "Nate B Jones",
}
PERSONA_SHORT = {
    "heather_cox_richardson": "HCR",
    "nate_b_jones": "NBJ",
}


def api_get(path: str) -> dict:
    """GET request to the API."""
    try:
        req = urllib.request.Request(f"{API}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def api_post_stream(path: str, body: dict):
    """POST request with SSE streaming, yields parsed events."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    yield json.loads(line[6:])
    except Exception as e:
        yield {"type": "error", "message": str(e)}


# ── Banner ────────────────────────────────────────────────
def print_banner():
    print(f"""
{TITLE_COLOR}{BOLD}  ⚔️  Longspear — AI-Moderated Debates{RESET}
{DIM}  Powered by Mistral-large:123b on Metal GPU{RESET}
{DIM}  RAG grounded in real YouTube transcripts{RESET}
""")


# ── Monitor ───────────────────────────────────────────────
def cmd_monitor():
    """Display system status."""
    print(f"\n{BOLD}═══ System Monitor ═══{RESET}\n")

    # Health
    health = api_get("/health")
    if "error" in health:
        print(f"  {ERR_COLOR}✗ Cannot connect to API: {health['error']}{RESET}")
        return

    print(f"  {BOLD}Health{RESET}")
    for svc, status in health.get("services", {}).items():
        color = OK_COLOR if status == "healthy" else ERR_COLOR
        print(f"    {svc}: {color}{status}{RESET}")

    # Stats
    stats = api_get("/stats")
    if "counts" in stats:
        print(f"\n  {BOLD}Document Counts{RESET}")
        for backend, models in stats["counts"].items():
            print(f"    {DIM}{backend}{RESET}")
            for model, count in models.items():
                count_str = f"{count:,}" if isinstance(count, int) else str(count)
                print(f"      {model}: {BOLD}{count_str}{RESET}")

    # Personas
    personas = api_get("/personas")
    if isinstance(personas, list):
        print(f"\n  {BOLD}Personas{RESET}")
        for p in personas:
            color = PERSONA_COLORS.get(p["slug"], SYS_COLOR)
            print(f"    {color}● {p['name']}{RESET} ({DIM}{p['slug']}{RESET})")

    print()


# ── Chat ──────────────────────────────────────────────────
def cmd_chat(persona_slug: str = "heather_cox_richardson"):
    """Interactive one-on-one chat."""
    name = PERSONA_NAMES.get(persona_slug, persona_slug)
    color = PERSONA_COLORS.get(persona_slug, SYS_COLOR)
    short = PERSONA_SHORT.get(persona_slug, "?")

    print(f"\n{BOLD}Chatting with {color}{name}{RESET}")
    print(f"{DIM}Type 'quit' to exit, 'switch' to change persona{RESET}\n")

    while True:
        try:
            question = input(f"{USER_COLOR}{BOLD}You ▸{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "switch":
            persona_slug = (
                "nate_b_jones"
                if persona_slug == "heather_cox_richardson"
                else "heather_cox_richardson"
            )
            name = PERSONA_NAMES[persona_slug]
            color = PERSONA_COLORS[persona_slug]
            short = PERSONA_SHORT[persona_slug]
            print(f"\n{SYS_COLOR}Switched to {color}{name}{RESET}\n")
            continue

        # Stream response
        print(f"\n{color}{BOLD}{short} ▸{RESET} ", end="", flush=True)

        full_response = ""
        for event in api_post_stream("/chat", {
            "question": question,
            "persona": persona_slug,
            "stream": True,
        }):
            if event.get("type") == "token":
                token = event["content"]
                print(token, end="", flush=True)
                full_response += token
            elif event.get("type") == "error":
                print(f"\n{ERR_COLOR}Error: {event['message']}{RESET}")
            elif event.get("type") == "meta":
                pass  # Could show source info

        print("\n")


# ── Debate ────────────────────────────────────────────────
def cmd_debate():
    """Interactive debate mode."""
    print(f"\n{BOLD}═══ Moderated Debate ═══{RESET}")
    print(f"  {HCR_COLOR}● Heather Cox Richardson{RESET} vs {NBJ_COLOR}● Nate B Jones{RESET}")
    print(f"{DIM}Type 'quit' to exit{RESET}\n")

    while True:
        try:
            question = input(f"{USER_COLOR}{BOLD}Moderator ▸{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        current_persona = None
        for event in api_post_stream("/debate", {"question": question}):
            t = event.get("type")
            if t == "turn_start":
                persona = event["persona"]
                current_persona = persona
                color = PERSONA_COLORS.get(persona, SYS_COLOR)
                short = PERSONA_SHORT.get(persona, "?")
                name = PERSONA_NAMES.get(persona, persona)
                print(f"\n{color}{BOLD}{short} ({name}) ▸{RESET} ", end="", flush=True)
            elif t == "token":
                print(event["content"], end="", flush=True)
            elif t == "turn_end":
                print()
            elif t == "error":
                print(f"\n{ERR_COLOR}Error: {event['message']}{RESET}")

        print()


# ── Interactive Menu ──────────────────────────────────────
def cmd_interactive():
    """Main interactive menu."""
    print_banner()

    while True:
        print(f"{BOLD}Choose a mode:{RESET}")
        print(f"  {HCR_COLOR}[1]{RESET} Debate  — HCR vs NBJ")
        print(f"  {NBJ_COLOR}[2]{RESET} Chat    — One-on-one")
        print(f"  {SYS_COLOR}[3]{RESET} Monitor — System status")
        print(f"  {DIM}[q]{RESET} Quit")
        print()

        try:
            choice = input(f"{USER_COLOR}▸{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("1", "debate"):
            cmd_debate()
        elif choice in ("2", "chat"):
            # Choose persona
            print(f"\n  {HCR_COLOR}[1]{RESET} Heather Cox Richardson")
            print(f"  {NBJ_COLOR}[2]{RESET} Nate B Jones\n")
            try:
                p = input(f"{USER_COLOR}▸{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                continue
            slug = "nate_b_jones" if p in ("2", "nbj", "nate") else "heather_cox_richardson"
            cmd_chat(slug)
        elif choice in ("3", "monitor"):
            cmd_monitor()
        elif choice in ("q", "quit", "exit"):
            break
        else:
            print(f"{DIM}Unknown option{RESET}\n")


# ── CLI ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Longspear TUI — Terminal conversation interface",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["chat", "debate", "monitor"],
        help="Direct command (skip interactive menu)",
    )
    parser.add_argument(
        "persona",
        nargs="?",
        help="Persona slug for chat mode (hcr or nbj)",
    )
    parser.add_argument(
        "--api",
        default="http://localhost:28000",
        help="API base URL",
    )

    args = parser.parse_args()

    global API
    API = args.api

    if args.command == "chat":
        print_banner()
        slug = "heather_cox_richardson"
        if args.persona and args.persona.lower() in ("nbj", "nate", "nate_b_jones"):
            slug = "nate_b_jones"
        cmd_chat(slug)
    elif args.command == "debate":
        print_banner()
        cmd_debate()
    elif args.command == "monitor":
        cmd_monitor()
    else:
        cmd_interactive()


if __name__ == "__main__":
    main()
