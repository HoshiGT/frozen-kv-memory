#!/usr/bin/env python3
"""Turn Claude session transcripts into long-form dialogue text.

Real conversations are the target workload for context compression, and they
carry exactly the mix we care about: derivable reasoning plus hard facts
(paths, numbers, decisions) that no prior can reconstruct.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"
OUT = Path(__file__).resolve().parent / "data"
MIN_CHARS = 8000


def blocks_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "text":
            out.append(b.get("text", ""))
    return "\n".join(out)


def read_session(path: Path) -> str:
    turns = []
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = d.get("type")
        if kind not in ("user", "assistant"):
            continue
        text = blocks_to_text(d.get("message", {}).get("content")).strip()
        if not text or text.startswith("<"):
            continue
        who = "Hoshi" if kind == "user" else "Chroma"
        turns.append(f"{who}: {text}")
    return "\n\n".join(turns)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    kept = []
    for p in sorted(PROJECTS.glob("*/*.jsonl")):
        text = read_session(p)
        if len(text) < MIN_CHARS:
            continue
        dest = OUT / f"{p.parent.name[-24:]}_{p.stem[:8]}.txt"
        dest.write_text(text, encoding="utf-8")
        kept.append((len(text), dest.name))

    kept.sort(reverse=True)
    total = sum(n for n, _ in kept)
    print(f"{len(kept)} sessions, {total/1e6:.2f}M chars -> {OUT}")
    for n, name in kept[:10]:
        print(f"  {n/1000:8.1f}k  {name}")


if __name__ == "__main__":
    main()
