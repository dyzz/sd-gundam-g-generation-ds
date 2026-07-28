#!/usr/bin/env python3
"""Verify or materialize the composable chapter-title glyph layer.

The tracked atlas remains the shared baseline.  The normal ROM build composes
this plan in memory; this helper verifies the same contracts or writes the
effective atlas to an explicit output path for inspection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import font_atlas  # noqa: E402

ATLAS_PATH = REPO / "data/font/atlas12.bin"
CHARMAP_PATH = REPO / "data/charmap.json"
PLAN_PATH = REPO / "data/font/stage_title_glyphs.json"
CELL_BYTES = font_atlas.CELL_BYTES
ATLAS_SLOTS = font_atlas.ATLAS_SLOTS
sha256 = font_atlas.sha256


def _verify_charmap(plan: dict) -> None:
    charmap = json.loads(CHARMAP_PATH.read_text(encoding="utf-8"))
    mapping = {char: int(slot) for char, slot in charmap["two_byte_zh"].items()}
    expected = {
        entry["char"]: int(entry["to_slot"])
        for entry in plan["moves"]
    }
    expected.update(
        {entry["char"]: int(entry["slot"]) for entry in plan["mints"]}
    )
    expected.update(
        {entry["char"]: int(entry["to_slot"]) for entry in plan["remaps"]}
    )
    expected.update(
        {entry["char"]: int(entry["slot"]) for entry in plan["promotions"]}
    )
    expected.update(
        {entry["char"]: int(entry["slot"])
         for entry in plan.get("native_reuses", [])}
    )
    bad = {
        char: {"mapping": mapping.get(char), "expected": slot}
        for char, slot in expected.items()
        if mapping.get(char) != slot
    }
    if bad:
        raise ValueError(f"stage-title charmap/identity drift: {bad}")
    slots = list(mapping.values())
    if len(slots) != len(set(slots)):
        raise ValueError("two_byte_zh contains duplicate glyph slots")


def _verify_move_scope(plan: dict) -> None:
    """Keep JP-band recipients confined to their declared renderA sources."""
    root = REPO / "data" / "zh"
    excluded = {CHARMAP_PATH.resolve(), PLAN_PATH.resolve()}
    for move in plan["moves"] + plan["remaps"]:
        char = move["char"]
        allowed = set(move["allowed_text_sources"])
        hits: set[str] = set()
        for path in root.rglob("*.json"):
            if path.resolve() in excluded or (root / "font") in path.parents:
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            def walk(value) -> None:
                if isinstance(value, dict):
                    for child in value.values():
                        walk(child)
                elif isinstance(value, list):
                    for child in value:
                        walk(child)
                elif isinstance(value, str) and char in value:
                    hits.add(str(path.relative_to(REPO)))

            walk(document)
        if not hits or not hits <= allowed:
            raise ValueError(
                f"moved glyph {char!r} escaped renderA-only scope: "
                f"hits={sorted(hits)}, allowed={sorted(allowed)}"
            )


build_target = font_atlas.compose_title_glyphs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--atlas", type=Path, default=ATLAS_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for a materialized effective atlas",
    )
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    _verify_charmap(plan)
    _verify_move_scope(plan)
    atlas_path = args.atlas.resolve()
    current = atlas_path.read_bytes()
    target = build_target(current, plan)
    if args.check:
        print(
            f"stage-title glyph plan verified: {len(plan['moves'])} moves, "
            f"{len(plan['remaps'])} remaps, {len(plan['mints'])} mints, "
            f"{len(plan['promotions'])} promotions, "
            f"{len(plan.get('native_reuses', []))} native reuses, "
            f"effective sha256 {sha256(target)}"
        )
        return 0
    if args.output is None:
        print(f"effective atlas sha256 {sha256(target)}")
        return 0
    output_path = args.output.resolve()
    output_path.write_bytes(target)
    print(f"wrote {output_path}: sha256 {sha256(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
