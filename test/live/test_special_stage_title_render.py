#!/usr/bin/env python3
"""Cheat-assisted live guard for special chapter-code rendering.

The test keeps the supplied cartridge save and gameplay flow intact, but
temporarily replaces the save-slot prefix/title pointers in all 101 ARM9 stage
descriptors with one real translated record. This makes otherwise late SP/X/TR
codes observable on the load screen or battle terrain page without a
playthrough. No savestate is used and the mutation exists only in emulator RAM.

Usage:
  .venv/bin/python test/live/test_special_stage_title_render.py \
      ROM.nds SAVE.sav {load,battle} SP1A [--out DIR]

Exit 0 = exact target prefix/title reached the expected draw geometry,
1 = wrong payload/geometry, 2 = navigation failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# These must be set before importing py-desmume.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_RENDER_DRIVER", "software")

from desmume.controls import Keys, keymask  # noqa: E402
from desmume.emulator import DeSmuME  # noqa: E402

TEST_DIR = Path(__file__).resolve().parents[1]
REPO = TEST_DIR.parent

RAM_BASE = 0x02000000
DESC_BASE = RAM_BASE + 0x175560
DESC_COUNT = 101
DESC_STRIDE = 0x34
SAVE_PREFIX_FIELD = 0x0C
SAVE_TITLE_FIELD = 0x10
DRAW_HELPERS = (0x02012EFC, 0x02013C00)


def cycles(emu: Any, count: int) -> None:
    for _ in range(count):
        emu.cycle(False)


def press(emu: Any, name: str, *, hold: int = 8, settle: int = 120) -> None:
    mask = keymask(getattr(Keys, f"KEY_{name}"))
    emu.input.keypad_add_key(mask)
    cycles(emu, hold)
    emu.input.keypad_rm_key(mask)
    cycles(emu, settle)


def read_c_string(emu: Any, pointer: int, limit: int = 96) -> bytes:
    output = bytearray()
    for index in range(limit):
        value = int(emu.memory.unsigned.read_byte(pointer + index))
        if value == 0:
            break
        output.append(value)
    return bytes(output)


def target_pair(document: dict, prefix: str) -> tuple[dict, dict, int]:
    prefixes = [
        entry
        for entry in document["entries"]
        if entry["view"] == "save_slot"
        and entry["domain"] == "stage_title_prefix"
        and entry["zh"] == prefix
    ]
    if not prefixes:
        raise ValueError(f"prefix {prefix!r}: no translated entry")
    payloads = {entry["payload_hex"] for entry in prefixes}
    if len(payloads) != 1:
        raise ValueError(
            f"prefix {prefix!r}: duplicate entries disagree: {sorted(payloads)}"
        )
    prefix_entry = prefixes[0]
    record_id = int(prefix_entry["record_ids"][0])
    titles = [
        entry
        for entry in document["entries"]
        if entry["view"] == "save_slot"
        and entry["domain"] == "stage_title_stream"
        and record_id in {int(value) for value in entry["record_ids"]}
    ]
    if len(titles) != 1:
        raise ValueError(
            f"record {record_id}: expected one save-slot title, got {len(titles)}"
        )
    return prefix_entry, titles[0], record_id


def expected_geometry(mode: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if mode == "load":
        return (64, 32, 4), (48, 56, 26)
    return (56, 8, 4), (96, 8, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("save", type=Path)
    parser.add_argument("mode", choices=("load", "battle"))
    parser.add_argument("prefix")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rom = args.rom.resolve()
    save = args.save.resolve()
    if not rom.is_file() or not save.is_file():
        print(f"missing ROM/save: {rom} / {save}", file=sys.stderr)
        return 2
    out = (
        args.out.resolve()
        if args.out
        else Path(tempfile.mkdtemp(prefix="special-stage-title-"))
    )
    out.mkdir(parents=True, exist_ok=True)

    document = json.loads(
        (REPO / "data/zh/placements/stage_titles.json").read_text(
            encoding="utf-8"
        )
    )
    prefix, title, record_id = target_pair(document, args.prefix)
    prefix_ptr = int(prefix["ptr"], 0)
    title_ptr = int(title["ptr"], 0)
    expected_prefix = bytes.fromhex(prefix["payload_hex"])[:-1]
    expected_title = bytes.fromhex(title["payload_hex"])[:-1]
    calls: list[dict[str, Any]] = []

    emu = DeSmuME()
    try:
        emu.volume_set(0)
        emu.open(str(rom))
        if not emu.backup.import_file(str(save)):
            raise RuntimeError("py-desmume backup.import_file returned false")
        emu.reset()
        cycles(emu, 1800)

        # Cheat only the live descriptor pointers; the source ROM and cartridge
        # save remain untouched.
        for index in range(DESC_COUNT):
            record = DESC_BASE + index * DESC_STRIDE
            emu.memory.write_long(record + SAVE_PREFIX_FIELD, prefix_ptr)
            emu.memory.write_long(record + SAVE_TITLE_FIELD, title_ptr)

        def on_draw(_address: int, _size: int) -> None:
            registers = emu.memory.register_arm9
            pointer = int(registers.r3)
            if pointer not in (prefix_ptr, title_ptr):
                return
            stack = int(registers.sp)
            calls.append(
                {
                    "pointer": pointer,
                    "x": int(registers.r1),
                    "y": int(registers.r2),
                    "max_glyphs": int(emu.memory.signed.read_long(stack)),
                    "raw": read_c_string(emu, pointer),
                }
            )

        for helper in DRAW_HELPERS:
            emu.memory.register_exec(helper, on_draw)

        if args.mode == "load":
            press(emu, "START", settle=60)
            press(emu, "A", settle=120)
        else:
            for key, settle in (
                ("START", 180),
                ("DOWN", 80),
                ("A", 180),
                ("A", 100),
                ("UP", 60),
                ("A", 1500),
                ("LEFT", 180),
            ):
                press(emu, key, settle=settle)

        screenshot = out / f"{args.prefix}-{args.mode}.png"
        emu.screenshot().convert("RGB").save(screenshot)
        prefix_calls = [call for call in calls if call["pointer"] == prefix_ptr]
        title_calls = [call for call in calls if call["pointer"] == title_ptr]
        want_prefix_geometry, want_title_geometry = expected_geometry(args.mode)
        observed_prefix = next(
            (
                call
                for call in reversed(prefix_calls)
                if (call["x"], call["y"], call["max_glyphs"])
                == want_prefix_geometry
            ),
            None,
        )
        observed_title = next(
            (
                call
                for call in reversed(title_calls)
                if (call["x"], call["y"], call["max_glyphs"])
                == want_title_geometry
            ),
            None,
        )
        valid = (
            observed_prefix is not None
            and observed_title is not None
            and observed_prefix["raw"] == expected_prefix
            and observed_title["raw"] == expected_title
        )
        report = {
            "rom": str(rom),
            "save": str(save),
            "mode": args.mode,
            "cheat": {
                "kind": "RAM descriptor pointer substitution",
                "records_patched": DESC_COUNT,
                "target_record_id": record_id,
                "target_prefix": args.prefix,
                "prefix_ptr": f"0x{prefix_ptr:08X}",
                "title_ptr": f"0x{title_ptr:08X}",
            },
            "expected": {
                "prefix_raw": expected_prefix.hex(),
                "title_raw": expected_title.hex(),
                "prefix_geometry": want_prefix_geometry,
                "title_geometry": want_title_geometry,
                "title_zh": title["zh"],
            },
            "observed": {
                "prefix_raw": (
                    observed_prefix["raw"].hex() if observed_prefix else None
                ),
                "title_raw": (
                    observed_title["raw"].hex() if observed_title else None
                ),
            },
            "screenshot": str(screenshot),
            "savestate_used": False,
            "ram_mutation_used": True,
            "valid": valid,
        }
        (out / f"{args.prefix}-{args.mode}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if valid else 1
    finally:
        emu.volume_set(0)
        try:
            emu.close()
        finally:
            emu.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
