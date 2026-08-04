#!/usr/bin/env python3
"""Runtime guard for the localized stage-opening chapter title card.

The test boots a blank cartridge save and follows the normal New Game flow.
It does not load a savestate or mutate emulated RAM.  The shared text-draw
helper must receive both the localized ``章节`` prefix (with its two leading
layout blanks, so it sits next to the fixed chapter number) and the exact
pointer / token stream generated for the first chapter title
(``桑吉巴尔级追击``).

Usage:
  .venv/bin/python test/live/test_stage_title_render.py ROM.nds [--out DIR]

Exit 0 = localized prefix and title observed, 1 = wrong payload,
2 = navigation failure.
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
sys.path.insert(0, str(REPO))
from utils.text_codec import decode  # noqa: E402


DRAW_HELPER = 0x02012EFC
FIRST_TITLE_SITE = 0x175560 + 0x20
FIRST_TITLE_ZH = "桑吉巴尔级追击"
PREFIX_PLACEMENT_OFFSET = 0x6A9
PREFIX_PTR = 0x0214B2DD
PREFIX_ZH = "章节"
PREFIX_LAYOUT_ZH = "{01}{01}章节"


def cycles(emu: Any, count: int) -> None:
    for _ in range(count):
        emu.cycle(False)


def press(emu: Any, name: str, *, hold: int = 4, settle: int = 180) -> None:
    mask = keymask(getattr(Keys, f"KEY_{name}"))
    emu.input.keypad_add_key(mask)
    cycles(emu, hold)
    emu.input.keypad_rm_key(mask)
    cycles(emu, settle)


def read_c_string(emu: Any, address: int, limit: int = 96) -> bytes:
    if not 0x02000000 <= address < 0x02400000:
        return b""
    out = bytearray()
    for index in range(limit):
        value = int(emu.memory.unsigned.read_byte(address + index))
        if value == 0:
            break
        out.append(value)
    return bytes(out)


def first_title_entry() -> dict[str, Any]:
    placement = json.loads(
        (REPO / "data/zh/placements/stage_titles.json").read_text()
    )
    matches = [
        entry
        for entry in placement["entries"]
        if FIRST_TITLE_SITE in {int(site, 0) for site in entry["sites"]}
    ]
    if len(matches) != 1 or matches[0]["zh"] != FIRST_TITLE_ZH:
        raise RuntimeError("first stage-title placement is missing or ambiguous")
    return matches[0]


def prefix_entry() -> dict[str, Any]:
    placement = json.loads(
        (REPO / "data/zh/placements/post_dict_labels.json").read_text()
    )
    matches = [
        entry
        for entry in placement["entries"]
        if int(entry["offset"], 0) == PREFIX_PLACEMENT_OFFSET
    ]
    if len(matches) != 1 or matches[0]["text"] != PREFIX_ZH:
        raise RuntimeError("stage-title prefix placement is missing or ambiguous")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rom", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rom = args.rom.resolve()
    if not rom.is_file():
        print(f"ROM not found: {rom}", file=sys.stderr)
        return 2
    out = (
        args.out.resolve()
        if args.out
        else Path(tempfile.mkdtemp(prefix="stage-title-render-"))
    )
    out.mkdir(parents=True, exist_ok=True)

    title = first_title_entry()
    prefix = prefix_entry()
    expected_title_ptr = int(title["ptr"], 0)
    expected_title_raw = bytes.fromhex(title["payload_hex"][:-2])
    expected_prefix_raw = bytes.fromhex(prefix["payload_hex"]).split(b"\0", 1)[0]
    if PREFIX_PTR != 0x02000000 + 0x14AC34 + PREFIX_PLACEMENT_OFFSET:
        raise AssertionError("stage-title prefix pointer constant drift")

    save = out / "blank.sav"
    save.write_bytes(b"\xFF" * (256 * 1024))

    observed_titles: list[dict[str, Any]] = []
    prefix_reads: list[dict[str, int]] = []
    emu = DeSmuME()
    try:
        emu.volume_set(0)
        emu.open(str(rom))
        if not emu.backup.import_file(str(save)):
            raise RuntimeError("py-desmume backup.import_file returned false")
        emu.reset()
        cycles(emu, 2000)

        def on_draw(_address: int, _size: int) -> None:
            registers = emu.memory.register_arm9
            pointer = int(registers.r3)
            if pointer != expected_title_ptr:
                return
            observed_titles.append(
                {
                    "x": int(registers.r1),
                    "y": int(registers.r2),
                    "ptr": pointer,
                    "raw": read_c_string(emu, pointer),
                }
            )

        def on_prefix_read(address: int, size: int) -> None:
            prefix_reads.append(
                {
                    "address": int(address),
                    "size": int(size),
                    "pc": int(emu.memory.register_arm9.pc),
                }
            )

        emu.memory.register_exec(DRAW_HELPER, on_draw)
        emu.memory.register_read(
            PREFIX_PTR,
            on_prefix_read,
            size=len(expected_prefix_raw) + 1,
        )

        press(emu, "START")
        for step in range(1, 7):
            press(emu, "A")
            if prefix_reads and observed_titles:
                # The helper is called while the card is composed on a hidden
                # BG.  Let the following fade expose it before capture.
                cycles(emu, 180)
                screenshot = out / "stage_title_scene.png"
                frame = emu.screenshot().convert("RGB")
                frame.save(screenshot)
                visual_ink = sum(
                    1
                    for y in range(320, 370)
                    for x in range(256)
                    if min(frame.getpixel((x, y))) >= 230
                    and (
                        max(frame.getpixel((x, y)))
                        - min(frame.getpixel((x, y)))
                    )
                    <= 12
                )
                title_call = observed_titles[-1]
                emu.memory.register_read(
                    PREFIX_PTR,
                    None,
                    size=len(expected_prefix_raw) + 1,
                )
                prefix_raw = read_c_string(emu, PREFIX_PTR)
                title_decoded = decode(
                    title_call["raw"], control_escapes=False
                )
                prefix_decoded = decode(
                    prefix_raw, control_escapes=False
                )
                report = {
                    "rom": str(rom),
                    "normal_flow": True,
                    "savestate_used": False,
                    "ram_mutation_used": False,
                    "step": step,
                    "prefix": {
                        "pointer": f"0x{PREFIX_PTR:08X}",
                        "raw_hex": prefix_raw.hex(),
                        "decoded": prefix_decoded,
                        "runtime_reads": [
                            {
                                "address": f"0x{read['address']:08X}",
                                "size": read["size"],
                                "pc": f"0x{read['pc']:08X}",
                            }
                            for read in prefix_reads
                        ],
                    },
                    "title": {
                        "pointer": f"0x{title_call['ptr']:08X}",
                        "coordinates": [title_call["x"], title_call["y"]],
                        "raw_hex": title_call["raw"].hex(),
                        "decoded": title_decoded,
                    },
                    "chapter_card_white_pixels": visual_ink,
                    "screenshot": str(screenshot),
                }
                (out / "report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
                )
                valid = (
                    prefix_raw == expected_prefix_raw
                    and prefix_decoded == PREFIX_LAYOUT_ZH
                    and bool(prefix_reads)
                    and title_call["raw"] == expected_title_raw
                    and title_decoded == FIRST_TITLE_ZH
                    and visual_ink >= 100
                )
                print(json.dumps(report, ensure_ascii=False, indent=2))
                if not valid:
                    print(
                        "localized prefix/title payload or visible card is wrong",
                        file=sys.stderr,
                    )
                    return 1
                print(
                    "PASS: localized 章节 prefix and first chapter title "
                    "observed in normal New Game flow"
                )
                return 0

        emu.screenshot().save(out / "navigation_failure.png")
        seen = {
            "prefix_reads": len(prefix_reads),
            "title_draws": len(observed_titles),
        }
        print(
            f"chapter title card was not fully observed: {seen}; "
            f"artifacts: {out}",
            file=sys.stderr,
        )
        return 2
    finally:
        emu.volume_set(0)
        try:
            emu.close()
        finally:
            emu.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
