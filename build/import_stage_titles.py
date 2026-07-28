#!/usr/bin/env python3
"""Import the reviewed chapter-title TSVs into both UI title streams.

This is a canonical data importer, not a post-build ROM fixer. It verifies all
101 descriptor records against the Japanese ROM, encodes the stage-opening
renderA strings and the shared save-list/battle-header trampoline strings, then
emits one deduplicated high-bank fragment consumed by ``utils.arm9_layout``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils import rom, text_codec  # noqa: E402
from utils.extract import layout as L  # noqa: E402


RECORD_COUNT = L.STAGE_DESC_N
RECORD_STRIDE = L.STAGE_DESC_STRIDE
TABLE_FILE_OFFSET = L.STAGE_DESC
BANK_RAM = 0x023E7000
DEFAULT_FRAGMENT_OFFSET = 0x9AA0
MIN_PARENT_SIZE = 0xA3B4

DISPLAY_LIMITS = {
    "load_screen": {
        "title_x": 48,
        "right_boundary_x": 200,
        "engine_max_glyphs": 26,
        "safe_fullwidth_glyphs": 12,
    },
    "battle_terrain_page": {
        "title_x": 96,
        "right_boundary_x": 216,
        "engine_max_glyphs": 10,
        "safe_fullwidth_glyphs": 10,
    },
    "shared_save_slot_cap": 10,
}

DOMAIN_FIELDS = {
    ("save_slot", "stage_title_prefix"): L.STAGE_DESC_LABEL,
    ("save_slot", "stage_title_stream"): L.STAGE_DESC_TITLE,
    ("chapter_card", "stage_title_prefix"): L.STAGE_DESC_CARD_LABEL,
    ("chapter_card", "stage_title_stream"): L.STAGE_DESC_CARD_TITLE,
}
DOMAIN_COLUMNS = {
    "stage_title_prefix": ("prefix_text_ptr", "prefix_zh"),
    "stage_title_stream": ("title_text_ptr", "title_zh"),
}
VIEW_SURFACES = {
    "chapter_card": "stage",
    "save_slot": "bank",
}
RENDERB_CHARSET_PATH = REPO / "data/renderb_charset.json"

# The save/load list uses this generic fallback for free-battle slots instead
# of the six descriptor title pointers.  It is the same semantic title and can
# safely share the committed save-slot payload, but its three consumers must
# be retargeted explicitly.
FREE_BATTLE_FALLBACK = {
    "id": "LOAD_FREE_BATTLE_FALLBACK",
    "unique_text_id": "UTXT02048",
    "old_ptr": 0x0214B00B,
    "old_hex": "f33965fa1000",
    "sites": (0x32550, 0x3B098, 0x8540C),
}

# The reviewed corpus predates three established ref-project terminology and
# punctuation decisions. Keep source_zh and record each deliberate adjustment.
REF_ADJUSTMENTS = {
    "UTXT02036": ("那黑暗之名、乃是木星", "ref punctuation: ， -> 、"),
    "UTXT02037": ("扎比涅的叛乱", "ref terminology: 萨比尼 -> 扎比涅"),
    "UTXT02040": ("特别演习（预防者）", "ref terminology: 守护者 -> 预防者"),
}

# The save/load list is narrower and uses the trampoline path. Two compact,
# semantically equivalent labels avoid otherwise unshareable renderB/atlas
# glyphs; the full reviewed wording remains unchanged on the chapter card.
SAVE_SLOT_ADJUSTMENTS = {
    "UTXT01974": ("撼动宇宙", "save-slot compact wording: 震撼的宇宙 -> 撼动宇宙"),
    "UTXT01975": (
        "撼动宇宙 另一侧",
        "save-slot compact wording: 震撼的宇宙 -> 撼动宇宙",
    ),
    "UTXT02029": (
        "独眼高达 后篇",
        "save-slot localized suffix: After -> 后篇",
    ),
    "UTXT02036": (
        "那黑暗之名乃是木星",
        "save-slot compact punctuation: omit 、",
    ),
    "UTXT02041": (
        "特别演习（隆德贝尔）",
        "save-slot shared 10-cell title budget: omit middle dot",
    ),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0]


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _display_text(row: dict[str, str], view: str) -> tuple[str, list[str]]:
    text = row["zh_text"]
    notes: list[str] = []
    replacement = REF_ADJUSTMENTS.get(row["unique_text_id"])
    if replacement:
        text, note = replacement
        notes.append(note)
    if view == "save_slot" and row["unique_text_id"] in SAVE_SLOT_ADJUSTMENTS:
        text, note = SAVE_SLOT_ADJUSTMENTS[row["unique_text_id"]]
        notes.append(note)
    if " " in text:
        text = text.replace(" ", "{01}")
        notes.append("literal visual space -> {01} control cell")
    return text, notes


def _renderb_identity_slots() -> dict[str, int]:
    """Return the native renderB slots for the ASCII chapter-code alphabet."""
    document = json.loads(RENDERB_CHARSET_PATH.read_text(encoding="utf-8"))
    slots: dict[str, int] = {}
    for slot_text, entry in document["slots"].items():
        char = entry.get("char")
        if char is None or len(char) != 1 or not char.isascii():
            continue
        slot = int(slot_text)
        if char in slots:
            raise ValueError(
                f"renderB identity {char!r} is ambiguous: "
                f"slots {slots[char]} and {slot}"
            )
        slots[char] = slot
    required = set("0123456789ABFSPTLRUX")
    if not required <= slots.keys():
        raise ValueError(
            f"renderB chapter-code identities missing: {sorted(required - slots.keys())}"
        )
    return slots


def _encode_save_prefix(text: str) -> bytes:
    """Encode compact chapter codes with the font the stock field already uses.

    The load-list/battle-header prefix field is a renderB surface. Encoding a
    chapter code as injected 12 px ZH cells creates excess leading space and
    12 px tracking between every ASCII character. Four-character codes such as
    SP1A then run into the independently drawn title at x=96. The original
    renderB identities are already correct for these neutral characters, so
    retain them and use ZH-band glyphs only for localized 前/后 suffixes.
    """
    identities = _renderb_identity_slots()
    output = bytearray()
    for char in text:
        if char.isascii() and char.isalnum():
            try:
                slot = identities[char]
            except KeyError as exc:
                raise ValueError(
                    f"save-slot chapter prefix has no renderB identity for {char!r}"
                ) from exc
            output += (
                bytes((slot,))
                if slot < text_codec.TWO_BYTE_SLOT_OFFSET
                else text_codec.encode_slot(slot)
            )
        else:
            output += text_codec.encode(
                char,
                surface="bank",
                allow_low15=True,
            )
    return bytes(output)


def build_document(
    unique_path: Path,
    records_path: Path,
    source_rom: Path,
    fragment_offset: int,
) -> dict:
    unique_raw = unique_path.read_bytes()
    records_raw = records_path.read_bytes()
    unique = _read_tsv(unique_path)
    records = _read_tsv(records_path)
    if len(unique) != 178:
        raise ValueError(f"unique title row count {len(unique)} != 178")
    domains = Counter(row["domain"] for row in unique)
    if domains != {"stage_title_prefix": 89, "stage_title_stream": 89}:
        raise ValueError(f"unexpected title domains: {dict(domains)}")
    if len(records) != RECORD_COUNT:
        raise ValueError(
            f"record title row count {len(records)} != {RECORD_COUNT}"
        )

    game = rom.load_rom(source_rom)
    arm9 = bytes(game.arm9)
    table = arm9[
        TABLE_FILE_OFFSET:TABLE_FILE_OFFSET + RECORD_COUNT * RECORD_STRIDE
    ]
    if len(table) != RECORD_COUNT * RECORD_STRIDE:
        raise ValueError("source ARM9 stage descriptor table is truncated")

    record_by_id = {int(row["record_id"]): row for row in records}
    if set(record_by_id) != set(range(RECORD_COUNT)):
        raise ValueError("record TSV must contain record_id 0..100 exactly once")
    unique_by_key = {
        (row["domain"], int(row["text_ptr"], 0)): row
        for row in unique
    }
    if len(unique_by_key) != len(unique):
        raise ValueError("unique TSV repeats a domain/text_ptr key")

    grouped: dict[
        tuple[str, str, int], dict[str, object]
    ] = {}
    canonical_keys_seen: set[tuple[str, int]] = set()
    for record_id in range(RECORD_COUNT):
        source = record_by_id[record_id]
        base = TABLE_FILE_OFFSET + record_id * RECORD_STRIDE
        file_ptr = _u32(arm9, base + L.STAGE_DESC_FILE)
        file_off = file_ptr - L.RAM_BASE
        file_end = arm9.find(b"\x00", file_off)
        file_name = arm9[file_off:file_end].decode("ascii")
        if file_name != source["group"]:
            raise ValueError(
                f"record {record_id}: source file {file_name!r} "
                f"!= TSV {source['group']!r}"
            )

        for domain, (pointer_column, zh_column) in DOMAIN_COLUMNS.items():
            canonical_ptr = int(source[pointer_column], 0)
            canonical_field = DOMAIN_FIELDS[("chapter_card", domain)]
            if _u32(arm9, base + canonical_field) != canonical_ptr:
                raise ValueError(
                    f"record {record_id} {domain}: chapter-card pointer "
                    "does not match the reviewed TSV"
                )
            canonical_key = (domain, canonical_ptr)
            row = unique_by_key.get(canonical_key)
            if row is None or row["zh_text"] != source[zh_column]:
                raise ValueError(
                    f"record {record_id} {domain}: unique/record TSV mismatch"
                )
            canonical_keys_seen.add(canonical_key)

            for view in ("chapter_card", "save_slot"):
                field = DOMAIN_FIELDS[(view, domain)]
                old_ptr = _u32(arm9, base + field)
                key = (view, domain, old_ptr)
                group = grouped.setdefault(
                    key,
                    {
                        "row": row,
                        "sites": [],
                        "record_ids": [],
                    },
                )
                if group["row"]["unique_text_id"] != row["unique_text_id"]:
                    raise ValueError(
                        f"{view} pointer {old_ptr:#010x} maps to conflicting "
                        "reviewed translations"
                    )
                group["sites"].append(base + field)
                group["record_ids"].append(record_id)

    if canonical_keys_seen != set(unique_by_key):
        missing = sorted(set(unique_by_key) - canonical_keys_seen)
        extra = sorted(canonical_keys_seen - set(unique_by_key))
        raise ValueError(
            f"TSV/table key mismatch: missing={missing[:3]} extra={extra[:3]}"
        )

    payload_offsets: dict[bytes, int] = {}
    cursor = fragment_offset
    plans: list[dict] = []

    def add_group(view: str, domain: str, old_ptr: int, group: dict) -> None:
        nonlocal cursor
        row = group["row"]
        zh, adjustments = _display_text(row, view)
        surface = VIEW_SURFACES[view]
        renderb_identity = (
            view == "save_slot"
            and domain == "stage_title_prefix"
        )
        if renderb_identity:
            encoded = _encode_save_prefix(zh)
            adjustments.append(
                "save-list chapter code uses native renderB identities"
            )
        else:
            encoded = text_codec.encode(
                zh,
                surface=surface,
                allow_low15=True,
            )
        payload = encoded + b"\x00"
        if payload not in payload_offsets:
            payload_offsets[payload] = cursor
            cursor += len(payload)
        off = payload_offsets[payload]
        entry_id = row["unique_text_id"]
        if view == "save_slot":
            entry_id = f"SLOT_{entry_id}"
        plans.append(
            {
                "id": entry_id,
                "view": view,
                "domain": domain,
                "surface": surface,
                "old_ptr": f"0x{old_ptr:08X}",
                "sites": [f"0x{site:X}" for site in group["sites"]],
                "record_ids": group["record_ids"],
                "jp": row["jp_texts_full"],
                "source_zh": row["zh_text"],
                "zh": zh,
                "adjustments": adjustments,
                "encoding": (
                    "renderb_identity+zh"
                    if renderb_identity
                    else "text_codec"
                ),
                "offset": f"0x{off:X}",
                "ptr": f"0x{BANK_RAM + off:08X}",
                "payload_hex": payload.hex(),
            }
        )

    # Preserve the reviewed TSV order for the chapter card, then mirror that
    # order for the save/load slot stream.
    for view in ("chapter_card", "save_slot"):
        for row in unique:
            domain = row["domain"]
            matches = [
                (old_ptr, group)
                for (candidate_view, candidate_domain, old_ptr), group
                in grouped.items()
                if candidate_view == view
                and candidate_domain == domain
                and group["row"]["unique_text_id"] == row["unique_text_id"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{view} {row['unique_text_id']}: expected one pointer group, "
                    f"found {len(matches)}"
                )
            add_group(view, domain, matches[0][0], matches[0][1])

    fallback_row = next(
        row
        for row in unique
        if row["unique_text_id"] == FREE_BATTLE_FALLBACK["unique_text_id"]
        and row["domain"] == "stage_title_stream"
    )
    fallback_zh, fallback_adjustments = _display_text(
        fallback_row, "save_slot"
    )
    fallback_payload = (
        text_codec.encode(
            fallback_zh,
            surface="bank",
            allow_low15=True,
        )
        + b"\x00"
    )
    if fallback_payload not in payload_offsets:
        raise AssertionError(
            "free-battle fallback did not deduplicate with a save-slot title"
        )
    fallback_old_ptr = int(FREE_BATTLE_FALLBACK["old_ptr"])
    fallback_old_off = fallback_old_ptr - L.RAM_BASE
    fallback_old = bytes.fromhex(FREE_BATTLE_FALLBACK["old_hex"])
    if arm9[
        fallback_old_off:fallback_old_off + len(fallback_old)
    ] != fallback_old:
        raise ValueError("free-battle fallback source string drift")
    for site in FREE_BATTLE_FALLBACK["sites"]:
        if _u32(arm9, site) != fallback_old_ptr:
            raise ValueError(
                f"free-battle fallback site {site:#x} source pointer drift"
            )
    fallback_off = payload_offsets[fallback_payload]
    extra_pointer_entries = [
        {
            "id": FREE_BATTLE_FALLBACK["id"],
            "view": "save_slot",
            "domain": "stage_title_stream",
            "surface": "bank",
            "old_ptr": f"0x{fallback_old_ptr:08X}",
            "sites": [
                f"0x{site:X}" for site in FREE_BATTLE_FALLBACK["sites"]
            ],
            "jp": fallback_row["jp_texts_full"],
            "source_zh": fallback_row["zh_text"],
            "zh": fallback_zh,
            "adjustments": fallback_adjustments,
            "offset": f"0x{fallback_off:X}",
            "ptr": f"0x{BANK_RAM + fallback_off:08X}",
            "payload_hex": fallback_payload.hex(),
        }
    ]

    final_size = max(_align4(cursor), MIN_PARENT_SIZE)
    pointer_sites = sum(len(entry["sites"]) for entry in plans)
    if len(plans) != 356 or pointer_sites != RECORD_COUNT * 4:
        raise AssertionError(
            f"title ownership mismatch: entries={len(plans)}, sites={pointer_sites}"
        )
    over_budget = []
    for entry in plans:
        if (
            entry["view"] != "save_slot"
            or entry["domain"] != "stage_title_stream"
        ):
            continue
        payload = bytes.fromhex(entry["payload_hex"])[:-1]
        glyphs = sum(
            token != 0x01
            for _offset, token, _length in text_codec.iter_tokens(payload)
        )
        if glyphs > DISPLAY_LIMITS["shared_save_slot_cap"]:
            over_budget.append((entry["id"], glyphs, entry["zh"]))
    if over_budget:
        raise ValueError(
            "save-slot titles exceed the shared load/battle cap: "
            f"{over_budget}"
        )

    return {
        "_about": (
            "Reviewed chapter-title translations for both title streams in the "
            "101-record ARM9 stage descriptor table: chapter-card renderA fields "
            "(+0x1C/+0x20) and shared save-list/battle-header trampoline fields "
            "(+0x0C/+0x10). "
            "All save-list chapter-code prefixes retain the stock renderB "
            "ASCII identities and use ZH-band glyphs only for localized "
            "suffixes, keeping SP/X/TR/TU/TL/FB codes inside the same fixed "
            "battle-header field as numeric codes. "
            "All strings are NUL-terminated and deduplicated in one high-bank "
            "fragment; every pointer write is source-asserted by the builder. "
            "The generic free-battle fallback used by the save/load list shares "
            "the same translated payload through three separately declared "
            "consumer pointers."
        ),
        "source": {
            "unique_tsv": unique_path.name,
            "unique_tsv_sha256": _sha256(unique_raw),
            "record_tsv": records_path.name,
            "record_tsv_sha256": _sha256(records_raw),
            "source_rom_sha1": hashlib.sha1(source_rom.read_bytes()).hexdigest(),
            "source_arm9_sha256": _sha256(arm9),
            "source_table_sha256": _sha256(table),
        },
        "table": {
            "ram_base": f"0x{L.RAM_BASE + TABLE_FILE_OFFSET:08X}",
            "file_offset": f"0x{TABLE_FILE_OFFSET:X}",
            "record_count": RECORD_COUNT,
            "stride": f"0x{RECORD_STRIDE:X}",
            "save_slot_prefix_ptr_offset": f"0x{L.STAGE_DESC_LABEL:X}",
            "save_slot_title_ptr_offset": f"0x{L.STAGE_DESC_TITLE:X}",
            "chapter_card_prefix_ptr_offset": f"0x{L.STAGE_DESC_CARD_LABEL:X}",
            "chapter_card_title_ptr_offset": f"0x{L.STAGE_DESC_CARD_TITLE:X}",
        },
        "bank": {
            "parent": "zh/placements/briefing_blobs.json",
            "ram_base": f"0x{BANK_RAM:08X}",
            "fragment_offset": f"0x{fragment_offset:X}",
            "fragment_end": f"0x{cursor:X}",
            "required_parent_size": f"0x{final_size:X}",
        },
        "display_limits": DISPLAY_LIMITS,
        "counts": {
            "records": RECORD_COUNT,
            "entries": len(plans),
            "chapter_card_entries": sum(
                entry["view"] == "chapter_card" for entry in plans
            ),
            "save_slot_entries": sum(
                entry["view"] == "save_slot" for entry in plans
            ),
            "pointer_sites": pointer_sites,
            "extra_pointer_entries": len(extra_pointer_entries),
            "extra_pointer_sites": sum(
                len(entry["sites"]) for entry in extra_pointer_entries
            ),
            "unique_payloads": len(payload_offsets),
            "payload_bytes": sum(len(payload) for payload in payload_offsets),
        },
        "entries": plans,
        "extra_pointer_entries": extra_pointer_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-tsv", required=True, type=Path)
    parser.add_argument("--records-tsv", required=True, type=Path)
    parser.add_argument("--source-rom", required=True, type=Path)
    parser.add_argument(
        "--fragment-offset",
        default=hex(DEFAULT_FRAGMENT_OFFSET),
        type=lambda value: int(value, 0),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data/zh/placements/stage_titles.json",
    )
    args = parser.parse_args()
    document = build_document(
        args.unique_tsv.resolve(),
        args.records_tsv.resolve(),
        args.source_rom.resolve(),
        args.fragment_offset,
    )
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    counts = document["counts"]
    print(
        f"wrote {args.output}: {counts['entries']} entries / "
        f"{counts['unique_payloads']} payloads / "
        f"{counts['pointer_sites']} pointer sites; "
        f"bank size {document['bank']['required_parent_size']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
