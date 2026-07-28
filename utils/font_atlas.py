"""Deterministic composition for optional glyph layers over the base atlas.

The committed ``data/font/atlas12.bin`` stays byte-identical to the shared
baseline.  Feature-specific glyph plans are applied in memory by every build
consumer, which lets independently developed branches add disjoint glyphs
without creating an unavoidable binary merge conflict.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


CELL_BYTES = 36
ATLAS_SLOTS = 4320
ATLAS_REL = Path("font/atlas12.bin")
TITLE_PLAN_REL = Path("font/stage_title_glyphs.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cell(atlas: bytes | bytearray, slot: int) -> bytes:
    start = slot * CELL_BYTES
    return bytes(atlas[start:start + CELL_BYTES])


def _payload(entry: dict, what: str) -> bytes:
    payload = bytes.fromhex(entry["cell_hex"])
    if len(payload) != CELL_BYTES:
        raise ValueError(
            f"{what} payload has {len(payload)} bytes, expected {CELL_BYTES}"
        )
    return payload


def _replace_cell(
    target: bytearray,
    *,
    slot: int,
    payload: bytes,
    old_sha256: str,
    what: str,
) -> None:
    """Apply one cell contract while accepting an already-composed input."""
    current = _cell(target, slot)
    if current == payload:
        return
    current_hash = sha256(current)
    if current_hash != old_sha256:
        raise ValueError(
            f"{what} target drifted at slot {slot}: "
            f"{current_hash} != {old_sha256}"
        )
    start = slot * CELL_BYTES
    target[start:start + CELL_BYTES] = payload


def compose_title_glyphs(source: bytes, plan: dict) -> bytes:
    """Return ``source`` with the chapter-title glyph layer composed over it.

    Contracts are asserted per touched cell rather than against one whole-file
    hash.  Therefore an atlas carrying unrelated changes from another branch is
    accepted, while any true collision in a title-owned cell is rejected.
    """
    if len(source) != ATLAS_SLOTS * CELL_BYTES:
        raise ValueError(
            f"atlas size {len(source)} != {ATLAS_SLOTS * CELL_BYTES}"
        )

    target = bytearray(source)

    # Moves are performed before their old ZH cells are reused by mints.
    for move in plan["moves"]:
        char = move["char"]
        from_slot = int(move["from_slot"])
        to_slot = int(move["to_slot"])
        payload = _payload(move, f"move {char!r}")
        if sha256(payload) != move["source_cell_sha256"]:
            raise ValueError(f"move payload hash drifted for {char!r}")
        if _cell(target, to_slot) != payload:
            source_cell = _cell(source, from_slot)
            if source_cell != payload:
                raise ValueError(f"move source drifted for {char!r}")
            _replace_cell(
                target,
                slot=to_slot,
                payload=payload,
                old_sha256=move["target_old_cell_sha256"],
                what=f"move {char!r}",
            )

    for mint in plan["mints"]:
        char = mint["char"]
        payload = _payload(mint, f"mint {char!r}")
        if sha256(payload) != mint["new_cell_sha256"]:
            raise ValueError(f"mint payload hash drifted for {char!r}")
        _replace_cell(
            target,
            slot=int(mint["slot"]),
            payload=payload,
            old_sha256=mint["old_cell_sha256"],
            what=f"mint {char!r}",
        )

    for promotion in plan["promotions"]:
        char = promotion["char"]
        source_slot = int(promotion["from_slot"])
        payload = _payload(promotion, f"promotion {char!r}")
        if sha256(payload) != promotion["source_cell_sha256"]:
            raise ValueError(f"promotion payload hash drifted for {char!r}")
        if _cell(target, int(promotion["slot"])) != payload:
            if _cell(source, source_slot) != payload:
                raise ValueError(f"promotion source drifted for {char!r}")
            _replace_cell(
                target,
                slot=int(promotion["slot"]),
                payload=payload,
                old_sha256=promotion["old_cell_sha256"],
                what=f"promotion {char!r}",
            )

    for restore in plan["restores"]:
        payload = _payload(restore, f"restore slot {restore['slot']}")
        if sha256(payload) != restore["new_cell_sha256"]:
            raise ValueError(
                f"restore payload hash drifted at slot {restore['slot']}"
            )
        _replace_cell(
            target,
            slot=int(restore["slot"]),
            payload=payload,
            old_sha256=restore["old_cell_sha256"],
            what="restore",
        )

    for reuse in plan.get("native_reuses", []):
        slot = int(reuse["slot"])
        if sha256(_cell(target, slot)) != reuse["cell_sha256"]:
            raise ValueError(
                f"native glyph reuse drifted for {reuse['char']!r} "
                f"at slot {slot}"
            )

    result = bytes(target)
    if sha256(source) == plan["source_atlas_sha256"]:
        result_hash = sha256(result)
        if result_hash != plan["target_atlas_sha256"]:
            raise ValueError(
                "stage-title glyph target atlas hash mismatch: "
                f"{result_hash} != {plan['target_atlas_sha256']}"
            )
    return result


def load_effective_atlas(data_dir: Path | str) -> bytes:
    """Load the base atlas and compose any checked-in chapter-title layer."""
    data_dir = Path(data_dir)
    source = (data_dir / ATLAS_REL).read_bytes()
    plan_path = data_dir / TITLE_PLAN_REL
    if not plan_path.exists():
        return source
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return compose_title_glyphs(source, plan)
