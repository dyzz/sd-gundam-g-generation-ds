"""Deterministic writer for the shared save/load screen resource (c34.bin).

``c34.bin`` is a small graphics container rather than a flat bitmap:

* pointer 0 selects one custom-LZSS-compressed 4bpp tile set;
* pointer 1 selects four 16-colour palettes;
* pointers 2..21 select the full save/load screen and its overlay layouts.

The Japanese labels share tiles across the three save slots and the save/load
header variants.  Replacing a source tile in place therefore leaks pixels into
other labels.  This writer renders every layout to visible pixels, repaints the
declared Chinese surfaces from the committed 12x12 atlas, deduplicates all
resulting cells, and rewrites the layouts and compressed tile set atomically
inside the original file size.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

from . import battle_system_graphics, static_graphics


TILE_BYTES = 32


def _int(value: Any, *, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where}: expected integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError as exc:
            raise ValueError(f"{where}: invalid integer {value!r}") from exc
    raise ValueError(f"{where}: expected integer, got {value!r}")


def _hash(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hash(
    actual: bytes | bytearray, expected: str, *, where: str
) -> None:
    got = _hash(actual)
    if got != expected:
        raise ValueError(f"{where}: sha256 {got} != {expected}")


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class Container:
    pointer_table_offset: int
    pointers: tuple[int, ...]
    graphics_offset: int
    palette_offset: int
    layout_start: int
    layout_end: int
    first_layout: int
    last_layout: int
    compressed_capacity: int
    compressed_size: int
    tile_capacity: int
    graphics: bytes
    layouts: dict[int, Layout]


def _get_nibble(tile: bytes, x: int, y: int) -> int:
    packed = tile[y * 4 + x // 2]
    return (packed >> (4 if x & 1 else 0)) & 0xF


def _tile_from_pixels(
    pixels: list[list[int]], cell_x: int, cell_y: int
) -> bytes:
    tile = bytearray(TILE_BYTES)
    for y in range(8):
        for x in range(8):
            value = pixels[cell_y * 8 + y][cell_x * 8 + x] & 0xF
            offset = y * 4 + x // 2
            tile[offset] |= value << (4 if x & 1 else 0)
    return bytes(tile)


def _render_layout(
    graphics: bytes, layout: Layout
) -> list[list[int]]:
    pixels = [
        [0] * (layout.width * 8) for _ in range(layout.height * 8)
    ]
    tile_count = len(graphics) // TILE_BYTES
    for cell, value in enumerate(layout.values):
        tile_index = value & 0x3FF
        if tile_index >= tile_count:
            raise ValueError(
                f"layout references tile {tile_index}; only {tile_count} exist"
            )
        tile = graphics[
            tile_index * TILE_BYTES:(tile_index + 1) * TILE_BYTES
        ]
        horizontal = bool(value & 0x0400)
        vertical = bool(value & 0x0800)
        origin_x = (cell % layout.width) * 8
        origin_y = (cell // layout.width) * 8
        for y in range(8):
            source_y = 7 - y if vertical else y
            for x in range(8):
                source_x = 7 - x if horizontal else x
                pixels[origin_y + y][origin_x + x] = _get_nibble(
                    tile, source_x, source_y
                )
    return pixels


def _decode_container(
    data: bytes,
    table: dict,
    *,
    verify_source: bool,
) -> Container:
    contract = table["source"]
    expected_size = _int(contract["size"], where="source.size")
    if len(data) != expected_size:
        raise ValueError(
            f"{table['file']}: size {len(data)} != {expected_size}"
        )
    if verify_source:
        _require_hash(
            data,
            contract["sha256"],
            where=f"{table['file']} source",
        )

    pointer_table_offset = _int(
        contract["pointer_table_offset"],
        where="source.pointer_table_offset",
    )
    root_pointer = struct.unpack_from("<I", data, 0)[0]
    if root_pointer != pointer_table_offset:
        raise ValueError(
            f"{table['file']}: root pointer {root_pointer:#x} != "
            f"{pointer_table_offset:#x}"
        )
    pointer_count = _int(
        contract["pointer_count"], where="source.pointer_count"
    )
    pointer_end = pointer_table_offset + pointer_count * 4
    if pointer_end != len(data):
        raise ValueError(
            f"{table['file']}: pointer table does not end at EOF"
        )
    pointer_bytes = data[pointer_table_offset:pointer_end]
    if verify_source:
        _require_hash(
            pointer_bytes,
            contract["pointer_table_sha256"],
            where=f"{table['file']} source pointer table",
        )
    pointers = struct.unpack_from(
        f"<{pointer_count}I", data, pointer_table_offset
    )

    graphics_offset = _int(
        contract["graphics_offset"], where="source.graphics_offset"
    )
    palette_offset = _int(
        contract["palette_offset"], where="source.palette_offset"
    )
    layout_start = _int(
        contract["layout_start"], where="source.layout_start"
    )
    layout_end = _int(contract["layout_end"], where="source.layout_end")
    first_layout = _int(
        contract["first_layout"], where="source.first_layout"
    )
    last_layout = _int(
        contract["last_layout"], where="source.last_layout"
    )
    if not (
        0 <= graphics_offset < palette_offset <= layout_start
        < layout_end == pointer_table_offset < len(data)
    ):
        raise ValueError(f"{table['file']}: invalid region ordering")
    if not (
        0 <= first_layout <= last_layout < pointer_count
        and pointers[0] == graphics_offset
        and pointers[1] == palette_offset
        and pointers[first_layout] == layout_start
    ):
        raise ValueError(f"{table['file']}: pointer topology drift")

    descriptor = struct.unpack_from("<I", data, graphics_offset)[0]
    if not descriptor & 0x80000000:
        raise ValueError(
            f"{table['file']}: graphics descriptor is not compressed"
        )
    if verify_source:
        expected_descriptor = _int(
            contract["graphics_descriptor"],
            where="source.graphics_descriptor",
        )
        if descriptor != expected_descriptor:
            raise ValueError(
                f"{table['file']}: graphics descriptor {descriptor:#x} != "
                f"{expected_descriptor:#x}"
            )
        _require_hash(
            data[graphics_offset:palette_offset],
            contract["graphics_region_sha256"],
            where=f"{table['file']} source graphics region",
        )
        _require_hash(
            data[palette_offset:layout_start],
            contract["palette_region_sha256"],
            where=f"{table['file']} source palette region",
        )
        _require_hash(
            data[layout_start:layout_end],
            contract["layout_region_sha256"],
            where=f"{table['file']} source layout region",
        )

    compressed_capacity = palette_offset - (graphics_offset + 4)
    compressed_size = descriptor & 0xFFFF
    if compressed_size > compressed_capacity:
        raise ValueError(
            f"{table['file']}: compressed payload {compressed_size} exceeds "
            f"capacity {compressed_capacity}"
        )
    graphics = battle_system_graphics.decompress_lzss(
        data[
            graphics_offset + 4:
            graphics_offset + 4 + compressed_size
        ]
    )
    expected_decoded_size = _int(
        contract["decoded_size"], where="source.decoded_size"
    )
    if len(graphics) != expected_decoded_size or len(graphics) % TILE_BYTES:
        raise ValueError(
            f"{table['file']}: decoded size {len(graphics)} != "
            f"{expected_decoded_size}"
        )
    if verify_source:
        _require_hash(
            graphics,
            contract["decoded_sha256"],
            where=f"{table['file']} source decoded graphics",
        )
    tile_capacity = _int(
        contract["tile_capacity"], where="source.tile_capacity"
    )
    if len(graphics) != tile_capacity * TILE_BYTES:
        raise ValueError(f"{table['file']}: tile capacity drift")

    layouts: dict[int, Layout] = {}
    for index in range(first_layout, last_layout + 1):
        offset = pointers[index]
        end = (
            pointers[index + 1]
            if index < last_layout
            else layout_end
        )
        if not layout_start <= offset < end <= layout_end:
            raise ValueError(
                f"{table['file']}: layout {index} pointer bounds drift"
            )
        width, height, mode = struct.unpack_from("<BBH", data, offset)
        if not width or not height or mode != 0:
            raise ValueError(
                f"{table['file']}: layout {index} has unsupported "
                f"{width}x{height} mode {mode}"
            )
        count = width * height
        if offset + 4 + count * 2 != end:
            raise ValueError(
                f"{table['file']}: layout {index} payload-size drift"
            )
        values = struct.unpack_from(f"<{count}H", data, offset + 4)
        bad = sorted(
            {value & 0x3FF for value in values}
            - set(range(tile_capacity))
        )
        if bad:
            raise ValueError(
                f"{table['file']}: layout {index} references missing "
                f"tile(s) {bad}"
            )
        layouts[index] = Layout(width, height, values)

    return Container(
        pointer_table_offset,
        tuple(pointers),
        graphics_offset,
        palette_offset,
        layout_start,
        layout_end,
        first_layout,
        last_layout,
        compressed_capacity,
        compressed_size,
        tile_capacity,
        graphics,
        layouts,
    )


def _label_box(
    label: dict, *, width: int, height: int, where: str
) -> tuple[int, int, int, int]:
    box = label["box"]
    x0 = _int(box["x"], where=f"{where}.box.x")
    y0 = _int(box["y"], where=f"{where}.box.y")
    box_width = _int(box["width"], where=f"{where}.box.width")
    box_height = _int(box["height"], where=f"{where}.box.height")
    x1 = x0 + box_width
    y1 = y0 + box_height
    if (
        box_width <= 0
        or box_height <= 0
        or x0 < 0
        or y0 < 0
        or x1 > width
        or y1 > height
    ):
        raise ValueError(f"{where}: box {(x0, y0, x1, y1)} is out of bounds")
    return x0, y0, x1, y1


def _draw_label(
    canvas: list[list[int]],
    label: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
    where: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _label_box(
        label,
        width=len(canvas[0]),
        height=len(canvas),
        where=where,
    )
    background = _int(label["background"], where=f"{where}.background")
    palette = label["palette_indices"]
    stroke = _int(palette["stroke"], where=f"{where}.palette_indices.stroke")
    shadow = _int(palette["shadow"], where=f"{where}.palette_indices.shadow")
    if any(not 0 <= value <= 15 for value in (background, stroke, shadow)):
        raise ValueError(f"{where}: palette indices must be in 0..15")
    for y in range(y0, y1):
        canvas[y][x0:x1] = [background] * (x1 - x0)

    text = label["text"]
    advance = _int(
        label.get("advance", static_graphics.ATLAS_CELL),
        where=f"{where}.advance",
    )
    if not 1 <= advance <= static_graphics.ATLAS_CELL:
        raise ValueError(
            f"{where}: advance {advance} must be in "
            f"1..{static_graphics.ATLAS_CELL}"
        )
    text_width = (
        static_graphics.ATLAS_CELL
        + max(0, len(text) - 1) * advance
    )
    text_height = static_graphics.ATLAS_CELL
    if text_width > x1 - x0 or text_height > y1 - y0:
        raise ValueError(f"{where}: {text!r} does not fit its box")
    target_x = x0 + (x1 - x0 - text_width) // 2
    target_y = y0 + (y1 - y0 - text_height) // 2
    for char in text:
        try:
            slot = int(char_slots[char])
        except KeyError as exc:
            raise ValueError(
                f"{where}: no committed atlas slot for {char!r}"
            ) from exc
        glyph = static_graphics.atlas_cell(atlas, slot)
        for glyph_y, row in enumerate(glyph):
            for glyph_x, value in enumerate(row):
                if value == 1:
                    canvas[target_y + glyph_y][target_x + glyph_x] = stroke
                elif value == 2:
                    canvas[target_y + glyph_y][target_x + glyph_x] = shadow
                elif value != 0:
                    raise ValueError(
                        f"{where}: atlas slot {slot} has value {value}"
                    )
        target_x += advance
    return x0, y0, x1, y1


def resource_facts(data: bytes, table: dict) -> dict[str, int | str]:
    """Validate one rebuilt container and return its fixed-capacity facts."""
    container = _decode_container(data, table, verify_source=False)
    referenced = {
        value & 0x3FF
        for layout in container.layouts.values()
        for value in layout.values
    }
    return {
        "used_tiles": len(referenced),
        "tile_capacity": container.tile_capacity,
        "compressed_size": container.compressed_size,
        "compressed_capacity": container.compressed_capacity,
        "graphics_sha256": _hash(container.graphics),
        "layouts_sha256": _hash(
            data[container.layout_start:container.layout_end]
        ),
        "file_sha256": _hash(data),
    }


def repaint_save_load(
    source: bytes,
    table: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
) -> bytes:
    """Rebuild ``c34.bin`` from its Japanese source and semantic label spec."""
    if table.get("format") != "save_load_graphics":
        raise ValueError(
            f"{table.get('file', '<unknown>')}: expected save_load_graphics"
        )
    container = _decode_container(source, table, verify_source=True)
    stock = {
        index: _render_layout(container.graphics, layout)
        for index, layout in container.layouts.items()
    }
    desired = {
        index: [row[:] for row in pixels]
        for index, pixels in stock.items()
    }
    changed_boxes: dict[int, list[tuple[int, int, int, int]]] = {
        index: [] for index in container.layouts
    }
    palette_overrides: dict[int, dict[int, int]] = {
        index: {} for index in container.layouts
    }

    for label_index, label in enumerate(table["labels"]):
        layout_index = _int(
            label["layout"], where=f"labels[{label_index}].layout"
        )
        if layout_index not in desired:
            raise ValueError(
                f"{table['file']}: label {label_index} selects missing "
                f"layout {layout_index}"
            )
        copies = label.get("boxes")
        if copies is None:
            copies = [label["box"]]
        for copy_index, box in enumerate(copies):
            copy = dict(label)
            copy["box"] = box
            where = (
                f"{table['file']} label {label_index} "
                f"{label.get('id', '')!r} copy {copy_index}"
            )
            changed_box = _draw_label(
                desired[layout_index],
                copy,
                atlas=atlas,
                char_slots=char_slots,
                where=where,
            )
            changed_boxes[layout_index].append(changed_box)
            if "palette_bank" in label:
                palette_bank = _int(
                    label["palette_bank"],
                    where=f"{where}.palette_bank",
                )
                if not 0 <= palette_bank <= 15:
                    raise ValueError(f"{where}: palette bank must be in 0..15")
                x0, y0, x1, y1 = changed_box
                if any(value % 8 for value in changed_box):
                    raise ValueError(
                        f"{where}: palette override box must be tile-aligned"
                    )
                layout = container.layouts[layout_index]
                for cell_y in range(y0 // 8, y1 // 8):
                    for cell_x in range(x0 // 8, x1 // 8):
                        cell = cell_y * layout.width + cell_x
                        previous = palette_overrides[layout_index].get(cell)
                        if previous is not None and previous != palette_bank:
                            raise ValueError(
                                f"{where}: conflicting palette overrides"
                            )
                        palette_overrides[layout_index][cell] = palette_bank

    for index, before in stock.items():
        boxes = changed_boxes[index]
        for y, row in enumerate(before):
            for x, value in enumerate(row):
                if any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in boxes):
                    continue
                if desired[index][y][x] != value:
                    raise AssertionError(
                        f"{table['file']}: layout {index} changed outside "
                        f"declared label boxes"
                    )

    tile_ids: dict[bytes, int] = {}
    tiles: list[bytes] = []
    rewritten: dict[int, tuple[int, ...]] = {}
    for index in range(container.first_layout, container.last_layout + 1):
        layout = container.layouts[index]
        values: list[int] = []
        for cell, old_value in enumerate(layout.values):
            tile = _tile_from_pixels(
                desired[index], cell % layout.width, cell // layout.width
            )
            if tile not in tile_ids:
                tile_ids[tile] = len(tiles)
                tiles.append(tile)
            palette_bank = palette_overrides[index].get(
                cell, (old_value >> 12) & 0xF
            )
            # Tiles are encoded in their visible orientation, so old H/V flip
            # bits are deliberately discarded.
            values.append((palette_bank << 12) | tile_ids[tile])
        rewritten[index] = tuple(values)
    if len(tiles) > container.tile_capacity:
        raise ValueError(
            f"{table['file']}: rebuilt layouts need {len(tiles)} tiles; "
            f"capacity is {container.tile_capacity}"
        )

    graphics = (
        b"".join(tiles)
        + bytes((container.tile_capacity - len(tiles)) * TILE_BYTES)
    )
    compressed = battle_system_graphics.compress_lzss_optimal(graphics)
    if len(compressed) > container.compressed_capacity:
        raise ValueError(
            f"{table['file']}: compressed payload {len(compressed)} exceeds "
            f"capacity {container.compressed_capacity}"
        )
    if battle_system_graphics.decompress_lzss(compressed) != graphics:
        raise AssertionError(f"{table['file']}: LZSS round-trip failed")

    output = bytearray(source)
    struct.pack_into(
        "<I",
        output,
        container.graphics_offset,
        0x80000000 | len(compressed),
    )
    compressed_start = container.graphics_offset + 4
    output[compressed_start:container.palette_offset] = bytes(
        container.compressed_capacity
    )
    output[
        compressed_start:compressed_start + len(compressed)
    ] = compressed
    for index, values in rewritten.items():
        offset = container.pointers[index]
        struct.pack_into(f"<{len(values)}H", output, offset + 4, *values)

    if (
        len(output) != len(source)
        or output[container.palette_offset:container.layout_start]
        != source[container.palette_offset:container.layout_start]
        or output[container.pointer_table_offset:]
        != source[container.pointer_table_offset:]
    ):
        raise AssertionError(
            f"{table['file']}: fixed palette/pointer/file-size contract drift"
        )

    rebuilt = _decode_container(bytes(output), table, verify_source=False)
    if rebuilt.graphics != graphics:
        raise AssertionError(f"{table['file']}: graphics write-back drift")
    for index, expected_pixels in desired.items():
        actual_pixels = _render_layout(
            rebuilt.graphics, rebuilt.layouts[index]
        )
        if actual_pixels != expected_pixels:
            raise AssertionError(
                f"{table['file']}: layout {index} pixel round-trip drift"
            )

    facts = resource_facts(bytes(output), table)
    expected = table.get("expected", {})
    for key in ("used_tiles", "compressed_size"):
        if key in expected and facts[key] != _int(
            expected[key], where=f"expected.{key}"
        ):
            raise ValueError(
                f"{table['file']}: {key} {facts[key]} != {expected[key]}"
            )
    for key in ("graphics_sha256", "layouts_sha256", "file_sha256"):
        if key in expected and facts[key] != expected[key]:
            raise ValueError(
                f"{table['file']}: {key} {facts[key]} != {expected[key]}"
            )
    return bytes(output)
