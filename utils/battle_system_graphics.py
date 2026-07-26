"""Deterministic ARM9 writer for the in-battle START system menu.

The menu labels are not runtime strings. They are cells in a shared 4bpp tile
set compressed inside ARM9, with 21 small layout records selecting those
tiles. Repainting individual source tiles is unsafe because Japanese glyph
tiles are reused by several menu states.

This writer therefore treats the graphics block and layouts as one fixed-size
contract: validate the pristine Japanese bytes, render every layout to visible
pixels, repaint the translated surfaces from the committed 12x12 atlas,
deduplicate the resulting cells, optimally LZSS-compress the fixed 182-tile
set, and repack all layout records inside their original 804-byte arena.
"""
from __future__ import annotations

import hashlib
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Any

from . import static_graphics


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


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    mode: int
    values: tuple[int, ...]


def decompress_lzss(data: bytes) -> bytes:
    """Decode the game's 4 KiB-ring LZSS payload (descriptor excluded)."""
    ring = bytearray(0x1000)
    flags = 0
    source = 0
    ring_position = 0xFEE
    output = bytearray()
    while source < len(data):
        flags >>= 1
        if flags & 0x100 == 0:
            flags = data[source] | 0xFF00
            source += 1
        if flags & 1:
            if source >= len(data):
                break
            value = data[source]
            source += 1
            output.append(value)
            ring[ring_position] = value
            ring_position = (ring_position + 1) & 0xFFF
            continue
        if source + 1 >= len(data):
            break
        low = data[source]
        high = data[source + 1]
        source += 2
        reference = (low | ((high & 0xF0) << 4)) & 0xFFF
        count = (high & 0x0F) + 3
        for index in range(count):
            value = ring[(reference + index) & 0xFFF]
            output.append(value)
            ring[ring_position] = value
            ring_position = (ring_position + 1) & 0xFFF
    return bytes(output)


def _match_length(
    data: bytes, position: int, reference: int, ring: bytearray
) -> int:
    overwritten: dict[int, int] = {}
    limit = min(18, len(data) - position)
    length = 0
    while length < limit:
        source_slot = (reference + length) & 0xFFF
        value = overwritten.get(source_slot, ring[source_slot])
        if value != data[position + length]:
            break
        destination_slot = (0xFEE + position + length) & 0xFFF
        overwritten[destination_slot] = value
        length += 1
    return length


def _copy_options(data: bytes) -> list[list[tuple[int, int]]]:
    ring = bytearray(0x1000)
    references_by_first: list[set[int]] = [set() for _ in range(256)]
    references_by_first[0].update(range(0x1000))
    options: list[list[tuple[int, int]]] = []
    for position, value in enumerate(data):
        reference_by_length: dict[int, int] = {}
        # Sets make membership updates cheap, but their iteration order is not
        # a serialization contract. Lowest-reference-first makes equal-length
        # matches byte-identical across Python versions and hosts.
        for reference in sorted(references_by_first[value]):
            length = _match_length(data, position, reference, ring)
            for candidate_length in range(3, length + 1):
                reference_by_length.setdefault(candidate_length, reference)
        options.append(
            sorted(
                (length, reference)
                for length, reference in reference_by_length.items()
            )
        )

        slot = (0xFEE + position) & 0xFFF
        references_by_first[ring[slot]].discard(slot)
        ring[slot] = value
        references_by_first[value].add(slot)
    return options


def compress_lzss_optimal(data: bytes) -> bytes:
    """Return a minimum-byte tokenization for the game's LZSS format."""
    copy_options = _copy_options(data)
    infinity = 1 << 30
    # dp[position][token_index_mod_8]; a fresh group costs one flag byte.
    dp = [[infinity] * 8 for _ in range(len(data) + 1)]
    previous: list[
        list[tuple[int, int, str, int, int] | None]
    ] = [[None] * 8 for _ in range(len(data) + 1)]
    dp[0][0] = 0
    for position in range(len(data)):
        for modulo in range(8):
            current = dp[position][modulo]
            if current >= infinity:
                continue
            overhead = 1 if modulo == 0 else 0
            next_modulo = (modulo + 1) & 7

            literal_cost = current + overhead + 1
            if literal_cost < dp[position + 1][next_modulo]:
                dp[position + 1][next_modulo] = literal_cost
                previous[position + 1][next_modulo] = (
                    position,
                    modulo,
                    "literal",
                    data[position],
                    1,
                )

            for length, reference in copy_options[position]:
                copy_cost = current + overhead + 2
                if copy_cost < dp[position + length][next_modulo]:
                    dp[position + length][next_modulo] = copy_cost
                    previous[position + length][next_modulo] = (
                        position,
                        modulo,
                        "copy",
                        reference,
                        length,
                    )

    end_modulo = min(
        range(8), key=lambda modulo: dp[len(data)][modulo]
    )
    if dp[len(data)][end_modulo] >= infinity:
        raise RuntimeError("LZSS optimal parser failed")

    tokens: list[tuple[str, int, int]] = []
    position = len(data)
    modulo = end_modulo
    while position:
        item = previous[position][modulo]
        if item is None:
            raise RuntimeError("LZSS optimal parser backtrack failed")
        old_position, old_modulo, kind, value, length = item
        tokens.append((kind, value, length))
        position, modulo = old_position, old_modulo
    tokens.reverse()

    output = bytearray()
    for group_start in range(0, len(tokens), 8):
        group = tokens[group_start:group_start + 8]
        flags = 0
        payload = bytearray()
        for bit, (kind, value, length) in enumerate(group):
            if kind == "literal":
                flags |= 1 << bit
                payload.append(value)
            else:
                payload.append(value & 0xFF)
                payload.append(
                    ((value >> 4) & 0xF0) | ((length - 3) & 0x0F)
                )
        output.append(flags)
        output.extend(payload)
    return bytes(output)


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


def _flipped_tile(
    tile: bytes, *, horizontal: bool = False, vertical: bool = False
) -> bytes:
    pixels = [[0] * 8 for _ in range(8)]
    for y in range(8):
        for x in range(8):
            source_x = 7 - x if horizontal else x
            source_y = 7 - y if vertical else y
            pixels[y][x] = _get_nibble(tile, source_x, source_y)
    return _tile_from_pixels(pixels, 0, 0)


def _decode_layout(
    arm9: bytes | bytearray,
    index: int,
    *,
    arm9_base: int,
    pointer_table_offset: int,
    layout_start: int,
    layout_end: int,
) -> Layout:
    pointer_offset = pointer_table_offset + index * 4
    if pointer_offset + 4 > len(arm9):
        raise ValueError(f"layout {index}: pointer lies outside ARM9")
    address = struct.unpack_from("<I", arm9, pointer_offset)[0]
    offset = address - arm9_base
    if not layout_start <= offset < layout_end or offset + 4 > layout_end:
        raise ValueError(
            f"layout {index}: pointer {address:#010x} is outside "
            f"[{arm9_base + layout_start:#010x},"
            f"{arm9_base + layout_end:#010x})"
        )
    width, height, mode = struct.unpack_from("<BBH", arm9, offset)
    expected_count = width * height
    if width == 0 or height == 0:
        raise ValueError(f"layout {index}: zero geometry {width}x{height}")
    cursor = offset + 4
    values: list[int] = []
    if mode == 0:
        payload_size = expected_count * 2
        if cursor + payload_size > layout_end:
            raise ValueError(f"layout {index}: explicit payload overruns arena")
        values = list(
            struct.unpack_from(f"<{expected_count}H", arm9, cursor)
        )
    elif mode == 1:
        while len(values) < expected_count:
            if cursor + 4 > layout_end:
                raise ValueError(f"layout {index}: RLE payload overruns arena")
            count, operation, value = struct.unpack_from(
                "<BBH", arm9, cursor
            )
            cursor += 4
            if not count:
                raise ValueError(f"layout {index}: zero-length RLE command")
            if operation == 2:
                values.extend(range(value, value + count))
            elif operation == 1:
                values.extend([value] * count)
            elif operation == 0 and count == 1:
                values.append(value)
            else:
                raise ValueError(
                    f"layout {index}: unsupported RLE command "
                    f"({count}, {operation}, {value:#x})"
                )
            if len(values) > expected_count:
                raise ValueError(f"layout {index}: RLE expands past geometry")
    else:
        raise ValueError(f"layout {index}: unsupported mode {mode}")
    return Layout(width, height, mode, tuple(values))


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
        if value & 0x0400:
            tile = _flipped_tile(tile, horizontal=True)
        if value & 0x0800:
            tile = _flipped_tile(tile, vertical=True)
        origin_x = (cell % layout.width) * 8
        origin_y = (cell // layout.width) * 8
        for y in range(8):
            for x in range(8):
                pixels[origin_y + y][origin_x + x] = _get_nibble(
                    tile, x, y
                )
    return pixels


def _draw_text(
    pixels: list[list[int]],
    text: str,
    box: tuple[int, int, int, int],
    *,
    atlas: bytes,
    char_slots: dict[str, int],
    stroke: int,
    shadow: int,
    base_pixels: list[list[int]] | None,
    clear_indices: set[int],
    fallback_background: int,
    where: str,
) -> None:
    x0, y0, x1, y1 = box
    width = len(pixels[0])
    height = len(pixels)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(f"{where}: text box {box} exceeds {width}x{height}")
    if base_pixels is not None and (
        len(base_pixels) < y1
        or any(len(row) < x1 for row in base_pixels[:y1])
    ):
        raise ValueError(f"{where}: reconstructed base is too small")
    for y in range(y0, y1):
        for x in range(x0, x1):
            if base_pixels is not None:
                pixels[y][x] = base_pixels[y][x]
            elif pixels[y][x] in clear_indices:
                pixels[y][x] = fallback_background

    text_width = len(text) * static_graphics.ATLAS_CELL
    text_height = static_graphics.ATLAS_CELL
    if text_width > x1 - x0 or text_height > y1 - y0:
        raise ValueError(f"{where}: {text!r} does not fit {box}")
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
        for y, row in enumerate(glyph):
            for x, value in enumerate(row):
                if value == 1:
                    pixels[target_y + y][target_x + x] = stroke
                elif value == 2:
                    pixels[target_y + y][target_x + x] = shadow
                elif value != 0:
                    raise ValueError(
                        f"{where}: atlas slot {slot} has value {value}"
                    )
        target_x += static_graphics.ATLAS_CELL


def _encode_rle_layout(
    width: int, height: int, values: tuple[int, ...]
) -> bytes:
    count = len(values)
    infinity = 1 << 30
    best = [infinity] * (count + 1)
    choice: list[tuple[int, int] | None] = [None] * (count + 1)
    best[count] = 0
    for position in range(count - 1, -1, -1):
        candidates: list[tuple[int, int]] = [(0, 1)]
        repeat_end = position + 1
        while (
            repeat_end < count
            and values[repeat_end] == values[position]
            and repeat_end - position < 255
        ):
            repeat_end += 1
            candidates.append((1, repeat_end - position))
        sequence_end = position + 1
        while (
            sequence_end < count
            and values[sequence_end] == values[sequence_end - 1] + 1
            and sequence_end - position < 255
        ):
            sequence_end += 1
            candidates.append((2, sequence_end - position))
        for operation, length in candidates:
            cost = 1 + best[position + length]
            candidate_key = (cost, -length, operation)
            current = choice[position]
            current_key = (
                best[position],
                -(current[1] if current is not None else 0),
                current[0] if current is not None else 9,
            )
            if candidate_key < current_key:
                best[position] = cost
                choice[position] = (operation, length)

    output = bytearray(struct.pack("<BBH", width, height, 1))
    position = 0
    while position < count:
        selected = choice[position]
        if selected is None:
            raise AssertionError(f"RLE layout choice missing at {position}")
        operation, length = selected
        output.extend(
            struct.pack("<BBH", length, operation, values[position])
        )
        position += length
    return bytes(output)


def _encode_layout(layout: Layout, values: tuple[int, ...]) -> bytes:
    explicit = (
        struct.pack("<BBH", layout.width, layout.height, 0)
        + struct.pack(f"<{len(values)}H", *values)
    )
    rle = _encode_rle_layout(layout.width, layout.height, values)
    return min((explicit, rle), key=lambda payload: (len(payload), payload))


def _hash(data: bytes | bytearray) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_hash(
    actual: bytes | bytearray, expected: str, *, where: str
) -> None:
    got = _hash(actual)
    if got != expected:
        raise ValueError(f"{where}: sha256 {got} != {expected}")


def patch_arm9(
    source: bytes,
    target: bytearray,
    table: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
) -> dict[str, int | str]:
    """Patch ``target`` from the pristine Japanese ARM9 source.

    ``target`` may already contain unrelated translated head edits. Every byte
    owned by this writer must still match ``source`` before the atomic rewrite,
    so overlapping patch families fail instead of silently winning by order.
    """
    if table.get("format") != "arm9_battle_system_graphics":
        raise ValueError("battle system menu has unexpected format")
    contract = table["source"]
    arm9_base = _int(contract["arm9_base"], where="source.arm9_base")
    graphics_offset = _int(
        contract["graphics_offset"], where="source.graphics_offset"
    )
    pointer_table_offset = _int(
        contract["pointer_table_offset"],
        where="source.pointer_table_offset",
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
    expected_arm9_size = _int(
        contract["arm9_size"], where="source.arm9_size"
    )
    if len(source) != expected_arm9_size:
        raise ValueError(
            f"battle system menu: source ARM9 size {len(source)} != "
            f"{expected_arm9_size}"
        )
    if len(target) < pointer_table_offset + (last_layout + 1) * 4:
        raise ValueError("battle system menu: target ARM9 head is too short")
    if not (
        0 <= graphics_offset < layout_start < layout_end
        <= pointer_table_offset < len(source)
    ):
        raise ValueError("battle system menu: invalid region ordering")

    descriptor = struct.unpack_from("<I", source, graphics_offset)[0]
    expected_descriptor = _int(
        contract["graphics_descriptor"],
        where="source.graphics_descriptor",
    )
    if descriptor != expected_descriptor or not descriptor & 0x80000000:
        raise ValueError(
            f"battle system menu: descriptor {descriptor:#010x} != "
            f"{expected_descriptor:#010x}"
        )
    compressed_capacity = descriptor & 0xFFFF
    graphics_span = source[
        graphics_offset:graphics_offset + 4 + compressed_capacity
    ]
    if len(graphics_span) != 4 + compressed_capacity:
        raise ValueError("battle system menu: compressed block is truncated")
    _require_hash(
        graphics_span,
        contract["graphics_block_sha256"],
        where="battle system graphics source block",
    )
    compressed = graphics_span[4:]
    decoded = decompress_lzss(compressed)
    expected_decoded_size = _int(
        contract["decoded_size"], where="source.decoded_size"
    )
    if len(decoded) != expected_decoded_size or len(decoded) % TILE_BYTES:
        raise ValueError(
            f"battle system menu: decoded size {len(decoded)} != "
            f"{expected_decoded_size}"
        )
    _require_hash(
        decoded,
        contract["decoded_sha256"],
        where="battle system decoded graphics",
    )
    tile_capacity = _int(
        contract["tile_capacity"], where="source.tile_capacity"
    )
    if len(decoded) != tile_capacity * TILE_BYTES:
        raise ValueError("battle system menu: tile capacity disagrees with block")

    source_layout_region = source[layout_start:layout_end]
    _require_hash(
        source_layout_region,
        contract["layout_region_sha256"],
        where="battle system source layout arena",
    )
    pointer_start = pointer_table_offset + first_layout * 4
    pointer_end = pointer_table_offset + (last_layout + 1) * 4
    source_pointers = source[pointer_start:pointer_end]
    _require_hash(
        source_pointers,
        contract["pointer_table_sha256"],
        where="battle system source pointer slice",
    )

    owned_spans = (
        (
            graphics_offset,
            graphics_offset + 4 + compressed_capacity,
            "graphics block",
        ),
        (layout_start, layout_end, "layout arena"),
        (pointer_start, pointer_end, "pointer slice"),
    )
    for start, end, what in owned_spans:
        if target[start:end] != source[start:end]:
            raise ValueError(
                f"battle system menu: {what} was changed by another writer"
            )

    layouts = {
        index: _decode_layout(
            source,
            index,
            arm9_base=arm9_base,
            pointer_table_offset=pointer_table_offset,
            layout_start=layout_start,
            layout_end=layout_end,
        )
        for index in range(first_layout, last_layout + 1)
    }
    stock_surfaces = {
        index: _render_layout(decoded, layout)
        for index, layout in layouts.items()
    }

    palette = table["palette_indices"]
    stroke = _int(palette["stroke"], where="palette_indices.stroke")
    shadow = _int(palette["shadow"], where="palette_indices.shadow")
    fallback_background = _int(
        palette["fallback_background"],
        where="palette_indices.fallback_background",
    )
    clear_indices = {
        _int(value, where="palette_indices.clear")
        for value in palette["clear"]
    }
    if (
        any(not 0 <= value <= 15 for value in clear_indices)
        or not 0 <= stroke <= 15
        or not 0 <= shadow <= 15
        or not 0 <= fallback_background <= 15
    ):
        raise ValueError("battle system menu: invalid 4bpp palette index")

    menu_base_indices = [
        _int(value, where="menu_base_indices")
        for value in table["menu_base_indices"]
    ]
    if not menu_base_indices:
        raise ValueError("battle system menu: no menu base surfaces")
    base_width = min(
        len(stock_surfaces[index][0]) for index in menu_base_indices
    )
    base_height = min(
        len(stock_surfaces[index]) for index in menu_base_indices
    )
    menu_base = [[0] * base_width for _ in range(base_height)]
    for y in range(base_height):
        for x in range(base_width):
            candidates = [
                stock_surfaces[index][y][x]
                for index in menu_base_indices
                if stock_surfaces[index][y][x] not in {stroke, shadow}
            ]
            menu_base[y][x] = (
                Counter(candidates).most_common(1)[0][0]
                if candidates
                else fallback_background
            )

    prompt = table["prompt"]
    prompt_index = _int(prompt["index"], where="prompt.index")
    prompt_base = [row[:] for row in stock_surfaces[prompt_index]]
    row_backgrounds = [
        _int(value, where="prompt.row_backgrounds")
        for value in prompt["row_backgrounds"]
    ]
    if len(row_backgrounds) != len(prompt_base):
        raise ValueError("battle system menu: prompt row backgrounds drift")
    for y, row in enumerate(prompt_base):
        for x, value in enumerate(row):
            if value in {stroke, shadow}:
                row[x] = row_backgrounds[y]

    confirmation = table["confirmation"]
    yes_index = _int(
        confirmation["yes_index"], where="confirmation.yes_index"
    )
    no_index = _int(
        confirmation["no_index"], where="confirmation.no_index"
    )
    dialog_index = _int(
        confirmation["dialog_index"], where="confirmation.dialog_index"
    )
    confirmation_base = [
        [0] * len(stock_surfaces[yes_index][0])
        for _ in range(len(stock_surfaces[yes_index]))
    ]
    for y, row in enumerate(confirmation_base):
        for x in range(len(row)):
            candidates = [
                stock_surfaces[index][y][x]
                for index in (yes_index, no_index)
                if stock_surfaces[index][y][x] not in {stroke, shadow}
            ]
            row[x] = (
                Counter(candidates).most_common(1)[0][0]
                if candidates
                else menu_base[y][min(x, base_width - 1)]
            )

    confirmation_surfaces: dict[int, list[list[int]]] = {}
    confirm_box_document = confirmation["box"]
    confirm_box = tuple(
        _int(confirm_box_document[name], where=f"confirmation.box.{name}")
        for name in ("x0", "y0", "x1", "y1")
    )
    for index, key in ((yes_index, "yes_text"), (no_index, "no_text")):
        surface = [row[:] for row in confirmation_base]
        _draw_text(
            surface,
            confirmation[key],
            confirm_box,
            atlas=atlas,
            char_slots=char_slots,
            stroke=stroke,
            shadow=shadow,
            base_pixels=confirmation_base,
            clear_indices=clear_indices,
            fallback_background=fallback_background,
            where=f"battle system confirmation layout {index}",
        )
        confirmation_surfaces[index] = surface

    label_by_index: dict[int, str] = {}
    for label in table["menu_labels"]:
        text = label["text"]
        for raw_index in label["indices"]:
            index = _int(raw_index, where=f"menu label {text!r}.indices")
            if index in label_by_index:
                raise ValueError(
                    f"battle system menu: layout {index} has two labels"
                )
            label_by_index[index] = text

    menu_box_document = table["menu_box"]
    menu_box = tuple(
        _int(menu_box_document[name], where=f"menu_box.{name}")
        for name in ("x0", "y0", "x1", "y1")
    )
    prompt_box_document = prompt["box"]
    prompt_box = tuple(
        _int(prompt_box_document[name], where=f"prompt.box.{name}")
        for name in ("x0", "y0", "x1", "y1")
    )

    desired_surfaces: dict[int, list[list[int]]] = {}
    for index in range(first_layout, last_layout + 1):
        pixels = [row[:] for row in stock_surfaces[index]]
        if index in label_by_index:
            _draw_text(
                pixels,
                label_by_index[index],
                menu_box,
                atlas=atlas,
                char_slots=char_slots,
                stroke=stroke,
                shadow=shadow,
                base_pixels=menu_base,
                clear_indices=clear_indices,
                fallback_background=fallback_background,
                where=f"battle system menu layout {index}",
            )
        elif index == prompt_index:
            _draw_text(
                pixels,
                prompt["text"],
                prompt_box,
                atlas=atlas,
                char_slots=char_slots,
                stroke=stroke,
                shadow=shadow,
                base_pixels=prompt_base,
                clear_indices=clear_indices,
                fallback_background=fallback_background,
                where=f"battle system prompt layout {index}",
            )
        elif index == dialog_index:
            destination = confirmation["dialog_destination"]
            destination_x = _int(
                destination["x"], where="confirmation.dialog_destination.x"
            )
            yes_y = _int(
                destination["yes_y"],
                where="confirmation.dialog_destination.yes_y",
            )
            no_y = _int(
                destination["no_y"],
                where="confirmation.dialog_destination.no_y",
            )
            copy_width = _int(
                destination["width"],
                where="confirmation.dialog_destination.width",
            )
            copy_height = _int(
                destination["height"],
                where="confirmation.dialog_destination.height",
            )
            for source_surface, destination_y in (
                (confirmation_surfaces[yes_index], yes_y),
                (confirmation_surfaces[no_index], no_y),
            ):
                if (
                    destination_x + copy_width > len(pixels[0])
                    or destination_y + copy_height > len(pixels)
                ):
                    raise ValueError(
                        "battle system menu: confirmation dialog copy overruns"
                    )
                for y in range(copy_height):
                    pixels[destination_y + y][
                        destination_x:destination_x + copy_width
                    ] = source_surface[y][:copy_width]
        elif index in confirmation_surfaces:
            pixels = [row[:] for row in confirmation_surfaces[index]]
        desired_surfaces[index] = pixels

    desired_cells: dict[int, tuple[bytes, ...]] = {}
    for index, layout in layouts.items():
        pixels = desired_surfaces[index]
        desired_cells[index] = tuple(
            _tile_from_pixels(
                pixels, cell % layout.width, cell // layout.width
            )
            for cell in range(layout.width * layout.height)
        )

    tile_ids: dict[bytes, int] = {}
    tiles: list[bytes] = []
    rewritten: dict[int, tuple[int, ...]] = {}
    for index in range(first_layout, last_layout + 1):
        values: list[int] = []
        if len(layouts[index].values) != len(desired_cells[index]):
            raise AssertionError(
                f"battle system layout {index} cell-count drift"
            )
        for old_value, tile in zip(
            layouts[index].values, desired_cells[index]
        ):
            if tile not in tile_ids:
                tile_ids[tile] = len(tiles)
                tiles.append(tile)
            values.append((old_value & 0xF000) | tile_ids[tile])
        rewritten[index] = tuple(values)
    if len(tiles) > tile_capacity:
        raise ValueError(
            f"battle system menu needs {len(tiles)} tiles; "
            f"capacity is {tile_capacity}"
        )
    repacked_graphics = (
        b"".join(tiles)
        + bytes((tile_capacity - len(tiles)) * TILE_BYTES)
    )
    recompressed = compress_lzss_optimal(repacked_graphics)
    if len(recompressed) > compressed_capacity:
        raise ValueError(
            f"battle system menu compressed size {len(recompressed)} "
            f"exceeds capacity {compressed_capacity}"
        )
    if decompress_lzss(recompressed) != repacked_graphics:
        raise AssertionError("battle system graphics LZSS round-trip failed")

    layout_payloads = {
        index: _encode_layout(layouts[index], rewritten[index])
        for index in range(first_layout, last_layout + 1)
    }
    layout_blob = b"".join(layout_payloads.values())
    layout_capacity = layout_end - layout_start
    if len(layout_blob) > layout_capacity:
        raise ValueError(
            f"battle system layouts need {len(layout_blob)} bytes; "
            f"capacity is {layout_capacity}"
        )

    expected = table.get("expected", {})
    facts = {
        "used_tiles": len(tiles),
        "compressed_size": len(recompressed),
        "layout_size": len(layout_blob),
        "graphics_sha256": _hash(repacked_graphics),
        "layouts_sha256": _hash(layout_blob),
    }
    for key in ("used_tiles", "compressed_size", "layout_size"):
        if key in expected and facts[key] != _int(
            expected[key], where=f"expected.{key}"
        ):
            raise ValueError(
                f"battle system menu: {key} {facts[key]} != "
                f"{expected[key]}"
            )
    for key in ("graphics_sha256", "layouts_sha256"):
        if key in expected and facts[key] != expected[key]:
            raise ValueError(
                f"battle system menu: {key} {facts[key]} != "
                f"{expected[key]}"
            )

    struct.pack_into(
        "<I",
        target,
        graphics_offset,
        0x80000000 | len(recompressed),
    )
    target[
        graphics_offset + 4:
        graphics_offset + 4 + compressed_capacity
    ] = bytes(compressed_capacity)
    target[
        graphics_offset + 4:
        graphics_offset + 4 + len(recompressed)
    ] = recompressed
    target[layout_start:layout_end] = bytes(layout_capacity)
    cursor = layout_start
    for index in range(first_layout, last_layout + 1):
        payload = layout_payloads[index]
        target[cursor:cursor + len(payload)] = payload
        struct.pack_into(
            "<I",
            target,
            pointer_table_offset + index * 4,
            arm9_base + cursor,
        )
        cursor += len(payload)

    new_descriptor = struct.unpack_from("<I", target, graphics_offset)[0]
    new_compressed_size = new_descriptor & 0xFFFF
    new_graphics = decompress_lzss(
        bytes(
            target[
                graphics_offset + 4:
                graphics_offset + 4 + new_compressed_size
            ]
        )
    )
    if new_graphics != repacked_graphics:
        raise AssertionError("battle system graphics write-back drift")
    for index in range(first_layout, last_layout + 1):
        rebuilt = _decode_layout(
            target,
            index,
            arm9_base=arm9_base,
            pointer_table_offset=pointer_table_offset,
            layout_start=layout_start,
            layout_end=layout_end,
        )
        if (
            rebuilt.width != layouts[index].width
            or rebuilt.height != layouts[index].height
            or rebuilt.values != rewritten[index]
        ):
            raise AssertionError(
                f"battle system layout {index} write-back drift"
            )
        if _render_layout(new_graphics, rebuilt) != desired_surfaces[index]:
            raise AssertionError(
                f"battle system layout {index} pixel round-trip drift"
            )

    return {
        **facts,
        "tile_capacity": tile_capacity,
        "tile_slack": tile_capacity - len(tiles),
        "compressed_capacity": compressed_capacity,
        "compressed_slack": compressed_capacity - len(recompressed),
        "layout_capacity": layout_capacity,
        "layout_slack": layout_capacity - len(layout_blob),
    }
