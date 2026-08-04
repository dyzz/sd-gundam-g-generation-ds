"""Writer for the paired system-settings BG resources (3e3.bin/3e4.bin).

The settings screen is split across two fixed-capacity 4bpp tile resources:

* 3e3.bin carries the descriptions and every unfocused value.
* 3e4.bin carries the focused/selected value sprites in two palette styles.

The runtime loads both resources consecutively and conditionally hides the
New Game+ row through its screen map. This writer changes no game logic: it
repaints the two source canvases from semantic JSON specs, then copy-on-write
repacks each canvas inside its original tile budget.
"""
from __future__ import annotations

import hashlib
from typing import Any

from . import static_graphics


def _int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: expected integer, got {value!r}")
    return value


def _box(document: dict, *, where: str) -> tuple[int, int, int, int]:
    x = _int(document["x"], where=f"{where}.x")
    y = _int(document["y"], where=f"{where}.y")
    width = _int(document["width"], where=f"{where}.width")
    height = _int(document["height"], where=f"{where}.height")
    if width <= 0 or height <= 0:
        raise ValueError(f"{where}: width and height must be positive")
    return x, y, x + width, y + height


def _assert_bounds(
    box: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    where: str,
) -> None:
    x0, y0, x1, y1 = box
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(
            f"{where}: box {box} exceeds {width}x{height} canvas"
        )


def _palette(document: dict, *, where: str) -> tuple[int, int]:
    stroke = _int(document["stroke"], where=f"{where}.stroke")
    shadow = _int(document["shadow"], where=f"{where}.shadow")
    if not (0 <= stroke <= 15 and 0 <= shadow <= 15):
        raise ValueError(f"{where}: 4bpp indices must be in 0..15")
    return stroke, shadow


def _draw_text(
    canvas: list[list[int]],
    text: str,
    x: int,
    y: int,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
    stroke: int,
    shadow: int,
    where: str,
) -> None:
    width = len(canvas[0])
    height = len(canvas)
    text_box = (
        x,
        y,
        x + len(text) * static_graphics.ATLAS_CELL,
        y + static_graphics.ATLAS_CELL,
    )
    _assert_bounds(text_box, width=width, height=height, where=where)
    for char_index, char in enumerate(text):
        try:
            slot = int(char_slots[char])
        except KeyError as exc:
            raise ValueError(
                f"{where}: no committed atlas slot for {char!r}"
            ) from exc
        glyph = static_graphics.atlas_cell(atlas, slot)
        origin_x = x + char_index * static_graphics.ATLAS_CELL
        for glyph_y, row in enumerate(glyph):
            for glyph_x, value in enumerate(row):
                if value == 1:
                    canvas[y + glyph_y][origin_x + glyph_x] = stroke
                elif value == 2:
                    canvas[y + glyph_y][origin_x + glyph_x] = shadow
                elif value != 0:
                    raise ValueError(
                        f"{where}: atlas slot {slot} has unsupported value "
                        f"{value}"
                    )


def _paint_solid_label(
    canvas: list[list[int]],
    label: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
    where: str,
) -> None:
    clear = _box(label["clear"], where=f"{where}.clear")
    _assert_bounds(
        clear,
        width=len(canvas[0]),
        height=len(canvas),
        where=f"{where}.clear",
    )
    background = _int(
        label["clear"]["background"], where=f"{where}.clear.background"
    )
    if not 0 <= background <= 15:
        raise ValueError(f"{where}: 4bpp background must be in 0..15")
    for y in range(clear[1], clear[3]):
        canvas[y][clear[0]:clear[2]] = [background] * (
            clear[2] - clear[0]
        )
    stroke, shadow = _palette(
        label["palette_indices"], where=f"{where}.palette_indices"
    )
    _draw_text(
        canvas,
        label["text"],
        _int(label["x"], where=f"{where}.x"),
        _int(label["y"], where=f"{where}.y"),
        atlas=atlas,
        char_slots=char_slots,
        stroke=stroke,
        shadow=shadow,
        where=where,
    )


def _paint_transparent_label(
    canvas: list[list[int]],
    label: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
    where: str,
) -> None:
    clear_document = label["clear"]
    clear = _box(clear_document, where=f"{where}.clear")
    _assert_bounds(
        clear,
        width=len(canvas[0]),
        height=len(canvas),
        where=f"{where}.clear",
    )
    old_indices = {
        _int(value, where=f"{where}.clear.old_indices")
        for value in clear_document["old_indices"]
    }
    if not old_indices or any(not 0 <= value <= 15 for value in old_indices):
        raise ValueError(f"{where}: old_indices must be nonempty 4bpp values")
    background = _int(
        clear_document["background"], where=f"{where}.clear.background"
    )
    if not 0 <= background <= 15:
        raise ValueError(f"{where}: 4bpp background must be in 0..15")
    for y in range(clear[1], clear[3]):
        for x in range(clear[0], clear[2]):
            if canvas[y][x] in old_indices:
                canvas[y][x] = background
    stroke, shadow = _palette(
        label["palette_indices"], where=f"{where}.palette_indices"
    )
    _draw_text(
        canvas,
        label["text"],
        _int(label["x"], where=f"{where}.x"),
        _int(label["y"], where=f"{where}.y"),
        atlas=atlas,
        char_slots=char_slots,
        stroke=stroke,
        shadow=shadow,
        where=where,
    )


def _paint_button(
    canvas: list[list[int]],
    text: str,
    x: int,
    y: int,
    style: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
    where: str,
) -> None:
    width = _int(style["width"], where=f"{where}.style.width")
    height = _int(style["height"], where=f"{where}.style.height")
    _assert_bounds(
        (x, y, x + width, y + height),
        width=len(canvas[0]),
        height=len(canvas),
        where=where,
    )
    if width != 32 or height != 16:
        raise ValueError(
            f"{where}: settings button frame must be 32x16, got "
            f"{width}x{height}"
        )
    frame = style["frame_indices"]
    frame_indices = {
        name: _int(frame[name], where=f"{where}.style.frame_indices.{name}")
        for name in (
            "outside",
            "outer",
            "middle",
            "inner_border",
            "background",
        )
    }
    if any(not 0 <= value <= 15 for value in frame_indices.values()):
        raise ValueError(f"{where}: button-frame indices must be in 0..15")

    # Rebuild the complete beveled base instead of merely clearing its 26x10
    # interior. Japanese labels reach into the lower/side border pixels;
    # repainting only the interior leaves fragments below Chinese values.
    outside = frame_indices["outside"]
    outer = frame_indices["outer"]
    middle = frame_indices["middle"]
    inner_border = frame_indices["inner_border"]
    background = frame_indices["background"]
    frame_rows = [
        [outside] + [outer] * 30 + [outside],
        [outer] + [middle] * 30 + [outer],
        [outer, middle] + [inner_border] * 28 + [middle, outer],
    ]
    frame_rows.extend(
        [
            [outer, middle, inner_border]
            + [background] * 26
            + [inner_border, middle, outer]
        ]
        * 10
    )
    frame_rows.extend((frame_rows[2], frame_rows[1], frame_rows[0]))
    for row_index, row in enumerate(frame_rows):
        canvas[y + row_index][x:x + width] = row

    text_width = len(text) * static_graphics.ATLAS_CELL
    if text_width > width:
        raise ValueError(
            f"{where}: {text!r} is {text_width}px wide; button is {width}px"
        )
    text_x = x + (width - text_width) // 2
    text_y = y + _int(style["text_y"], where=f"{where}.style.text_y")
    stroke, shadow = _palette(
        style["palette_indices"], where=f"{where}.style.palette_indices"
    )
    _draw_text(
        canvas,
        text,
        text_x,
        text_y,
        atlas=atlas,
        char_slots=char_slots,
        stroke=stroke,
        shadow=shadow,
        where=where,
    )
    # The atlas drop shadow lands on the lower inner bevel at y=2. Restore
    # that bevel after drawing; the glyph stroke ends one row above it.
    canvas[y + height - 3][x:x + width] = frame_rows[2]


def _paint_button_groups(
    canvas: list[list[int]],
    table: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
) -> None:
    styles = table.get("button_styles", {})
    for group_index, group in enumerate(table.get("button_groups", [])):
        style_name = group["style"]
        try:
            style = styles[style_name]
        except KeyError as exc:
            raise ValueError(
                f"{table['file']} button group {group_index}: unknown style "
                f"{style_name!r}"
            ) from exc
        for origin_index, origin in enumerate(group["origins"]):
            origin_x = _int(
                origin["x"],
                where=(
                    f"{table['file']} button group {group_index} "
                    f"origin {origin_index}.x"
                ),
            )
            origin_y = _int(
                origin["y"],
                where=(
                    f"{table['file']} button group {group_index} "
                    f"origin {origin_index}.y"
                ),
            )
            for button_index, button in enumerate(group["buttons"]):
                where = (
                    f"{table['file']} button group {group_index} "
                    f"origin {origin_index} button {button_index} "
                    f"{button['text']!r}"
                )
                _paint_button(
                    canvas,
                    button["text"],
                    origin_x + _int(button["dx"], where=f"{where}.dx"),
                    origin_y + _int(button.get("dy", 0), where=f"{where}.dy"),
                    style,
                    atlas=atlas,
                    char_slots=char_slots,
                    where=where,
                )


def _paint_pointed_label(
    canvas: list[list[int]],
    label: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
    where: str,
) -> None:
    body = label["body"]
    body_box = _box(body, where=f"{where}.body")
    _assert_bounds(
        body_box,
        width=len(canvas[0]),
        height=len(canvas),
        where=f"{where}.body",
    )
    if "source_ink_bounds" in label:
        source_ink_box = _box(
            label["source_ink_bounds"],
            where=f"{where}.source_ink_bounds",
        )
        _assert_bounds(
            source_ink_box,
            width=len(canvas[0]),
            height=len(canvas),
            where=f"{where}.source_ink_bounds",
        )
        if not (
            body_box[0] <= source_ink_box[0]
            and body_box[1] <= source_ink_box[1]
            and body_box[2] >= source_ink_box[2]
            and body_box[3] >= source_ink_box[3]
        ):
            raise ValueError(
                f"{where}: body does not cover the full source ink bounds"
            )
    backgrounds = [
        _int(value, where=f"{where}.body.row_backgrounds")
        for value in body["row_backgrounds"]
    ]
    if len(backgrounds) != body_box[3] - body_box[1]:
        raise ValueError(
            f"{where}: row_backgrounds length differs from body height"
        )
    if any(not 0 <= value <= 15 for value in backgrounds):
        raise ValueError(f"{where}: row backgrounds must be in 0..15")
    for row_index, y in enumerate(range(body_box[1], body_box[3])):
        canvas[y][body_box[0]:body_box[2]] = (
            [backgrounds[row_index]] * (body_box[2] - body_box[0])
        )
    stroke, shadow = _palette(
        label["palette_indices"], where=f"{where}.palette_indices"
    )
    _draw_text(
        canvas,
        label["text"],
        _int(label["x"], where=f"{where}.x"),
        _int(label["y"], where=f"{where}.y"),
        atlas=atlas,
        char_slots=char_slots,
        stroke=stroke,
        shadow=shadow,
        where=where,
    )


def _verify_source(source: bytes, table: dict) -> static_graphics.TileMap:
    contract = table["source"]
    expected_size = _int(contract["size"], where=f"{table['file']}.source.size")
    if len(source) != expected_size:
        raise ValueError(
            f"{table['file']}: source size {len(source)} != {expected_size}"
        )
    expected_sha256 = contract["sha256"]
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{table['file']}: source sha256 {actual_sha256} != "
            f"{expected_sha256}"
        )
    mapping = static_graphics.tile_map(source)
    geometry = (
        mapping.width_tiles,
        mapping.height_tiles,
        mapping.screen_offset,
        mapping.gfx_offset,
        mapping.gfx_len,
    )
    expected_geometry = (
        _int(contract["width_tiles"], where="source.width_tiles"),
        _int(contract["height_tiles"], where="source.height_tiles"),
        _int(contract["screen_offset"], where="source.screen_offset"),
        _int(contract["gfx_offset"], where="source.gfx_offset"),
        _int(contract["gfx_len"], where="source.gfx_len"),
    )
    if geometry != expected_geometry:
        raise ValueError(
            f"{table['file']}: source geometry {geometry} != "
            f"{expected_geometry}"
        )
    return mapping


def _verify_preserved_regions(
    before: list[list[int]],
    after: list[list[int]],
    table: dict,
) -> None:
    for index, region in enumerate(table.get("preserve_regions", [])):
        box = _box(region, where=f"{table['file']}.preserve_regions[{index}]")
        _assert_bounds(
            box,
            width=len(before[0]),
            height=len(before),
            where=f"{table['file']}.preserve_regions[{index}]",
        )
        for y in range(box[1], box[3]):
            if before[y][box[0]:box[2]] != after[y][box[0]:box[2]]:
                raise ValueError(
                    f"{table['file']}: preserved region {index} "
                    f"{region.get('what', '')!r} changed"
                )


def repaint_settings(
    source: bytes,
    table: dict,
    *,
    atlas: bytes,
    char_slots: dict[str, int],
) -> bytes:
    """Repaint one settings resource from its semantic fixed-capacity spec."""
    if table.get("format") != "settings_graphics":
        raise ValueError(
            f"{table.get('file', '<unknown>')}: expected settings_graphics"
        )
    mapping = _verify_source(source, table)
    canvas, decoded_mapping = static_graphics.decode_index_canvas(source)
    if decoded_mapping != mapping:
        raise AssertionError(f"{table['file']}: tile-map decode drift")
    original_canvas = [row[:] for row in canvas]

    for index, label in enumerate(table.get("solid_labels", [])):
        _paint_solid_label(
            canvas,
            label,
            atlas=atlas,
            char_slots=char_slots,
            where=f"{table['file']} solid label {index} {label['text']!r}",
        )
    for index, label in enumerate(table.get("transparent_labels", [])):
        _paint_transparent_label(
            canvas,
            label,
            atlas=atlas,
            char_slots=char_slots,
            where=(
                f"{table['file']} transparent label {index} "
                f"{label['text']!r}"
            ),
        )
    _paint_button_groups(
        canvas, table, atlas=atlas, char_slots=char_slots
    )
    for index, label in enumerate(table.get("pointed_labels", [])):
        _paint_pointed_label(
            canvas,
            label,
            atlas=atlas,
            char_slots=char_slots,
            where=f"{table['file']} pointed label {index} {label['text']!r}",
        )

    _verify_preserved_regions(original_canvas, canvas, table)
    output = static_graphics.repack_canvas(
        source, canvas, mapping, where=table["file"]
    )
    if len(output) != len(source):
        raise AssertionError(f"{table['file']}: fixed-capacity writer grew file")
    used_tiles = len(
        {
            entry & 0x3FF
            for entry in static_graphics.tile_map(output).entries
        }
    )
    expected_used = table.get("expected_used_tiles")
    if expected_used is not None and used_tiles != int(expected_used):
        raise ValueError(
            f"{table['file']}: output uses {used_tiles} tiles; "
            f"expected {expected_used}"
        )
    return output
