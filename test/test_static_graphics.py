"""Unit tests for guarded static BG label repainting."""
from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

from utils import font_atlas, static_graphics


REPO = Path(__file__).resolve().parent.parent


def resource(entries: tuple[int, int, int, int]) -> bytes:
    header = bytearray(12)
    header[2:4] = bytes((2, 2))
    struct.pack_into("<HHHH", header, 4, 12, 8, 20, 32 * 4)
    screen = b"".join(struct.pack("<H", entry) for entry in entries)
    tiles = bytes([0x66]) * (32 * 4)
    return bytes(header) + screen + tiles


def table(tiles: list[int]) -> dict:
    return {
        "file": "synthetic.bin",
        "palette_indices": {"stroke": 15, "shadow": 2},
        "labels": [
            {
                "text": "A",
                "tiles": tiles,
                "x": 0,
                "y": 0,
                "clear": {
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 12,
                    "sample_x": 15,
                },
            }
        ],
    }


class StaticGraphicsTest(unittest.TestCase):
    def test_unique_tiles_are_repainted_without_geometry_changes(self):
        source = resource((0, 1, 2, 3))
        atlas = bytes([1]) + bytes(static_graphics.ATLAS_CELL_BYTES - 1)
        output = static_graphics.repaint_atlas_text(
            source, table([0, 1, 2, 3]), atlas=atlas, char_slots={"A": 0}
        )
        canvas, _tile_map = static_graphics.decode_index_canvas(output)
        self.assertEqual(canvas[0][0], 15)
        self.assertEqual(len(output), len(source))
        self.assertEqual(output[12:20], source[12:20])

    def test_incompatible_shared_tile_is_rejected(self):
        source = resource((0, 0, 0, 0))
        atlas = bytes([1]) + bytes(static_graphics.ATLAS_CELL_BYTES - 1)
        with self.assertRaisesRegex(ValueError, "shared by incompatible"):
            static_graphics.repaint_atlas_text(
                source, table([0]), atlas=atlas, char_slots={"A": 0}
            )

    def test_repack_tiles_copy_on_write_shared_cells(self):
        source = resource((0, 0, 0, 0))
        atlas = bytes([1]) + bytes(static_graphics.ATLAS_CELL_BYTES - 1)
        spec = table([])
        spec["repack_tiles"] = True
        del spec["labels"][0]["tiles"]
        output = static_graphics.repaint_atlas_text(
            source, spec, atlas=atlas, char_slots={"A": 0}
        )
        canvas, tile_map = static_graphics.decode_index_canvas(output)
        self.assertEqual(canvas[0][0], 15)
        self.assertEqual(canvas[0][8], 6)
        self.assertNotEqual(tile_map.entries[0] & 0x3FF, tile_map.entries[1] & 0x3FF)
        self.assertEqual(len(output), len(source))

    def test_pixel_move_adds_tracking_without_moving_the_background(self):
        source = bytearray(resource((0, 1, 2, 3)))
        # One stroke pixel at visible (0, 0); the rest of tile 0 is index 6.
        source[20] = 0x6F
        spec = {
            "file": "synthetic.bin",
            "palette_indices": {"stroke": 15, "shadow": 2},
            "labels": [],
            "repack_tiles": True,
            "pixel_moves": [
                {
                    "what": "tracking",
                    "source": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "dx": 2,
                    "dy": 0,
                    "sample_x": 7,
                    "indices": [15],
                }
            ],
        }
        output = static_graphics.repaint_atlas_text(
            bytes(source),
            spec,
            atlas=bytes(static_graphics.ATLAS_CELL_BYTES),
            char_slots={},
        )
        canvas, _tile_map = static_graphics.decode_index_canvas(output)
        self.assertEqual(canvas[0][0], 6)
        self.assertEqual(canvas[0][2], 15)
        self.assertEqual(len(output), len(source))

    def test_terrain_badges_use_committed_fan_and_kong_glyphs(self):
        spec = json.loads(
            (
                REPO / "data/zh/files/graphics/48a.json"
            ).read_text(encoding="utf-8")
        )
        charmap = json.loads(
            (REPO / "data/charmap.json").read_text(encoding="utf-8")
        )
        atlas = font_atlas.load_effective_atlas(REPO / "data")
        regions = {
            int(region["offset"], 0): bytes.fromhex(region["zh_hex"])
            for region in spec["regions"]
        }

        def sprite_block(char: str) -> bytes:
            glyph = static_graphics.atlas_cell(
                atlas, charmap["two_byte_zh"][char]
            )
            canvas = [[0] * 16 for _ in range(16)]
            for y, row in enumerate(glyph):
                for x, value in enumerate(row):
                    canvas[y + 2][x + 2] = (
                        15 if value == 1 else 2 if value == 2 else 0
                    )
            output = bytearray()
            for quadrant in range(4):
                origin_x = (quadrant % 2) * 8
                origin_y = (quadrant // 2) * 8
                tile = bytearray(32)
                for y in range(8):
                    for x in range(8):
                        value = canvas[origin_y + y][origin_x + x]
                        tile[y * 4 + x // 2] |= value << (
                            4 if x & 1 else 0
                        )
                output.extend(tile)
            return bytes(output)

        self.assertEqual(set(regions), {0x3B0, 0x4B0})
        self.assertEqual(regions[0x3B0], sprite_block("泛"))
        self.assertEqual(regions[0x4B0], sprite_block("空"))

    def test_force_hud_battleship_has_no_drop_shadow(self):
        spec = json.loads(
            (
                REPO / "data/zh/files/graphics/478.json"
            ).read_text(encoding="utf-8")
        )
        regions = [
            (
                int(region["offset"], 0),
                bytes.fromhex(region["zh_hex"]),
            )
            for region in spec["regions"]
        ]

        def tile(tile_index: int) -> bytes:
            offset = 0x610 + tile_index * 32
            for region_offset, payload in regions:
                local = offset - region_offset
                if 0 <= local and local + 32 <= len(payload):
                    return payload[local:local + 32]
            self.fail(f"478.bin target tile {tile_index} is not fully specified")

        canvas = [[0] * 32 for _ in range(16)]
        for cell_y, tile_row in enumerate(((8, 9, 10, 11), (17, 18, 19, 20))):
            for cell_x, tile_index in enumerate(tile_row):
                payload = tile(tile_index)
                for y in range(8):
                    for x in range(8):
                        packed = payload[y * 4 + x // 2]
                        canvas[cell_y * 8 + y][cell_x * 8 + x] = (
                            packed >> (4 if x & 1 else 0)
                        ) & 0xF

        foreground = {
            (x, y)
            for y in range(3, 15)
            for x in range(5, 29)
            if canvas[y][x] == 15
        }
        actual_shadow = {
            (x, y)
            for y in range(3, 16)
            for x in range(5, 30)
            if canvas[y][x] == 2
        }
        self.assertTrue(foreground)
        self.assertEqual(actual_shadow, set())


if __name__ == "__main__":
    unittest.main()
