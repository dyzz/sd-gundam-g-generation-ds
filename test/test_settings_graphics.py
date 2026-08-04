"""Unit and schema tests for the paired system-settings resources."""
from __future__ import annotations

import hashlib
import json
import struct
import unittest
from pathlib import Path

from utils import data_files, settings_graphics, static_graphics


REPO = Path(__file__).resolve().parent.parent
GRAPHICS = REPO / "data" / "zh" / "files" / "graphics"


def synthetic_resource(
    width: int = 2, height: int = 2, tile_capacity: int = 8
) -> bytes:
    screen_len = width * height * 2
    gfx_offset = 12 + screen_len
    header = bytearray(12)
    header[2:4] = bytes((width, height))
    struct.pack_into(
        "<HHHH",
        header,
        4,
        12,
        screen_len,
        gfx_offset,
        tile_capacity * 32,
    )
    return (
        bytes(header)
        + struct.pack(
            f"<{width * height}H", *([0] * (width * height))
        )
        + bytes(tile_capacity * 32)
    )


def synthetic_table(source: bytes) -> dict:
    mapping = static_graphics.tile_map(source)
    return {
        "file": "synthetic.bin",
        "format": "settings_graphics",
        "source": {
            "sha256": hashlib.sha256(source).hexdigest(),
            "size": len(source),
            "width_tiles": mapping.width_tiles,
            "height_tiles": mapping.height_tiles,
            "screen_offset": mapping.screen_offset,
            "gfx_offset": mapping.gfx_offset,
            "gfx_len": mapping.gfx_len,
        },
        "solid_labels": [
            {
                "text": "A",
                "x": 0,
                "y": 0,
                "clear": {
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 12,
                    "background": 0,
                },
                "palette_indices": {"stroke": 15, "shadow": 2},
            }
        ],
    }


class SettingsGraphicsTest(unittest.TestCase):
    def test_semantic_label_is_repainted_without_file_growth(self):
        source = synthetic_resource()
        atlas = bytes([1]) + bytes(static_graphics.ATLAS_CELL_BYTES - 1)
        output = settings_graphics.repaint_settings(
            source,
            synthetic_table(source),
            atlas=atlas,
            char_slots={"A": 0},
        )
        canvas, _ = static_graphics.decode_index_canvas(output)
        self.assertEqual(canvas[0][0], 15)
        self.assertEqual(len(output), len(source))

    def test_wrong_source_hash_is_rejected(self):
        source = synthetic_resource()
        table = synthetic_table(source)
        table["source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source sha256"):
            settings_graphics.repaint_settings(
                source,
                table,
                atlas=bytes(static_graphics.ATLAS_CELL_BYTES),
                char_slots={"A": 0},
            )

    def test_preserved_region_catches_an_accidental_repaint(self):
        source = synthetic_resource()
        table = synthetic_table(source)
        table["preserve_regions"] = [
            {"what": "protected", "x": 0, "y": 0, "width": 8, "height": 8}
        ]
        atlas = bytes([1]) + bytes(static_graphics.ATLAS_CELL_BYTES - 1)
        with self.assertRaisesRegex(ValueError, "preserved region"):
            settings_graphics.repaint_settings(
                source, table, atlas=atlas, char_slots={"A": 0}
            )

    def test_pointed_label_must_clear_all_recorded_source_ink(self):
        source = synthetic_resource(width=4, height=2, tile_capacity=16)
        table = synthetic_table(source)
        table["solid_labels"] = []
        table["pointed_labels"] = [
            {
                "text": "A",
                "x": 8,
                "y": 2,
                "body": {
                    "x": 8,
                    "y": 2,
                    "width": 12,
                    "height": 12,
                    "row_backgrounds": [0] * 12,
                },
                "source_ink_bounds": {
                    "x": 7,
                    "y": 2,
                    "width": 14,
                    "height": 12,
                },
                "palette_indices": {"stroke": 15, "shadow": 2},
            }
        ]
        with self.assertRaisesRegex(
            ValueError, "does not cover the full source ink bounds"
        ):
            settings_graphics.repaint_settings(
                source,
                table,
                atlas=bytes(static_graphics.ATLAS_CELL_BYTES),
                char_slots={"A": 0},
            )

    def test_pair_is_registered_and_descriptions_are_functional(self):
        self.assertEqual(
            data_files.DATA_FILE_TABLES["3e3.bin"], "graphics/3e3.json"
        )
        self.assertEqual(
            data_files.DATA_FILE_TABLES["3e4.bin"], "graphics/3e4.json"
        )
        base = json.loads((GRAPHICS / "3e3.json").read_text(encoding="utf-8"))
        focus = json.loads((GRAPHICS / "3e4.json").read_text(encoding="utf-8"))
        descriptions = {
            label["id"]: label["text"]
            for label in base["transparent_labels"]
        }
        self.assertEqual(descriptions["text_speed"], "文字显示速度")
        self.assertEqual(descriptions["page_speed"], "画面切换速度")
        self.assertEqual(descriptions["message_skip"], "B键跳过消息")
        self.assertEqual(descriptions["command_confirmation"], "指令执行确认")
        self.assertEqual(descriptions["battle_instruction"], "战斗指示")
        self.assertEqual(
            descriptions["other_battle_animation"], "显示我军以外战斗动画"
        )
        self.assertEqual(
            descriptions["own_battle_animation_new_game_plus"],
            "显示我军战斗动画",
        )
        confirm = next(
            label for label in base["pointed_labels"] if label["id"] == "confirm"
        )
        body = confirm["body"]
        source_ink = confirm["source_ink_bounds"]
        # JP 決定 ink occupies x=203..228, y=174..185.  The whole source
        # raster must be replaced, not just the 24 px Chinese text width.
        self.assertEqual(
            (
                source_ink["x"],
                source_ink["y"],
                source_ink["width"],
                source_ink["height"],
            ),
            (203, 174, 26, 12),
        )
        self.assertLessEqual(body["x"], source_ink["x"])
        self.assertGreaterEqual(
            body["x"] + body["width"],
            source_ink["x"] + source_ink["width"],
        )
        self.assertLessEqual(body["y"], source_ink["y"])
        self.assertGreaterEqual(
            body["y"] + body["height"],
            source_ink["y"] + source_ink["height"],
        )
        preserved = " ".join(
            region["what"] for region in focus["preserve_regions"]
        )
        self.assertIn("normal/high-speed", preserved)

    def test_button_rebuild_restores_lower_bevel_below_text(self):
        source = synthetic_resource(width=4, height=2, tile_capacity=16)
        table = synthetic_table(source)
        table["solid_labels"] = []
        table["button_styles"] = {
            "light": {
                "width": 32,
                "height": 16,
                "text_y": 2,
                "frame_indices": {
                    "outside": 0,
                    "outer": 2,
                    "middle": 3,
                    "inner_border": 4,
                    "background": 5,
                },
                "palette_indices": {"stroke": 8, "shadow": 2},
            }
        }
        table["button_groups"] = [
            {
                "style": "light",
                "origins": [{"x": 0, "y": 0}],
                "buttons": [{"text": "A", "dx": 0}],
            }
        ]
        shadow_only_glyph = bytes(
            [0xAA] * static_graphics.ATLAS_CELL_BYTES
        )
        output = settings_graphics.repaint_settings(
            source,
            table,
            atlas=shadow_only_glyph,
            char_slots={"A": 0},
        )
        canvas, _ = static_graphics.decode_index_canvas(output)
        self.assertEqual(canvas[13], [2, 3] + [4] * 28 + [3, 2])
        self.assertIn(2, canvas[12][10:22])


if __name__ == "__main__":
    unittest.main()
