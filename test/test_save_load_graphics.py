"""Unit and schema tests for the shared save/load screen resource."""
from __future__ import annotations

import hashlib
import json
import struct
import unittest
from pathlib import Path

from utils import (
    battle_system_graphics,
    data_files,
    save_load_graphics,
    static_graphics,
)


REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "data" / "zh" / "files" / "graphics" / "c34.json"


def synthetic_resource() -> tuple[bytes, dict]:
    decoded = bytes(4 * 32)
    compressed = battle_system_graphics.compress_lzss_optimal(decoded)
    graphics_offset = 4
    palette_offset = 0x40
    layout_start = 0x64
    pointer_table_offset = 0x70
    size = 0x7C
    if len(compressed) > palette_offset - 8:
        raise AssertionError("synthetic LZSS payload does not fit")

    source = bytearray(size)
    struct.pack_into("<I", source, 0, pointer_table_offset)
    struct.pack_into(
        "<I", source, graphics_offset, 0x80000000 | len(compressed)
    )
    source[8:8 + len(compressed)] = compressed
    struct.pack_into("<I", source, palette_offset, 1)
    struct.pack_into("<BBH4H", source, layout_start, 2, 2, 0, 0, 0, 0, 0)
    struct.pack_into(
        "<3I",
        source,
        pointer_table_offset,
        graphics_offset,
        palette_offset,
        layout_start,
    )
    source_bytes = bytes(source)
    table = {
        "file": "synthetic.bin",
        "format": "save_load_graphics",
        "source": {
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "size": size,
            "pointer_table_offset": pointer_table_offset,
            "pointer_count": 3,
            "pointer_table_sha256": hashlib.sha256(
                source_bytes[pointer_table_offset:]
            ).hexdigest(),
            "graphics_offset": graphics_offset,
            "graphics_descriptor": 0x80000000 | len(compressed),
            "graphics_region_sha256": hashlib.sha256(
                source_bytes[graphics_offset:palette_offset]
            ).hexdigest(),
            "decoded_size": len(decoded),
            "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
            "tile_capacity": 4,
            "palette_offset": palette_offset,
            "palette_region_sha256": hashlib.sha256(
                source_bytes[palette_offset:layout_start]
            ).hexdigest(),
            "layout_start": layout_start,
            "layout_end": pointer_table_offset,
            "layout_region_sha256": hashlib.sha256(
                source_bytes[layout_start:pointer_table_offset]
            ).hexdigest(),
            "first_layout": 2,
            "last_layout": 2,
        },
        "labels": [
            {
                "id": "test",
                "layout": 2,
                "text": "A",
                "box": {"x": 0, "y": 0, "width": 16, "height": 16},
                "background": 0,
                "palette_indices": {"stroke": 15, "shadow": 2},
            }
        ],
    }
    return source_bytes, table


class SaveLoadGraphicsTest(unittest.TestCase):
    def test_semantic_repaint_round_trips_inside_fixed_container(self):
        source, table = synthetic_resource()
        atlas = bytes([1]) + bytes(static_graphics.ATLAS_CELL_BYTES - 1)
        output = save_load_graphics.repaint_save_load(
            source, table, atlas=atlas, char_slots={"A": 0}
        )
        self.assertEqual(len(output), len(source))
        self.assertNotEqual(output, source)
        facts = save_load_graphics.resource_facts(output, table)
        self.assertLessEqual(
            facts["used_tiles"], facts["tile_capacity"]
        )
        self.assertLessEqual(
            facts["compressed_size"], facts["compressed_capacity"]
        )

    def test_wrong_source_hash_is_rejected(self):
        source, table = synthetic_resource()
        table["source"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source: sha256"):
            save_load_graphics.repaint_save_load(
                source,
                table,
                atlas=bytes(static_graphics.ATLAS_CELL_BYTES),
                char_slots={"A": 0},
            )

    def test_c34_is_registered_with_complete_visible_labels(self):
        self.assertEqual(
            data_files.DATA_FILE_TABLES["c34.bin"], "graphics/c34.json"
        )
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        labels = {label["id"]: label for label in spec["labels"]}
        expected = {
            "header_data": "数据",
            "header_save_base": "保存",
            "session": "章节",
            "turn": "回合",
            "play_time": "游戏时间",
            "header_save_overlay": "保存",
            "header_load_overlay": "读取",
            "before_start": "开始前",
            "normal": "普通",
            "special": "SP",
            "cleared": "通关",
        }
        self.assertEqual(
            {name: labels[name]["text"] for name in expected}, expected
        )
        self.assertEqual(len(labels["session"]["boxes"]), 3)
        self.assertEqual(len(labels["turn"]["boxes"]), 3)
        self.assertEqual(len(labels["play_time"]["boxes"]), 3)
        self.assertEqual(labels["special"]["advance"], 6)

    def test_expected_output_fits_all_fixed_capacities(self):
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        source = spec["source"]
        expected = spec["expected"]
        compressed_capacity = (
            int(source["palette_offset"], 0)
            - int(source["graphics_offset"], 0)
            - 4
        )
        self.assertEqual(
            expected["used_tiles"], source["tile_capacity"]
        )
        self.assertEqual(
            compressed_capacity - expected["compressed_size"], 178
        )
        self.assertEqual(len(expected["file_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
