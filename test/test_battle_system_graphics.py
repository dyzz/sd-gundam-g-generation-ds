"""Unit and schema tests for the in-battle START-menu ARM9 writer."""
from __future__ import annotations

import hashlib
import json
import struct
import unittest
from pathlib import Path

from utils import battle_system_graphics


REPO = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO / "data" / "zh" / "battle_system_menu.json"


class BattleSystemGraphicsTest(unittest.TestCase):
    def test_lzss_optimal_round_trip_is_canonical(self):
        payload = (
            bytes(range(1, 256)) * 2
            + bytes(range(255, 0, -1))
        )
        compressed = battle_system_graphics.compress_lzss_optimal(payload)
        self.assertEqual(
            hashlib.sha256(compressed).hexdigest(),
            "fc2f0a15cb644b5b076645db586051ad5c63bd55b8e0de3d17c791bd5c07477a",
        )
        self.assertEqual(
            battle_system_graphics.decompress_lzss(compressed), payload
        )

    def test_layout_codec_round_trips_with_fixed_pointer_arena(self):
        values = (3, 3, 3, 8, 9, 10, 4, 4)
        layout = battle_system_graphics.Layout(4, 2, 0, values)
        payload = battle_system_graphics._encode_layout(layout, values)
        arm9_base = 0x02000000
        layout_start = 0x100
        layout_end = 0x180
        pointer_table = 0x200
        image = bytearray(0x300)
        image[layout_start:layout_start + len(payload)] = payload
        struct.pack_into(
            "<I", image, pointer_table + 2 * 4, arm9_base + layout_start
        )
        decoded = battle_system_graphics._decode_layout(
            image,
            2,
            arm9_base=arm9_base,
            pointer_table_offset=pointer_table,
            layout_start=layout_start,
            layout_end=layout_end,
        )
        self.assertEqual(decoded.width, 4)
        self.assertEqual(decoded.height, 2)
        self.assertEqual(decoded.values, values)

    def test_wrong_source_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            battle_system_graphics._require_hash(
                b"Japanese source",
                "0" * 64,
                where="synthetic source",
            )

    def test_spec_translates_only_runtime_proven_p1_surfaces(self):
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        translated = {
            int(index)
            for label in spec["menu_labels"]
            for index in label["indices"]
        }
        self.assertEqual(translated, {5, 6, 7, 8, 12, 13, 14, 15})
        self.assertTrue(translated.isdisjoint({9, 10, 11, 16, 17, 18}))
        self.assertEqual(spec["prompt"]["text"], "结束回合？")
        self.assertEqual(spec["confirmation"]["yes_text"], "是")
        self.assertEqual(spec["confirmation"]["no_text"], "否")

    def test_expected_output_stays_inside_all_three_fixed_capacities(self):
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        source = spec["source"]
        expected = spec["expected"]
        tile_capacity = int(source["tile_capacity"])
        compressed_capacity = int(source["graphics_descriptor"], 0) & 0xFFFF
        layout_capacity = (
            int(source["layout_end"], 0) - int(source["layout_start"], 0)
        )
        self.assertEqual(tile_capacity - expected["used_tiles"], 41)
        self.assertEqual(
            compressed_capacity - expected["compressed_size"], 275
        )
        self.assertEqual(layout_capacity - expected["layout_size"], 12)


if __name__ == "__main__":
    unittest.main()
