"""Schema and glyph-safety tests for chapter titles and the load screen."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

from utils import text_codec
from utils.extract import layout as L


REPO = Path(__file__).resolve().parent.parent
TITLE_PATH = REPO / "data/zh/placements/stage_titles.json"
LABEL_PATH = REPO / "data/zh/placements/post_dict_labels.json"
GLYPH_PLAN_PATH = REPO / "data/font/stage_title_glyphs.json"
ATLAS_PATH = REPO / "data/font/atlas12.bin"
CODE_PATCH_PATH = REPO / "data/patches/code_patches.json"
RENDERB_CHARSET_PATH = REPO / "data/renderb_charset.json"


def load_glyph_helper():
    path = REPO / "build/apply_stage_title_glyphs.py"
    spec = importlib.util.spec_from_file_location("stage_title_glyph_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def decode_renderb_prefix(payload: bytes) -> str:
    slots = json.loads(
        RENDERB_CHARSET_PATH.read_text(encoding="utf-8")
    )["slots"]
    output: list[str] = []
    for offset, token, length in text_codec.iter_tokens(payload):
        if length == 1:
            slot = token
        else:
            slot = token - 0xE000 + text_codec.TWO_BYTE_SLOT_OFFSET
        if slot >= text_codec.ZH_BAND_MIN:
            output.append(
                text_codec.decode(payload[offset:offset + length])
            )
        else:
            output.append(slots[str(slot)]["char"])
    return "".join(output)


class StageTitleTest(unittest.TestCase):
    def test_all_four_descriptor_fields_are_owned_once(self):
        document = json.loads(TITLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(document["counts"]["records"], 101)
        self.assertEqual(document["counts"]["entries"], 356)
        self.assertEqual(document["counts"]["chapter_card_entries"], 178)
        self.assertEqual(document["counts"]["save_slot_entries"], 178)
        self.assertEqual(document["counts"]["pointer_sites"], 404)
        self.assertEqual(document["counts"]["extra_pointer_entries"], 1)
        self.assertEqual(document["counts"]["extra_pointer_sites"], 3)

        fields = {
            ("save_slot", "stage_title_prefix"): L.STAGE_DESC_LABEL,
            ("save_slot", "stage_title_stream"): L.STAGE_DESC_TITLE,
            ("chapter_card", "stage_title_prefix"): L.STAGE_DESC_CARD_LABEL,
            ("chapter_card", "stage_title_stream"): L.STAGE_DESC_CARD_TITLE,
        }
        expected_sites = {
            L.STAGE_DESC + record * L.STAGE_DESC_STRIDE + field
            for record in range(L.STAGE_DESC_N)
            for field in fields.values()
        }
        seen: set[int] = set()
        unique_payloads: set[tuple[int, bytes]] = set()
        shared_title_cap = document["display_limits"]["shared_save_slot_cap"]
        for entry in document["entries"]:
            key = (entry["view"], entry["domain"])
            self.assertIn(key, fields)
            payload = bytes.fromhex(entry["payload_hex"])
            self.assertEqual(payload[-1], 0)
            expects_renderb_prefix = (
                entry["view"] == "save_slot"
                and entry["domain"] == "stage_title_prefix"
            )
            is_renderb_prefix = entry["encoding"] == "renderb_identity+zh"
            self.assertEqual(is_renderb_prefix, expects_renderb_prefix)
            self.assertEqual(
                (
                    decode_renderb_prefix(payload[:-1])
                    if is_renderb_prefix
                    else text_codec.decode(payload[:-1])
                ),
                entry["zh"],
            )
            self.assertEqual(
                entry["encoding"],
                "renderb_identity+zh" if is_renderb_prefix else "text_codec",
            )
            offset = int(entry["offset"], 0)
            unique_payloads.add((offset, payload))
            for site_text in entry["sites"]:
                site = int(site_text, 0)
                self.assertNotIn(site, seen)
                record, within = divmod(site - L.STAGE_DESC, L.STAGE_DESC_STRIDE)
                self.assertIn(record, range(L.STAGE_DESC_N))
                self.assertEqual(within, fields[key])
                seen.add(site)

            if entry["view"] == "save_slot":
                glyphs = 0
                for offset, token, length in text_codec.iter_tokens(payload[:-1]):
                    if token == 0x01:
                        continue
                    glyphs += 1
                    slot = (
                        token
                        if length == 1
                        else token - 0xE000 + text_codec.TWO_BYTE_SLOT_OFFSET
                    )
                    if is_renderb_prefix:
                        if slot < text_codec.ZH_BAND_MIN:
                            self.assertTrue(
                                decode_renderb_prefix(
                                    payload[offset:offset + length]
                                ).isascii(),
                                f"{entry['id']} uses a non-identity renderB token",
                            )
                    else:
                        self.assertEqual(
                            length,
                            2,
                            f"{entry['id']} contains renderB one-byte glyph {token:#x}",
                        )
                        self.assertGreaterEqual(
                            slot,
                            text_codec.ZH_BAND_MIN,
                            f"{entry['id']} mixes a renderB JP-band glyph",
                        )
                self.assertLessEqual(
                    glyphs,
                    shared_title_cap,
                    f"{entry['id']} exceeds the shared load/battle title budget",
                )

        self.assertEqual(seen, expected_sites)
        self.assertEqual(
            len(unique_payloads), document["counts"]["unique_payloads"]
        )

    def test_free_battle_load_fallback_reuses_the_bank_safe_title(self):
        document = json.loads(TITLE_PATH.read_text(encoding="utf-8"))
        extras = document["extra_pointer_entries"]
        self.assertEqual(len(extras), 1)
        fallback = extras[0]
        self.assertEqual(fallback["id"], "LOAD_FREE_BATTLE_FALLBACK")
        self.assertEqual(fallback["zh"], "自由战斗")
        self.assertEqual(fallback["surface"], "bank")
        self.assertEqual(fallback["old_ptr"], "0x0214B00B")
        self.assertEqual(
            {int(site, 0) for site in fallback["sites"]},
            {0x32550, 0x3B098, 0x8540C},
        )
        canonical = next(
            entry
            for entry in document["entries"]
            if entry["id"] == "SLOT_UTXT02048"
        )
        self.assertEqual(fallback["ptr"], canonical["ptr"])
        self.assertEqual(fallback["payload_hex"], canonical["payload_hex"])
        self.assertEqual(
            text_codec.decode(bytes.fromhex(fallback["payload_hex"])[:-1]),
            "自由战斗",
        )

    def test_reviewed_corpus_and_only_recorded_slot_compactions_are_pinned(self):
        document = json.loads(TITLE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            document["source"]["unique_tsv_sha256"],
            "0434b28f6c52fe6548eace809991f3366b8cd1df90815888280ebc0d85b62ad7",
        )
        self.assertEqual(
            document["source"]["record_tsv_sha256"],
            "d78098194b92a22ec241b55ec85e64d8ec3525868a113da3cb02751976c142c9",
        )
        adjusted = {
            entry["id"]: entry["zh"]
            for entry in document["entries"]
            if entry["view"] == "save_slot"
            and any(note.startswith("save-slot") for note in entry["adjustments"])
        }
        self.assertEqual(
            adjusted,
            {
                "SLOT_UTXT01974": "撼动宇宙",
                "SLOT_UTXT01975": "撼动宇宙{01}另一侧",
                "SLOT_UTXT02029": "独眼高达{01}后篇",
                "SLOT_UTXT02036": "那黑暗之名乃是木星",
                "SLOT_UTXT02041": "特别演习（隆德贝尔）",
            },
        )

    def test_measured_load_and_battle_title_limits_are_pinned(self):
        document = json.loads(TITLE_PATH.read_text(encoding="utf-8"))
        limits = document["display_limits"]
        self.assertEqual(
            limits["load_screen"],
            {
                "title_x": 48,
                "right_boundary_x": 200,
                "engine_max_glyphs": 26,
                "safe_fullwidth_glyphs": 12,
            },
        )
        self.assertEqual(
            limits["battle_terrain_page"],
            {
                "title_x": 96,
                "right_boundary_x": 216,
                "engine_max_glyphs": 10,
                "safe_fullwidth_glyphs": 10,
            },
        )
        self.assertEqual(limits["shared_save_slot_cap"], 10)

    def test_special_chapter_codes_use_compact_renderb_identities(self):
        document = json.loads(TITLE_PATH.read_text(encoding="utf-8"))
        prefixes = {
            entry["zh"]: entry
            for entry in document["entries"]
            if entry["view"] == "save_slot"
            and entry["domain"] == "stage_title_prefix"
        }
        for text in (
            "SP1A",
            "SP3A",
            "SP8S",
            "X1前",
            "X3A",
            "X7",
            "TR1",
            "TU",
            "TL",
            "FB",
        ):
            entry = prefixes[text]
            payload = bytes.fromhex(entry["payload_hex"])[:-1]
            self.assertEqual(entry["encoding"], "renderb_identity+zh")
            self.assertEqual(decode_renderb_prefix(payload), text)
            width = 0
            for _offset, token, length in text_codec.iter_tokens(payload):
                slot = (
                    token
                    if length == 1
                    else token - 0xE000 + text_codec.TWO_BYTE_SLOT_OFFSET
                )
                width += 8 if slot < text_codec.ZH_BAND_MIN else 12
            self.assertLessEqual(
                width,
                40,
                f"{text} overlaps the battle title starting at x=96",
            )

    def test_session_labels_and_load_prompts_are_fixed_length_asserted(self):
        table = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
        entries = {int(entry["offset"], 0): entry for entry in table["entries"]}
        expected = {
            0x24F: "章节",
            0x2D2: "没有存档",
            0x614: "泛",
            0x61A: "空",
            0x6A9: "章节",
            0x6B0: "进入下一章节",
            0x74D: "保存中",
            0x754: "读取中",
            0x75B: "请勿关闭电源",
            0x769: "保存失败",
            0x77F: "完成",
            0x785: "完成",
            0x78B: "保存",
            0x791: "读取",
            0x797: "取消",
            0x79D: "保存数据吗？",
            0x7AE: "读取失败",
            0x7BA: "请关机并将卡",
            0x7C8: "重新插入",
        }
        for offset, text in expected.items():
            entry = entries[offset]
            self.assertEqual(entry["text"], text)
            self.assertEqual(
                len(bytes.fromhex(entry["payload_hex"])),
                len(bytes.fromhex(entry["old_hex"])),
            )
        self.assertEqual(entries[0x6A9]["leading_blanks"], 2)
        self.assertEqual(
            bytes.fromhex(entries[0x6A9]["payload_hex"]),
            b"\x01\x01" + text_codec.encode("章节") + b"\x00",
        )
        for offset, text in ((0x614, "泛"), (0x61A, "空"), (0x6B0, "进入下一章节")):
            payload = bytes.fromhex(entries[offset]["payload_hex"]).split(b"\0", 1)[0]
            self.assertEqual(text_codec.decode(payload), text)

    def test_battle_terrain_page_visually_joins_session_label_and_number(self):
        patches = json.loads(CODE_PATCH_PATH.read_text(encoding="utf-8"))
        matches = [
            entry
            for entry in patches["entries"]
            if int(entry["file_offset"], 0) == 0x3AEB0
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["old_hex"], "1021")  # movs r1, #16
        self.assertEqual(matches[0]["new_hex"], "2021")  # movs r1, #32

    def test_base_atlas_composes_to_the_pinned_glyph_plan(self):
        helper = load_glyph_helper()
        plan = json.loads(GLYPH_PLAN_PATH.read_text(encoding="utf-8"))
        helper._verify_charmap(plan)
        helper._verify_move_scope(plan)
        atlas = ATLAS_PATH.read_bytes()
        self.assertEqual(helper.sha256(atlas), plan["source_atlas_sha256"])
        effective = helper.build_target(atlas, plan)
        self.assertEqual(
            helper.sha256(effective),
            plan["target_atlas_sha256"],
        )
        self.assertEqual(len(plan["moves"]), 9)
        self.assertEqual(len(plan["remaps"]), 0)
        self.assertEqual(len(plan["mints"]), 8)
        self.assertEqual(len(plan["promotions"]), 3)
        self.assertEqual(len(plan["native_reuses"]), 1)

    def test_glyph_plan_accepts_disjoint_layers_but_rejects_cell_collisions(self):
        helper = load_glyph_helper()
        plan = json.loads(GLYPH_PLAN_PATH.read_text(encoding="utf-8"))
        atlas = ATLAS_PATH.read_bytes()

        disjoint = bytearray(atlas)
        disjoint[0] ^= 1
        composed = helper.build_target(bytes(disjoint), plan)
        self.assertEqual(composed[0], disjoint[0])

        collision = bytearray(atlas)
        target = int(plan["moves"][4]["to_slot"])
        collision[target * helper.CELL_BYTES] ^= 1
        with self.assertRaisesRegex(ValueError, "target drifted"):
            helper.build_target(bytes(collision), plan)


if __name__ == "__main__":
    unittest.main()
