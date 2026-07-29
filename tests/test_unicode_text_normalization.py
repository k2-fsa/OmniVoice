import importlib.util
import unittest
import unicodedata
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "omnivoice" / "utils" / "text.py"
SPEC = importlib.util.spec_from_file_location("omnivoice_text", MODULE_PATH)
text_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(text_utils)

normalize_text_input = text_utils.normalize_text_input

BASE = "Khánh Huyền khuyên Quỳnh chuyển chuyến tàu đến Huế vào chiều thứ Năm."


class UnicodeTextNormalizationTest(unittest.TestCase):
    def test_clean_vietnamese_and_ascii_are_unchanged(self):
        self.assertEqual(normalize_text_input(BASE), BASE)
        self.assertEqual(normalize_text_input("Hello, world!"), "Hello, world!")

    def test_nfd_vietnamese_becomes_nfc(self):
        self.assertEqual(normalize_text_input(unicodedata.normalize("NFD", BASE)), BASE)

    def test_unusual_spaces_become_ascii_spaces(self):
        for special_space in ("\u00a0", "\u202f", "\u2007"):
            with self.subTest(codepoint=ord(special_space)):
                self.assertEqual(
                    normalize_text_input(f"one{special_space}two"), "one two"
                )

    def test_invisible_format_characters_are_removed(self):
        for invisible in ("\u200b", "\u2060", "\ufeff"):
            with self.subTest(codepoint=ord(invisible)):
                self.assertEqual(normalize_text_input(f"one{invisible}two"), "onetwo")
        self.assertEqual(normalize_text_input("Hu\u200byền"), "Huyền")

    def test_horizontal_whitespace_collapses_and_edges_are_stripped(self):
        self.assertEqual(normalize_text_input(" \t one\t  two \v "), "one two")

    def test_line_boundaries_are_preserved(self):
        self.assertEqual(
            normalize_text_input(" one  two\n three\t four\r\nfive "),
            "one two\n three four\r\nfive",
        )

    def test_normalization_is_idempotent(self):
        dirty = " \ufeffKhánh\u00a0\u00a0Hu\u200byền\n"
        once = normalize_text_input(dirty)
        self.assertEqual(normalize_text_input(once), once)

    def test_empty_and_invisible_only_text_normalizes_to_empty(self):
        self.assertEqual(normalize_text_input(" \t\u200b\u2060\ufeff "), "")

    def test_all_experiment_inputs_normalize_identically(self):
        cases = [
            unicodedata.normalize("NFC", BASE),
            unicodedata.normalize("NFD", BASE),
            BASE.replace(" ", "\u00a0"),
            BASE.replace("Huyền", "Hu\u200byền"),
        ]
        self.assertEqual([normalize_text_input(case) for case in cases], [BASE] * 4)

    def test_reference_transcript_uses_same_normalizer(self):
        ref_text = " Khánh\u00a0Hu\u200byền "
        self.assertEqual(normalize_text_input(ref_text), "Khánh Huyền")


if __name__ == "__main__":
    unittest.main()
