"""Tests for Malayalam text normalization.

Covers:
- Basic number conversion (0–9999+)
- Currency (₹) normalization
- Percentage (%) normalization
- Unit (km, kg, etc.) normalization
- Mixed Malayalam text with embedded numbers
- Word boundary safety (no false positives)
- Edge cases
- Language dispatch via normalize_text()

Note: Imports are done directly from submodules to avoid pulling in
torch/transformers via omnivoice.__init__.py, keeping tests lightweight.
"""

import sys
import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the text normalization modules directly by file path, bypassing
# omnivoice/__init__.py which imports torch/transformers.
# This keeps the tests lightweight and runnable without the full ML stack.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_TN_DIR = _ROOT / "omnivoice" / "utils" / "text_normalization"


def _load_module(name: str, filepath: Path):
    """Load a Python module directly from a file path."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load malayalam.py first (it has no intra-package dependencies)
_ml_mod = _load_module(
    "omnivoice.utils.text_normalization.malayalam",
    _TN_DIR / "malayalam.py",
)

# Load __init__.py (it imports from .malayalam, which is now in sys.modules)
_tn_mod = _load_module(
    "omnivoice.utils.text_normalization",
    _TN_DIR / "__init__.py",
)

normalize_text = _tn_mod.normalize_text
normalize_malayalam = _ml_mod.normalize_malayalam
_number_to_malayalam = _ml_mod._number_to_malayalam


# ---------------------------------------------------------------------------
# Number-to-word conversion (_number_to_malayalam)
# ---------------------------------------------------------------------------


class TestNumberToMalayalam:
    """Test the core number-to-word conversion function."""

    def test_zero(self):
        assert _number_to_malayalam(0) == "പൂജ്യം"

    def test_single_digits(self):
        assert _number_to_malayalam(1) == "ഒന്ന്"
        assert _number_to_malayalam(5) == "അഞ്ച്"
        assert _number_to_malayalam(9) == "ഒമ്പത്"

    def test_teens(self):
        assert _number_to_malayalam(10) == "പത്ത്"
        assert _number_to_malayalam(11) == "പതിനൊന്ന്"
        assert _number_to_malayalam(15) == "പതിനഞ്ച്"
        assert _number_to_malayalam(19) == "പത്തൊമ്പത്"

    def test_tens(self):
        assert _number_to_malayalam(20) == "ഇരുപത്"
        assert _number_to_malayalam(50) == "അമ്പത്"
        assert _number_to_malayalam(90) == "തൊണ്ണൂറ്"

    def test_tens_with_ones(self):
        assert _number_to_malayalam(21) == "ഇരുപത്തി ഒന്ന്"
        assert _number_to_malayalam(35) == "മുപ്പത്തി അഞ്ച്"
        assert _number_to_malayalam(99) == "തൊണ്ണൂറ്റി ഒമ്പത്"

    def test_hundreds(self):
        assert _number_to_malayalam(100) == "നൂറ്"
        assert _number_to_malayalam(200) == "ഇരുനൂറ്"
        assert _number_to_malayalam(500) == "അഞ്ഞൂറ്"
        assert _number_to_malayalam(900) == "തൊള്ളായിരം"

    def test_hundreds_with_remainder(self):
        assert _number_to_malayalam(101) == "നൂറ്റി ഒന്ന്"
        assert _number_to_malayalam(110) == "നൂറ്റി പത്ത്"
        assert _number_to_malayalam(123) == "നൂറ്റി ഇരുപത്തി മൂന്ന്"
        assert _number_to_malayalam(250) == "ഇരുനൂറ്റി അമ്പത്"
        assert _number_to_malayalam(999) == "തൊള്ളായിരത്തി തൊണ്ണൂറ്റി ഒമ്പത്"

    def test_thousands(self):
        assert _number_to_malayalam(1000) == "ആയിരം"
        assert _number_to_malayalam(2000) == "രണ്ടായിരം"
        assert _number_to_malayalam(5000) == "അയ്യായിരം"

    def test_thousands_with_remainder(self):
        assert _number_to_malayalam(1001) == "ആയിരത്തി ഒന്ന്"
        assert _number_to_malayalam(1500) == "ആയിരത്തി അഞ്ഞൂറ്"
        assert _number_to_malayalam(2500) == "രണ്ടായിരത്തി അഞ്ഞൂറ്"
        assert _number_to_malayalam(9999) == "ഒമ്പതിനായിരത്തി തൊള്ളായിരത്തി തൊണ്ണൂറ്റി ഒമ്പത്"

    def test_lakhs(self):
        assert _number_to_malayalam(100000) == "ഒരു ലക്ഷം"
        assert _number_to_malayalam(500000) == "അഞ്ച് ലക്ഷം"

    def test_lakhs_with_remainder(self):
        assert _number_to_malayalam(100001) == "ഒരു ലക്ഷത്തി ഒന്ന്"
        assert _number_to_malayalam(150000) == "ഒരു ലക്ഷത്തി അമ്പത് ആയിരം"

    def test_out_of_range(self):
        assert _number_to_malayalam(-1) is None
        assert _number_to_malayalam(10000000) is None

    def test_negative_returns_none(self):
        assert _number_to_malayalam(-5) is None


# ---------------------------------------------------------------------------
# normalize_malayalam() — full text normalization
# ---------------------------------------------------------------------------


class TestNormalizeMalayalam:
    """Test the full normalize_malayalam function."""

    # --- Currency ---

    def test_currency_basic(self):
        assert normalize_malayalam("₹100") == "നൂറ് രൂപ"

    def test_currency_with_space(self):
        assert normalize_malayalam("₹ 50") == "അമ്പത് രൂപ"

    def test_currency_in_sentence(self):
        result = normalize_malayalam("എനിക്ക് ₹100 ഉണ്ട്")
        assert result == "എനിക്ക് നൂറ് രൂപ ഉണ്ട്"

    def test_currency_250(self):
        result = normalize_malayalam("₹250")
        assert result == "ഇരുനൂറ്റി അമ്പത് രൂപ"

    # --- Percentages ---

    def test_percentage(self):
        assert normalize_malayalam("5%") == "അഞ്ച് ശതമാനം"

    def test_percentage_in_sentence(self):
        result = normalize_malayalam("10% കുറവ്")
        assert result == "പത്ത് ശതമാനം കുറവ്"

    # --- Units ---

    def test_unit_km(self):
        assert normalize_malayalam("10km") == "പത്ത് കിലോമീറ്റർ"

    def test_unit_kg(self):
        assert normalize_malayalam("5kg") == "അഞ്ച് കിലോഗ്രാം"

    def test_unit_cm(self):
        assert normalize_malayalam("30cm") == "മുപ്പത് സെന്റീമീറ്റർ"

    def test_unit_m(self):
        assert normalize_malayalam("100m") == "നൂറ് മീറ്റർ"

    def test_unit_ml(self):
        assert normalize_malayalam("500ml") == "അഞ്ഞൂറ് മില്ലിലിറ്റർ"

    def test_unit_g(self):
        assert normalize_malayalam("250g") == "ഇരുനൂറ്റി അമ്പത് ഗ്രാം"

    # --- Standalone numbers ---

    def test_standalone_number(self):
        assert normalize_malayalam("100") == "നൂറ്"

    def test_number_in_malayalam_sentence(self):
        result = normalize_malayalam("എനിക്ക് 100 രൂപ വേണം")
        assert result == "എനിക്ക് നൂറ് രൂപ വേണം"

    def test_multiple_numbers(self):
        result = normalize_malayalam("5 ആപ്പിളും 3 ഓറഞ്ചും")
        assert result == "അഞ്ച് ആപ്പിളും മൂന്ന് ഓറഞ്ചും"

    def test_digit_by_digit_fallback_large(self):
        # 10,000,000 is out of bounds for the word converter, so it falls back to digits
        assert normalize_malayalam("10000000") == "ഒന്ന് പൂജ്യം പൂജ്യം പൂജ്യം പൂജ്യം പൂജ്യം പൂജ്യം പൂജ്യം"

    def test_digit_by_digit_phone_number(self):
        assert normalize_malayalam("9876543210") == "ഒമ്പത് എട്ട് ഏഴ് ആറ് അഞ്ച് നാല് മൂന്ന് രണ്ട് ഒന്ന് പൂജ്യം"

    def test_digit_by_digit_leading_zero(self):
        assert normalize_malayalam("0484") == "പൂജ്യം നാല് എട്ട് നാല്"

    # --- Decimals ---

    def test_decimals_simple(self):
        assert normalize_malayalam("10.5") == "പത്ത് പോയിന്റ് അഞ്ച്"

    def test_decimals_zero(self):
        assert normalize_malayalam("0.5") == "പൂജ്യം പോയിന്റ് അഞ്ച്"

    def test_decimals_multiple_digits(self):
        assert normalize_malayalam("3.14") == "മൂന്ന് പോയിന്റ് ഒന്ന് നാല്"

    def test_decimals_large(self):
        assert normalize_malayalam("100.50") == "നൂറ് പോയിന്റ് അഞ്ച് പൂജ്യം"

    # --- Time ---

    def test_time_simple(self):
        assert normalize_malayalam("5:30") == "അഞ്ച് മുപ്പത്"

    def test_time_double_digits(self):
        assert normalize_malayalam("12:45") == "പന്ത്രണ്ട് നാല്പത്തി അഞ്ച്"

    def test_time_zeros(self):
        assert normalize_malayalam("10:05") == "പത്ത് അഞ്ച്" # "പത്ത് പൂജ്യം അഞ്ച്" isn't standard, but this matches our generic logic ("10" "05" -> "10" "5" -> പത്ത് അഞ്ച്)

    # --- Fractions ---

    def test_fractions_simple(self):
        assert normalize_malayalam("1/2") == "അര"
        assert normalize_malayalam("1/4") == "കാൽ"
        assert normalize_malayalam("3/4") == "മുക്കാൽ"

    def test_fractions_mixed(self):
        assert normalize_malayalam("1 1/2") == "ഒന്ന് അര"
        assert normalize_malayalam("2 1/4") == "രണ്ട് കാൽ"

    # --- Ordinals ---

    def test_ordinals_simple(self):
        assert normalize_malayalam("1st") == "ഒന്നാമത്തെ"
        assert normalize_malayalam("2nd") == "രണ്ടാമത്തെ"
        assert normalize_malayalam("3rd") == "മൂന്നാമത്തെ"
        assert normalize_malayalam("4th") == "നാലാമത്തെ"
        
    def test_ordinals_tens(self):
        assert normalize_malayalam("10th") == "പത്താമത്തെ"
        assert normalize_malayalam("20th") == "ഇരുപതാമത്തെ"

    def test_ordinals_hundreds(self):
        assert normalize_malayalam("100th") == "നൂറാമത്തെ"

    def test_ordinals_lakhs(self):
        assert normalize_malayalam("100000th") == "ഒരു ലക്ഷാമത്തെ"

    # --- Acronyms ---

    def test_acronym_simple(self):
        assert normalize_malayalam("AICTE") == "എ ഐ സി ടി ഇ"
        assert normalize_malayalam("WHO") == "ഡബ്ല്യൂ എച്ച് ഒ"

    def test_acronym_mixed(self):
        assert normalize_malayalam("TTS model") == "ടി ടി എസ് model"

    # --- Edge cases ---

    def test_empty_string(self):
        assert normalize_malayalam("") == ""

    def test_no_numbers(self):
        text = "ഇത് ഒരു പരീക്ഷണമാണ്"
        assert normalize_malayalam(text) == text

    def test_already_normalized(self):
        text = "എനിക്ക് നൂറ് രൂപ വേണം"
        assert normalize_malayalam(text) == text

    def test_comma_separated_numbers(self):
        # Indian number format: 1,00,000
        result = normalize_malayalam("₹1,00,000")
        assert result == "ഒരു ലക്ഷം രൂപ"


# ---------------------------------------------------------------------------
# normalize_text() — language dispatch
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """Test the top-level language dispatch function."""

    def test_malayalam_dispatch(self):
        result = normalize_text("100", language="ml")
        assert result == "നൂറ്"

    def test_none_language_passthrough(self):
        assert normalize_text("100", language=None) == "100"

    def test_unknown_language_passthrough(self):
        assert normalize_text("100", language="en") == "100"

    def test_empty_text(self):
        assert normalize_text("", language="ml") == ""

    def test_malayalam_full_sentence(self):
        result = normalize_text("എനിക്ക് ₹100 ഉണ്ട്", language="ml")
        assert result == "എനിക്ക് നൂറ് രൂപ ഉണ്ട്"
