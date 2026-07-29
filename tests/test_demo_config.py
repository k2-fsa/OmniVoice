import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "omnivoice" / "cli" / "_demo_config.py"
SPEC = importlib.util.spec_from_file_location("demo_config", MODULE_PATH)
demo_config = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(demo_config)

ensure_output_dir = demo_config.ensure_output_dir
env_bool = demo_config.env_bool
env_port = demo_config.env_port
env_value = demo_config.env_value


class DemoConfigTest(unittest.TestCase):
    def test_environment_value_and_default(self):
        with patch.dict(os.environ, {"OMNIVOICE_DEVICE": "cpu"}, clear=True):
            self.assertEqual(env_value("OMNIVOICE_DEVICE", "cuda"), "cpu")
            self.assertEqual(
                env_value("OMNIVOICE_MODEL", "default-model"), "default-model"
            )

    def test_port_validation(self):
        with patch.dict(os.environ, {"OMNIVOICE_PORT": "8001"}, clear=True):
            self.assertEqual(env_port(), 8001)
        with patch.dict(os.environ, {"OMNIVOICE_PORT": "70000"}, clear=True):
            with self.assertRaisesRegex(ValueError, "between 1 and 65535"):
                env_port()

    def test_boolean_validation(self):
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {"OMNIVOICE_NORMALIZE_TEXT": value}, clear=True
                ):
                    self.assertTrue(env_bool("OMNIVOICE_NORMALIZE_TEXT"))
        for value in ("0", "false", "no", "off"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {"OMNIVOICE_NORMALIZE_TEXT": value}, clear=True
                ):
                    self.assertFalse(env_bool("OMNIVOICE_NORMALIZE_TEXT", True))
        with patch.dict(
            os.environ, {"OMNIVOICE_NORMALIZE_TEXT": "sometimes"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                env_bool("OMNIVOICE_NORMALIZE_TEXT")

    def test_output_directory_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "outputs"
            self.assertEqual(ensure_output_dir(str(output_dir)), output_dir)
            self.assertTrue(output_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
