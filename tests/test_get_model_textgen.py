import os
import shutil
import unittest

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
TEXTGEN_GET_MODEL = ROOT / "serverless-text-generation" / "get_model.py"


spec = spec_from_file_location("textgen_get_model", TEXTGEN_GET_MODEL)
module = module_from_spec(spec)
spec.loader.exec_module(module)

get_model = module.get_model
get_tokenizer = module.get_tokenizer


class TestTextGenerationGetModel(unittest.TestCase):
    def setUp(self):
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def test_get_model_bad_name(self):
        with self.assertRaises(Exception):
            get_model("iuuiiu")

    def test_get_model_correct(self):
        get_model("distilgpt2")
        self.assertTrue((MODEL_DIR / "config.json").exists())

    def test_get_tokenizer_bad_name(self):
        with self.assertRaises(Exception):
            get_tokenizer("iuuiiu")

    def test_get_tokenizer_correct(self):
        get_tokenizer("distilgpt2")
        self.assertTrue((MODEL_DIR / "tokenizer_config.json").exists())


if __name__ == "__main__":
    unittest.main()
