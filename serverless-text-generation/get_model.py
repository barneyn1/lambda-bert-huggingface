from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_model(model_name: str):
    """Load a causal language model from Hugging Face and save it into the repo-root model/ directory."""
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cdn=True)
    model.save_pretrained(MODEL_DIR)


def get_tokenizer(tokenizer_name: str):
    """Load a tokenizer from Hugging Face and save it into the repo-root model/ directory."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    MODEL_ID = "distilgpt2"
    get_model(MODEL_ID)
    get_tokenizer(MODEL_ID)
