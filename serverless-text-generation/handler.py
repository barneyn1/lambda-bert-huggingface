import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tokenizer, prompt: str):
    """Encode prompt into input IDs."""
    return tokenizer(prompt, return_tensors="pt").input_ids


def decode(tokenizer, token_ids):
    """Decode generated token IDs back to text."""
    outputs = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return outputs[0].strip()


def serverless_pipeline(model_path="./model"):
    """Load model/tokenizer once and return a reusable predict function."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    def predict(prompt: str, max_length: int = 60) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("The 'prompt' field is required.")

        input_ids = encode(tokenizer, prompt)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                do_sample=False,
                max_length=max_length,
            )

        return decode(tokenizer, output_ids)

    return predict


# Load once at import time
text_generating_pipeline = serverless_pipeline()


def predict_text(prompt: str, max_length: int = 60) -> dict:
    """Direct Python entrypoint for local/CLI use."""
    answer = text_generating_pipeline(prompt=prompt, max_length=max_length)
    return {"answer": answer}


def handler(event, context):
    """AWS Lambda entrypoint."""
    try:
        raw_body = event.get("body", "{}")

        if isinstance(raw_body, str):
            body = json.loads(raw_body)
        elif isinstance(raw_body, dict):
            body = raw_body
        else:
            raise ValueError("event['body'] must be a JSON string or dict")

        prompt = body.get("prompt", "")
        max_length = int(body.get("max_length", 60))

        result = predict_text(prompt=prompt, max_length=max_length)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps(result),
        }

    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Credentials": True,
            },
            "body": json.dumps({"error": repr(e)}),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_length", type=int, default=60)
    args = parser.parse_args()

    print(json.dumps(
        predict_text(prompt=args.prompt, max_length=args.max_length),
        indent=2
    ))
