import json
import os
import time
import uuid

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

#Local Testing:
#python handler.py --question "Who developed BERT?" --context "BERT was developed by researchers at Google."

# Optional DynamoDB logging for deployed mode.
# Local/CLI mode can run without DYNAMODB_TABLE set.
table = None
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE")

if DYNAMODB_TABLE:
    try:
        import boto3

        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.Table(DYNAMODB_TABLE)
    except Exception as exc:
        print(f"Warning: DynamoDB disabled: {exc!r}")
        table = None


def encode(tokenizer, question, context):
    """Encode question + context into model inputs."""
    encoded = tokenizer.encode_plus(question, context)
    return encoded["input_ids"], encoded["attention_mask"]


def decode(tokenizer, token_ids):
    """Decode token IDs into a human-readable answer string."""
    answer_tokens = tokenizer.convert_ids_to_tokens(
        token_ids,
        skip_special_tokens=True,
    )
    return tokenizer.convert_tokens_to_string(answer_tokens).strip()


def serverless_pipeline(model_path="./model"):
    """Load tokenizer/model once and return a reusable predict function."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()

    def predict(question, context):
        input_ids, attention_mask = encode(tokenizer, question, context)

        with torch.no_grad():
            outputs = model(
                torch.tensor([input_ids]),
                attention_mask=torch.tensor([attention_mask]),
            )

        # Support both tuple-style and object-style transformer outputs
        if hasattr(outputs, "start_logits") and hasattr(outputs, "end_logits"):
            start_scores = outputs.start_logits[0]
            end_scores = outputs.end_logits[0]
        else:
            start_scores, end_scores = outputs
            start_scores = start_scores[0]
            end_scores = end_scores[0]

        start_idx = int(torch.argmax(start_scores))
        end_idx = int(torch.argmax(end_scores))

        if end_idx < start_idx:
            end_idx = start_idx

        answer_token_ids = input_ids[start_idx : end_idx + 1]
        answer = decode(tokenizer, answer_token_ids)
        return answer

    return predict


# Load model once at import time, same overall pattern as the repo.
question_answering_pipeline = serverless_pipeline()


def predict_qa(question: str, context: str) -> dict:
    """Direct Python entrypoint for CLI/local use."""
    if not question or not context:
        raise ValueError("Both 'question' and 'context' are required.")
    answer = question_answering_pipeline(question=question, context=context)
    return {"answer": answer}


def maybe_log_to_dynamodb(question: str, context: str, answer: str) -> None:
    """Write request/response to DynamoDB only if configured."""
    if table is None:
        return

    timestamp = str(time.time())
    item = {
        "primary_key": str(uuid.uuid1()),
        "createdAt": timestamp,
        "context": context,
        "question": question,
        "answer": answer,
    }
    table.put_item(Item=item)


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

        question = body.get("question", "")
        context_text = body.get("context", "")

        result = predict_qa(question=question, context=context_text)
        maybe_log_to_dynamodb(question, context_text, result["answer"])

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
    # Simple local CLI test:
    # python handler.py --question "Who developed BERT?" --context "BERT was developed by researchers at Google."
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()

    print(json.dumps(predict_qa(args.question, args.context), indent=2))
