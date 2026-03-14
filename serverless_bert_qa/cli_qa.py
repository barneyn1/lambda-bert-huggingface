import argparse
import json
from handler import question_answering_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()

    answer = question_answering_pipeline(
        question=args.question,
        context=args.context,
    )

    print(json.dumps({"answer": answer}, indent=2))

if __name__ == "__main__":
    main()
