import argparse
import json
from handler import predict_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_length", type=int, default=60)
    args = parser.parse_args()

    print(json.dumps(
        predict_text(prompt=args.prompt, max_length=args.max_length),
        indent=2
    ))

if __name__ == "__main__":
    main()
