import argparse
import json
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()

    payload = {
        "question": args.question,
        "context": args.context,
    }

    r = requests.post(args.url, json=payload, timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    main()
