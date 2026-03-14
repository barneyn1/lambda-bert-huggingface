import argparse
import json
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_length", type=int, default=60)
    args = parser.parse_args()

    payload = {
        "prompt": args.prompt,
        "max_length": args.max_length,
    }

    response = requests.post(args.url, json=payload, timeout=120)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    main()
