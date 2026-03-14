# Question Answering NLP Applications

This altered repository strips the front-end JavaScript and focuses on python-only CLI usages:

Local CLI:
```Bash
python cli_qa.py \
  --question "Who developed BERT?" \
  --context "BERT was developed by researchers at Google."
```

Remote API via Python:
```Bash
python cli_qa.py \
  --question "Who developed BERT?" \
  --context "BERT was developed by researchers at Google."
```

Remote API via curl:
```Bash
curl -X POST "$QA_URL" \
  -H "Content-Type: application/json" \
  -d '{"question":"Who developed BERT?","context":"BERT was developed by researchers at Google."}'
```
