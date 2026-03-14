#Lambda-Bert-Huggingface: Stripped JS Frontend and CLI Alterations for Debloat & Deployment

QA Function:
```bash
cd serverless_bert_qa
python handler.py --question "Who developed BERT?" --context "BERT was developed by researchers at Google."
```

Text Generation Function:
```bash
cd serverless-text-generation
python get_model.py
python handler.py --prompt "Machine learning is" --max_length 50
```
