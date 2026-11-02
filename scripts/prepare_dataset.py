import json
from pathlib import Path
from collections import Counter

SOURCE = Path("data/smoke-test")
DEST = Path("data/v2")
DEST.mkdir(parents=True, exist_ok=True)

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def validate(records, name):
    seen = set()

    for i, record in enumerate(records, 1):
        assert "prompt" in record, f"{name}:{i} missing prompt"
        assert "completion" in record, f"{name}:{i} missing completion"

        prompt = record["prompt"][0]
        completion = record["completion"][0]

        assert prompt["role"] == "user"
        assert completion["role"] == "assistant"
        assert prompt["content"].strip()
        assert completion["content"].strip()

        question = " ".join(prompt["content"].lower().split())
        assert question not in seen, f"Duplicate: {question}"
        seen.add(question)

    return seen

train = read_jsonl(SOURCE / "train.jsonl")
eval_data = read_jsonl(SOURCE / "eval.jsonl")

train_questions = validate(train, "train")
eval_questions = validate(eval_data, "eval")

assert not (train_questions & eval_questions), \
    "Training/evaluation question overlap"

for name, records in [("train", train), ("eval", eval_data)]:
    with (DEST / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("=== AI-P03 DATASET PREPARATION ===")
print("Training records:", len(train))
print("Evaluation records:", len(eval_data))
print("Exact question overlap: 0")
print("Output directory:", DEST)
print("Validation: PASS")
print("Note: Semantic overlap and factual accuracy not yet audited.")
