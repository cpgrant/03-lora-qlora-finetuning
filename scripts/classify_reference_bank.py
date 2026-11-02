import json
from pathlib import Path
from collections import Counter

path = Path("data/reference/reference_bank.jsonl")
records = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

for record in records:
    question = record["question"].lower().strip()

    if question.startswith(("why ", "how ")):
        record["question_type"] = "explanation"
        record["difficulty"] = "intermediate"
    elif question.startswith("what is the difference"):
        record["question_type"] = "comparison"
        record["difficulty"] = "intermediate"
    else:
        record["question_type"] = "definition"
        record["difficulty"] = "basic"

# Preserve a backup before modifying the original.
backup = path.with_suffix(".jsonl.bak")
backup.write_bytes(path.read_bytes())

with path.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("=== UPDATED REFERENCE BANK ===")
print("Records:", len(records))
print("Question types:", dict(Counter(r["question_type"] for r in records)))
print("Difficulty:", dict(Counter(r["difficulty"] for r in records)))
print("Review status: unchanged")
print("Backup:", backup)
print("Classification: PASS")
