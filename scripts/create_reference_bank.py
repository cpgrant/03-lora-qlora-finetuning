import ast
import json
from pathlib import Path

source = Path("scripts/build_corpus.py").read_text()

tree = ast.parse(source)

records = None
for node in tree.body:
    if isinstance(node, ast.Assign):
        if any(
            isinstance(target, ast.Name) and target.id == "records"
            for target in node.targets
        ):
            records = ast.literal_eval(node.value)
            break

if records is None:
    raise RuntimeError("Could not find records in build_corpus.py")

output = Path("data/reference/reference_bank.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8") as f:
    for i, (domain, concept, question, answer, split) in enumerate(records, 1):
        example = {
            "id": f"AI-P03-{i:04d}",
            "domain": domain,
            "concept": concept,
            "question_type": "definition",
            "difficulty": "basic",
            "question": question,
            "reference_answer": answer,
            "review_status": "pending",
            "split": split,
        }
        f.write(json.dumps(example, ensure_ascii=False) + "\n")

print("=== REFERENCE BANK ===")
print("Records:", len(records))
print("Output:", output)
print("Status: PASS")
