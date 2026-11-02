import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3-0.6B"
ADAPTER = "outputs/lora-smoke-test"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

questions = [
    "What does LoRA stand for?",
    "How does LoRA differ from full fine-tuning?",
    "Why does QLoRA use 4-bit quantization?",
    "Explain how low-rank matrices reduce trainable parameters.",
]

def generate(question):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

results = []

for question in questions:
    with model.disable_adapter():
        baseline = generate(question)

    finetuned = generate(question)

    results.append({
        "question": question,
        "baseline": baseline,
        "lora": finetuned
    })

    print("\nQUESTION:", question)
    print("\nBASE MODEL:", baseline)
    print("\nLORA MODEL:", finetuned)
    print("\n" + "=" * 60)

with open("results/lora_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nComparison saved to results/lora_comparison.json")
