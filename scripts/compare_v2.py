import json
import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODEL_ID = "Qwen/Qwen3-0.6B"

QUESTIONS = [
    "What does LoRA stand for?",
    "How does LoRA differ from full fine-tuning?",
    "Why does QLoRA use 4-bit quantization?",
    "Explain how low-rank matrices reduce trainable parameters.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def generate(model, question):
    messages = [{"role": "user", "content": question}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = output[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)

    return {
        "answer": answer,
        "tokens": len(tokens),
        "seconds": round(elapsed, 3),
        "tokens_per_second": round(len(tokens) / elapsed, 2),
    }

results = {question: {} for question in QUESTIONS}

# BASE + LoRA
print("\nLoading BF16 base model and LoRA adapter...")

base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda",
)

lora_model = PeftModel.from_pretrained(
    base, "outputs/lora-v2"
)
lora_model.eval()

for question in QUESTIONS:
    with lora_model.disable_adapter():
        results[question]["base"] = generate(lora_model, question)

    results[question]["lora"] = generate(lora_model, question)

del lora_model, base
torch.cuda.empty_cache()

# QLoRA
print("\nLoading 4-bit quantized model and QLoRA adapter...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

quant_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    quantization_config=quant_config,
    device_map={"": 0},
)

qlora_model = PeftModel.from_pretrained(
    quant_base, "outputs/qlora-v2"
)
qlora_model.eval()

for question in QUESTIONS:
    results[question]["qlora"] = generate(qlora_model, question)

# Save and print comparison
with open("results/compare_v2.json", "w") as f:
    json.dump(results, f, indent=2)

for question, variants in results.items():
    print("\n" + "=" * 70)
    print("QUESTION:", question)

    for name, result in variants.items():
        print(f"\n--- {name.upper()} ---")
        print(result["answer"])
        print(
            f"\nTokens: {result['tokens']} | "
            f"Time: {result['seconds']}s | "
            f"Speed: {result['tokens_per_second']} tok/s"
        )

print("\nResults saved: results/compare_v2.json")
