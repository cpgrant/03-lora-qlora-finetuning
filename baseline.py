import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-0.6B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model...")
start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda"
)
model.eval()

print(f"Load time: {time.perf_counter()-start:.2f} seconds")
print(f"Model parameters: {model.num_parameters():,}")

prompt = "Explain the difference between LoRA and full fine-tuning in three sentences."

messages = [{"role": "user", "content": prompt}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True
).to("cuda")

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

generated = outputs[0][inputs["input_ids"].shape[-1]:]
response = tokenizer.decode(generated, skip_special_tokens=True)

print("\n=== BASELINE RESPONSE ===")
print(response)

print("\n=== PERFORMANCE ===")
print(f"Generated tokens: {len(generated)}")
print(f"Generation time: {elapsed:.2f} seconds")
print(f"Tokens/sec: {len(generated)/elapsed:.2f}")
print(f"Peak GPU memory (PyTorch allocated): {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
