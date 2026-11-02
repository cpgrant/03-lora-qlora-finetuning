import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = "Qwen/Qwen3-0.6B"
adapter_path = "outputs/lora-smoke-test"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="cuda"
)

model = PeftModel.from_pretrained(base, adapter_path)
model.eval()

questions = [
    "What does LoRA stand for?",
    "How does LoRA differ from full fine-tuning?",
    "Why does QLoRA use 4-bit quantization?"
]

for question in questions:
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print("\nQUESTION:", question)
    print("ANSWER:", answer)
