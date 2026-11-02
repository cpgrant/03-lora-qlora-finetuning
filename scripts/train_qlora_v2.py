import json
import time
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/qlora-v2"

print("=== AI-P03: QLoRA TRAINING ===")
print("GPU:", torch.cuda.get_device_name(0))

# 1. Dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/benchmark-v2/train.jsonl",
        "validation": "data/benchmark-v2/eval.jsonl",
    },
)

# 2. Configure 4-bit NF4 quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 3. Load quantized base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    quantization_config=quant_config,
    device_map={"": 0},
)

print("4-bit model loaded")
print("Quantized:", model.is_loaded_in_4bit)

# 4. Prepare frozen quantized weights
model = prepare_model_for_kbit_training(model)

# 5. Same LoRA configuration as Experiment A
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# 6. Same training configuration
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_length=256,
    completion_only_loss=True,
    bf16=True,
    gradient_checkpointing=False,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    seed=42,
)

# 7. Trainer
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()

# 8. Training
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

result = trainer.train()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
peak_memory = torch.cuda.max_memory_allocated() / 1024**3

# 9. Evaluation
metrics = trainer.evaluate()

# 10. Save adapter
trainer.save_model(OUTPUT_DIR)

summary = {
    "model": MODEL_ID,
    "method": "QLoRA",
    "quantization": "4-bit NF4",
    "training_seconds": round(elapsed, 2),
    "peak_gpu_memory_gib": round(peak_memory, 3),
    "training_loss": result.training_loss,
    "evaluation_loss": metrics.get("eval_loss"),
    "train_examples": len(dataset["train"]),
    "eval_examples": len(dataset["validation"]),
}

with open("results/qlora_v2.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== QLORA TRAINING SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nAdapter saved to:", OUTPUT_DIR)
