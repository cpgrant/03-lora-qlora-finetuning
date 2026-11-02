import time
import json
import torch

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "outputs/lora-smoke-test"

print("=== AI-P03: LoRA TRAINING ===")
print("GPU:", torch.cuda.get_device_name(0))

# 1. Load dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/train.jsonl",
        "validation": "data/eval.jsonl",
    },
)

print("Training examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))

# 2. Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# 3. Configure training
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    model_init_kwargs={"dtype": "bfloat16"},
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

# 4. Create trainer
trainer = SFTTrainer(
    model=MODEL_ID,
    args=training_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()

# 5. Train and measure
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

result = trainer.train()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
peak_memory = torch.cuda.max_memory_allocated() / 1024**3

# 6. Evaluate
metrics = trainer.evaluate()

# 7. Save LoRA adapter
trainer.save_model(OUTPUT_DIR)

summary = {
    "model": MODEL_ID,
    "method": "LoRA",
    "training_seconds": round(elapsed, 2),
    "peak_gpu_memory_gib": round(peak_memory, 3),
    "training_loss": result.training_loss,
    "evaluation_loss": metrics.get("eval_loss"),
    "train_examples": len(dataset["train"]),
    "eval_examples": len(dataset["validation"]),
}

with open("results/lora_smoke_test.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== TRAINING SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\nAdapter saved to:", OUTPUT_DIR)
