import json
from pathlib import Path
from collections import Counter

# domain, concept, question_type, difficulty,
# question, reference_answer
examples = [
    # TRANSFORMERS
    (
        "transformers", "attention_scaling", "mathematics", "advanced",
        "Why is scaled dot-product attention divided by sqrt(d_k)?",
        "Scaled dot-product attention computes softmax(QK^T / sqrt(d_k)). "
        "When query and key components have approximately unit variance, "
        "the variance of their dot product grows with d_k. Dividing by "
        "sqrt(d_k) controls the scale of attention logits and helps prevent "
        "softmax saturation."
    ),
    (
        "transformers", "residual_connections", "architecture", "intermediate",
        "Why do transformers use residual connections?",
        "Residual connections add a sublayer's output to its input. "
        "They improve gradient flow, help preserve information, and make "
        "deep transformer networks easier to optimize."
    ),
    (
        "transformers", "attention_complexity", "mathematics", "advanced",
        "How does standard self-attention complexity scale with sequence length?",
        "For sequence length n and hidden dimension d, standard full "
        "self-attention requires O(n^2 d) arithmetic for attention score "
        "and value mixing operations. Materializing the attention matrix "
        "requires O(n^2) memory per head, although memory-efficient "
        "implementations can avoid storing the full matrix."
    ),
    (
        "transformers", "encoder_decoder", "comparison", "intermediate",
        "Compare encoder-only and decoder-only transformers.",
        "Encoder-only transformers typically use bidirectional attention "
        "and are suitable for representation learning and understanding "
        "tasks. Decoder-only transformers use causal attention and are "
        "commonly trained for autoregressive next-token prediction."
    ),
    (
        "transformers", "attention_debugging", "troubleshooting", "advanced",
        "A causal language model attends to future tokens during training. What should you check?",
        "Check the causal attention mask, attention implementation, "
        "position indexing, and any custom attention overrides. Ensure "
        "future positions are masked and verify the model's output "
        "at an earlier position cannot depend on later input tokens."
    ),

    # TOKENIZATION
    (
        "tokenization", "embedding_matrix", "mathematics", "intermediate",
        "How many parameters are in an embedding table with 32000 tokens and width 768?",
        "The embedding table has vocabulary_size multiplied by "
        "embedding_width parameters: 32000 x 768 = 24,576,000 parameters."
    ),
    (
        "tokenization", "bpe", "explanation", "intermediate",
        "Explain byte-pair encoding in tokenization.",
        "Byte-pair encoding starts from an initial collection of symbols "
        "and iteratively merges frequent adjacent symbol pairs. The "
        "resulting merge rules provide a subword vocabulary that can "
        "represent common patterns efficiently."
    ),
    (
        "tokenization", "context_window", "architecture", "intermediate",
        "How is tokenization related to the model context window?",
        "The context window limits the number of tokens the model can "
        "process in a context. A tokenizer determines how text maps to "
        "tokens, so the same text may consume different context lengths "
        "under different tokenizers."
    ),
    (
        "tokenization", "tokenizer_mismatch", "troubleshooting", "advanced",
        "Why can using the wrong tokenizer produce poor model outputs?",
        "A mismatched tokenizer can assign token IDs that represent "
        "different text units from those used during model training. "
        "This breaks the expected correspondence between text, token "
        "IDs, and learned embeddings."
    ),
    (
        "tokenization", "embedding_vs_token", "comparison", "basic",
        "What is the difference between a token ID and a token embedding?",
        "A token ID is a discrete integer index into a vocabulary. "
        "A token embedding is a continuous learned vector associated "
        "with that token ID."
    ),

    # LORA
    (
        "lora", "parameter_count", "mathematics", "advanced",
        "Calculate LoRA adapter parameters for a 4096 by 4096 weight matrix with rank 8.",
        "For dimensions d=k=4096 and rank r=8, a LoRA update has "
        "r(d+k) = 8(4096+4096) = 65,536 trainable parameters. "
        "A full weight matrix contains 16,777,216 parameters. "
        "The adapter therefore has approximately 99.61 percent fewer "
        "trainable parameters than updating the full matrix."
    ),
    (
        "lora", "lora_scaling", "mathematics", "intermediate",
        "What is the purpose of lora_alpha?",
        "In conventional LoRA, the adapter update is scaled by "
        "alpha/r, where alpha is lora_alpha and r is the adapter rank. "
        "The scaling controls the contribution of the adapter update. "
        "Some variants use a different scaling rule."
    ),
    (
        "lora", "adapter_insertion", "architecture", "advanced",
        "How are LoRA adapters integrated into a transformer linear layer?",
        "For a frozen weight matrix W, LoRA introduces two trainable "
        "low-rank matrices A and B. The effective transformation "
        "becomes y = Wx + (alpha/r)BAx, using compatible matrix dimensions. "
        "The base weights remain frozen."
    ),
    (
        "lora", "lora_vs_qlora", "comparison", "intermediate",
        "Compare LoRA and QLoRA memory usage.",
        "LoRA trains low-rank adapters while storing the frozen base "
        "model at its chosen precision. QLoRA additionally stores "
        "the frozen base weights in a low-bit quantized representation, "
        "commonly 4-bit NF4. Both still require memory for activations, "
        "adapter gradients, optimizer states, and temporary computations."
    ),
    (
        "lora", "adapter_not_learning", "troubleshooting", "advanced",
        "LoRA training loss is not improving. What should you inspect?",
        "Verify that adapter parameters have requires_grad enabled, "
        "target modules exist, optimizer settings are appropriate, "
        "training examples are correctly tokenized, and labels are "
        "not entirely masked. Inspect gradients, learning rate, "
        "loss values, and data quality."
    ),

    # OPTIMIZATION
    (
        "optimization", "effective_batch", "mathematics", "intermediate",
        "Calculate effective batch size for batch size 2, accumulation 8, and 1 GPU.",
        "Effective batch size is per_device_batch_size multiplied by "
        "gradient_accumulation_steps multiplied by the number of "
        "training devices: 2 x 8 x 1 = 16 examples per optimizer step."
    ),
    (
        "optimization", "adamw", "explanation", "intermediate",
        "How does AdamW differ from ordinary SGD?",
        "SGD updates parameters using gradients, optionally with "
        "momentum. AdamW adapts updates using moving estimates of "
        "gradient first and second moments and applies decoupled "
        "weight decay."
    ),
    (
        "optimization", "mixed_precision", "architecture", "intermediate",
        "Why use BF16 mixed precision during language model training?",
        "BF16 reduces storage and computation costs relative to FP32 "
        "for eligible operations while retaining FP32's exponent range. "
        "Some operations or optimizer states may still use higher "
        "precision for numerical stability."
    ),
    (
        "optimization", "gradient_explosion", "troubleshooting", "advanced",
        "Training loss suddenly becomes NaN. What should you check?",
        "Inspect learning rate, gradient norms, numerical precision, "
        "input data, labels, and loss calculations. Gradient clipping, "
        "a lower learning rate, and more stable precision settings "
        "may help, depending on the underlying cause."
    ),
    (
        "optimization", "epoch_vs_step", "comparison", "basic",
        "What is the difference between an epoch and an optimizer step?",
        "An epoch represents one pass through the training dataset. "
        "An optimizer step updates trainable parameters after gradients "
        "have been computed and, if configured, accumulated."
    ),

    # INFERENCE
    (
        "inference", "kv_cache", "architecture", "advanced",
        "Why does autoregressive inference use a KV cache?",
        "A key-value cache stores attention keys and values from "
        "previously processed tokens so they do not need to be "
        "recomputed for every new generated token. This reduces "
        "redundant computation during autoregressive decoding."
    ),
    (
        "inference", "weight_memory", "mathematics", "intermediate",
        "Estimate raw BF16 weight memory for a model with 7 billion parameters.",
        "BF16 uses 2 bytes per parameter. Seven billion parameters "
        "therefore require approximately 14 billion bytes, or "
        "13.04 GiB, for raw model weights alone. Runtime memory "
        "requirements are higher because of activations and other buffers."
    ),
    (
        "inference", "quantization_tradeoff", "comparison", "intermediate",
        "Why might 4-bit inference be slower than BF16 inference on some GPUs?",
        "Low-bit inference can require dequantization and may use "
        "kernels that are less efficient for a particular model, "
        "batch size, or GPU. Lower memory consumption does not "
        "guarantee greater computation throughput."
    ),
    (
        "inference", "latency_debugging", "troubleshooting", "advanced",
        "An LLM has poor tokens-per-second performance. What should you measure?",
        "Measure prompt length, generated token count, prefill time, "
        "decode time, GPU utilization, memory bandwidth, batch size, "
        "kernel efficiency, synchronization, and quantization overhead. "
        "Separate time-to-first-token from steady-state decoding throughput."
    ),
    (
        "inference", "temperature", "explanation", "intermediate",
        "What does sampling temperature change during text generation?",
        "Temperature scales logits before sampling. Lower positive "
        "temperatures generally make the probability distribution "
        "more concentrated, while higher temperatures make it "
        "flatter. Greedy decoding does not require temperature."
    ),

    # EVALUATION
    (
        "evaluation", "data_leakage", "troubleshooting", "advanced",
        "Validation accuracy is unexpectedly perfect. What should you investigate?",
        "Check for exact and near-duplicate examples, shared answer "
        "templates, contamination from training data, target leakage, "
        "incorrect evaluation code, and overly simple test questions."
    ),
    (
        "evaluation", "perplexity_calculation", "mathematics", "intermediate",
        "What is perplexity when average token negative log-likelihood is 2?",
        "Perplexity is exp(average negative log-likelihood). "
        "For a value of 2 using natural logarithms, perplexity "
        "is exp(2), approximately 7.39."
    ),
    (
        "evaluation", "quality_vs_loss", "comparison", "intermediate",
        "Why can lower validation loss fail to improve answer quality?",
        "Validation loss measures predictive likelihood on the "
        "chosen reference tokens. Human-perceived answer quality "
        "also depends on factuality, relevance, instruction following, "
        "clarity, and the evaluation distribution."
    ),
    (
        "evaluation", "evaluation_pipeline", "architecture", "advanced",
        "Design a basic evaluation pipeline for fine-tuned language models.",
        "Prepare a held-out test set with reference answers and scoring "
        "criteria. Run each model under controlled decoding settings, "
        "record outputs and performance metrics, score factual accuracy "
        "and task success, analyze failures, and preserve configurations "
        "and results for reproducibility."
    ),
    (
        "evaluation", "benchmark_variance", "explanation", "intermediate",
        "Why repeat model benchmarks rather than relying on one run?",
        "Benchmark results vary due to initialization, data ordering, "
        "sampling, hardware conditions, and measurement noise. "
        "Repeated runs help estimate variability and distinguish "
        "consistent effects from random fluctuations."
    ),
]

assert len(examples) == 30

bank_path = Path("data/reference/reference_bank.jsonl")
existing = [
    json.loads(line)
    for line in bank_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

existing_ids = {r["id"] for r in existing}
existing_questions = {
    " ".join(r["question"].lower().split())
    for r in existing
}

new_records = []

for i, (domain, concept, qtype, difficulty, question, answer) in enumerate(examples, 31):
    record_id = f"AI-P03-{i:04d}"
    normalized = " ".join(question.lower().split())

    assert record_id not in existing_ids
    assert normalized not in existing_questions

    new_records.append({
        "id": record_id,
        "domain": domain,
        "concept": concept,
        "question_type": qtype,
        "difficulty": difficulty,
        "question": question,
        "reference_answer": answer,
        "review_status": "pending",
        "split": "train",
    })

output = Path("data/reference/expansion/batch_01.jsonl")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", encoding="utf-8") as f:
    for record in new_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("=== AI-P03 EXPANSION BATCH 01 ===")
print("New records:", len(new_records))
print("Domains:", dict(sorted(Counter(r["domain"] for r in new_records).items())))
print("Question types:", dict(sorted(Counter(r["question_type"] for r in new_records).items())))
print("Output:", output)
print("Candidate reference answers: pending review")
print("Status: PASS")
