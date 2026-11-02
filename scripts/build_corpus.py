import json
import random
from pathlib import Path
from collections import Counter

# Each record contains:
# topic, concept, question, reference answer, split.
# Answers are initial authored references; they still need review.

records = [
    # TRANSFORMERS
    ("transformers", "architecture", "What is a transformer?",
     "A transformer is a neural network architecture that uses attention mechanisms to process relationships between tokens. It is widely used in language models.", "train"),
    ("transformers", "self_attention", "What is self-attention?",
     "Self-attention computes contextual representations by allowing each token to attend to permitted token positions in the same sequence.", "train"),
    ("transformers", "causal_attention", "What is causal attention?",
     "Causal attention uses a mask that prevents each position from attending to future tokens, supporting autoregressive generation.", "train"),
    ("transformers", "multi_head", "Why use multiple attention heads?",
     "Multi-head attention allows different learned projections to capture different relationships between tokens.", "train"),
    ("transformers", "positional_information", "Why do transformers need positional information?",
     "Attention alone does not inherently encode token order. Positional information enables a transformer to distinguish positions and order within a sequence.", "eval"),

    # TOKENIZATION
    ("tokenization", "tokens", "What is a token?",
     "A token is a unit represented by a tokenizer, such as a subword, word, character, or special symbol.", "train"),
    ("tokenization", "token_ids", "What are token IDs?",
     "Token IDs are integer identifiers assigned to tokens in a tokenizer vocabulary.", "train"),
    ("tokenization", "embeddings", "What are token embeddings?",
     "Token embeddings are learned vector representations that map discrete token IDs into continuous numerical spaces.", "train"),
    ("tokenization", "vocabulary", "What is tokenizer vocabulary size?",
     "Vocabulary size is the number of token identifiers supported by a tokenizer, including any configured special tokens.", "train"),
    ("tokenization", "unknown_words", "How can subword tokenization handle unfamiliar words?",
     "Subword tokenization can represent unfamiliar words using combinations of smaller known units, subject to the tokenizer's vocabulary and rules.", "eval"),

    # LORA / QLORA
    ("lora", "definition", "What does LoRA stand for?",
     "LoRA stands for Low-Rank Adaptation.", "train"),
    ("lora", "frozen_weights", "What happens to pretrained weights during LoRA training?",
     "Standard LoRA freezes pretrained base weights and trains additional low-rank adapter matrices.", "train"),
    ("lora", "matrix_factorization", "How does LoRA represent a weight update?",
     "LoRA represents a weight update as the product of two smaller trainable matrices. For a d-by-k matrix and rank r, the adapter requires r(d+k) parameters instead of dk.", "train"),
    ("lora", "quantization", "What is QLoRA?",
     "QLoRA combines LoRA adapters with a quantized frozen base model, commonly using 4-bit NF4 quantization.", "train"),
    ("lora", "memory_tradeoffs", "Why can QLoRA require less GPU memory than LoRA?",
     "QLoRA reduces storage memory for frozen base weights using quantization. Adapter weights, gradients, activations, and other training state still consume memory.", "eval"),

    # OPTIMIZATION
    ("optimization", "gradient", "What is a gradient?",
     "A gradient describes how a differentiable objective changes with respect to model parameters.", "train"),
    ("optimization", "backpropagation", "What is backpropagation?",
     "Backpropagation applies the chain rule to compute gradients through a computational graph.", "train"),
    ("optimization", "learning_rate", "What is a learning rate?",
     "The learning rate controls the scale of parameter updates produced by an optimizer.", "train"),
    ("optimization", "batch_size", "What is batch size?",
     "Batch size is the number of training examples processed together in a batch.", "train"),
    ("optimization", "gradient_accumulation", "Why use gradient accumulation?",
     "Gradient accumulation combines gradients from multiple microbatches before an optimizer update, enabling a larger effective batch size with limited memory.", "eval"),

    # QUANTIZATION / INFERENCE
    ("inference", "inference_definition", "What is model inference?",
     "Inference is the process of using a trained model to compute predictions or generate outputs from new inputs.", "train"),
    ("inference", "quantization_definition", "What is weight quantization?",
     "Weight quantization represents model weights using lower-precision numerical formats to reduce memory requirements and potentially improve execution efficiency.", "train"),
    ("inference", "bf16", "What is BF16?",
     "BF16 is a 16-bit floating-point format with an 8-bit exponent and 7 explicit fraction bits, providing a wide numerical range with reduced precision.", "train"),
    ("inference", "throughput", "What does tokens per second measure?",
     "Tokens per second measures token processing or generation throughput. Its interpretation depends on whether input processing, output generation, or both are included.", "train"),
    ("inference", "latency", "What is the difference between latency and throughput?",
     "Latency measures elapsed time for an operation or request, while throughput measures the amount of work completed per unit time.", "eval"),

    # EVALUATION
    ("evaluation", "training_loss", "What is training loss?",
     "Training loss measures prediction error on training data according to the chosen objective.", "train"),
    ("evaluation", "validation_loss", "What is validation loss?",
     "Validation loss measures prediction error on held-out validation examples and helps monitor generalization during model development.", "train"),
    ("evaluation", "overfitting", "What is overfitting?",
     "Overfitting occurs when a model learns training-specific patterns that do not generalize well to unseen data.", "train"),
    ("evaluation", "perplexity", "What is language model perplexity?",
     "Perplexity is the exponential of average negative log-likelihood per token, under a consistent tokenization and evaluation setup. Lower values indicate better likelihood prediction.", "train"),
    ("evaluation", "data_leakage", "Why must evaluation examples be separated from training data?",
     "Separating evaluation data reduces the risk of measuring memorization rather than generalization. Semantic duplication and contamination must also be considered.", "eval"),
]

assert len(records) == 30
assert len({r[2].lower() for r in records}) == 30

output = Path("data/v2")
output.mkdir(parents=True, exist_ok=True)

for split in ("train", "eval"):
    selected = [r for r in records if r[4] == split]
    random.Random(42).shuffle(selected)

    with (output / f"{split}.jsonl").open("w") as f:
        for topic, concept, question, answer, _ in selected:
            example = {
                "prompt": [{"role": "user", "content": question}],
                "completion": [{"role": "assistant", "content": answer}]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{split}: {len(selected)} examples")

print("\nDomain distribution:")
for topic, count in sorted(Counter(r[0] for r in records).items()):
    print(f"  {topic}: {count}")

print("\nCorpus creation: PASS")
print("Reference answers require technical review before benchmarking.")
