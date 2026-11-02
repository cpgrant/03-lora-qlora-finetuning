# AI-P03 — LoRA and QLoRA Fine-Tuning

Fine-tuning Qwen3-0.6B using LoRA and QLoRA on an NVIDIA RTX 4090.

## Environment

- Ubuntu / WSL2
- Python 3.11.9
- PyTorch 2.11.0 + CUDA 12.8
- NVIDIA RTX 4090 (24 GB)
- Transformers, PEFT, TRL, bitsandbytes

## Experiments

1. Baseline model inference
2. LoRA fine-tuning (BF16)
3. QLoRA fine-tuning (4-bit NF4)
4. Small-scale smoke tests
5. Expanded 200-example training experiment
6. Four-model inference comparison

## Training Results

| Metric | LoRA v2 | QLoRA v2 |
|---|---:|---:|
| Training time | 62.72 s | 130.86 s |
| Peak allocated GPU memory | 1.363 GiB | 0.964 GiB |
| Training loss | 1.0891 | 1.1900 |
| Validation loss | 3.4102 | 3.2689 |

## Dataset

- 200 training examples
- 40 validation examples
- Six AI engineering domains

The dataset is derived from 60 underlying concepts and contains
multiple formulations. Reference answers have not all been
independently reviewed.

## Key Findings

QLoRA reduced peak allocated GPU memory by approximately 29.3%
compared with LoRA but required approximately 2.09 times as long
to train in this experiment.

The limited evaluation did not demonstrate reliable generalization
on technical mathematics questions.

## Reproducibility

Install dependencies:

    python -m pip install -r requirements.txt

Run experiments from the project root:

    python scripts/train_lora_v2.py
    python scripts/train_qlora_v2.py
    python scripts/compare_final.py

The original model is downloaded from Hugging Face.
Fine-tuned adapter weights are not included in this repository.

## Status

Engineering implementation complete.
Independent model-quality validation remains future work.
