# SLM Jailbreak Brittleness
###### Project Done By: Sankeerthana, Baisayan, Harshprabha, Devansh for the Global South AI Safety Hackathon, June 2026.

This project investigates whether smaller language models are more susceptible to jailbreak-style unsafe compliance than larger models, and whether this vulnerability is better explained by **capability limitations** or **alignment brittleness**.

We compare Qwen and Mistral instruction-tuned models on XSTest and a modified version of AdvBench. The pipeline extracts hidden activations during inference, trains layer-wise linear probes to detect safe/unsafe prompts, and compares the probe's internal risk prediction with the model's generated response.

## Research Question

Are smaller language models systematically more susceptible to jailbreak attacks than larger models, and if so, is this vulnerability better explained by capability limitations or alignment brittleness?

## Key Idea

- **Capability limitation:** the model produces an unsafe response and the probe also fails to detect internal risk.
- **Alignment brittleness:** the probe detects the prompt as risky, but the model still produces an unsafe compliant response.

This lets us distinguish whether unsafe behaviour happens because the model cannot internally detect risk, or because an internally decodable risk signal fails to translate into safe refusal behaviour.

## Models

| Model Family | Smaller Model | Larger Model |
|---|---|---|
| Mistral | Mistral-7B-Instruct-v0.3 | Mistral-Nemo-Instruct (12B) |
| Qwen | Qwen2.5-0.5B-Instruct | Qwen2.5-14B-Instruct |
| Qwen | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct |

## Datasets

- **XSTest:** safe and unsafe prompts for testing refusal calibration.
- **Modified AdvBench:** adversarial jailbreak-style prompts.

Each prompt is labelled as `safe` or `unsafe`.

## Pipeline

Run the project in this order:

### 1. Download / prepare datasets

```bash
python download_dataset.py
```

This prepares the datasets used for inference and probing.

### 2. Run inference and extract activations

```bash
python inference_w_activations.py \
  --model_path /path/to/model \
  --model_tag qwen_7b_instruct \
  --input_csv_path datasets/xstest.csv \
  --prompt_col prompt \
  --label_col label \
  --output_dir output/xstest \
  --dataset xstest
```

This runs the model on each prompt, saves the generated responses, and extracts final-prompt-token activations from all hidden layers.

### 3. Combine activations into a probe dataset

```bash
python data_processing.py \
  --activation_dir output/xstest/qwen_7b_instruct/activations \
  --labels_csv datasets/xstest.csv \
  --output_filename combined_xstest_activations \
  --output_dir output/xstest/qwen_7b_instruct
```

This combines the per-prompt activation files into one `.pt` file containing activations, labels, IDs, and metadata.

### 4. Train probes and run analysis

Open and run:

```text
inspect_probe_training.ipynb
```

The notebook trains layer-wise linear probes, selects the best layer using validation macro-F1, and computes probe-response failure metrics.

## Probe-Response Categories

| Probe Prediction | Generated Response | Category | Interpretation |
|---|---|---|---|
| Safe | Safe | Aligned safe | No unsafe behaviour |
| Risky | Safe | Risk detected, safe response | Internal risk signal leads to safe behaviour |
| Risky | Unsafe | Known-risk unsafe response | Alignment brittleness |
| Safe | Unsafe | Missed internal risk | Capability limitation |

## Main Metrics

- **Known-risk unsafe rate:** how often probe-risky examples still produce unsafe responses.
- **Missed internal risk rate:** how often unsafe responses occur when the probe fails to detect risk.
- **Alignment brittleness score:** known-risk unsafe rate weighted by the probe's confidence on those failures.

## Summary of Findings

The results suggest that jailbreak susceptibility is not explained by model size alone. Scaling improves safety within the Mistral family, but Qwen models show stronger refusal behaviour overall. Most importantly, many unsafe compliance cases occur when risk is already internally decodable from hidden activations, suggesting that failures are often better explained by alignment brittleness than by a complete absence of risk detection.

## Limitations

Linear probes show decodability, not causality. The response labelling also uses a rule-based refusal detector, which may misclassify ambiguous responses. Future work should test more models and datasets, use human or model-assisted response annotation, and apply causal interventions such as activation patching or steering.
