from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"xstest": "xstest.csv", "jbb": "jbb.csv"}
MAX_NEW_TOKENS = 256
MODEL_DTYPE = torch.bfloat16


def load_rows(dataset: str) -> list[dict[str, str]]:
    """Load only the three fields used by this experiment."""

    path = ROOT / "data" / DATASETS[dataset]
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "prompt", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return [
            {"id": row["id"], "prompt": row["prompt"], "label": row["label"]}
            for row in reader
        ]


@torch.inference_mode()
def get_prompt_activations(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return [hidden_states, hidden_size] at the last prompt token."""

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    final_token = int(attention_mask[0].sum().item()) - 1
    return torch.stack(
        [state[0, final_token, :].detach().cpu() for state in outputs.hidden_states]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference and extract prompt activations."
    )
    parser.add_argument("--model", required=True, help="Folder name under models/")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    model_path = ROOT / "models" / args.model
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    rows = load_rows(args.dataset)
    output_dir = ROOT / "outputs" / args.dataset / args.model
    activation_dir = output_dir / "activations"
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.chat_template is None:
        raise RuntimeError("The local tokenizer does not have a chat template.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        dtype=MODEL_DTYPE,
    )
    model.to(device)
    model.eval()
    generation_config = model.generation_config
    generation_eos_token_id = tokenizer.eos_token_id
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.repetition_penalty = 1.0
        if generation_config.eos_token_id is not None:
            generation_eos_token_id = generation_config.eos_token_id

    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset} ({len(rows)} rows)")
    print(f"Output: {output_dir}")

    with responses_path.open("w", encoding="utf-8", newline="\n") as responses:
        for number, row in enumerate(rows, start=1):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            attention_mask = torch.ones_like(input_ids)

            activations = get_prompt_activations(model, input_ids, attention_mask)

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=generation_eos_token_id,
            )
            response_ids = generated_ids[0, input_ids.shape[1] :].cpu()
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            activation_path = activation_dir / f"{row['id']}.pt"
            torch.save(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "activations": activations,
                },
                activation_path,
            )
            responses.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "label": row["label"],
                        "response": response,
                        "activation_path": str(activation_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            responses.flush()
            print(f"{number}/{len(rows)}: {row['id']}")

    print("Done")


if __name__ == "__main__":
    main()
