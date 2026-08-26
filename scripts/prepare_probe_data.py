from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"xstest": "xstest.csv", "jbb": "jbb.csv"}
LABEL_MAP = {"safe": 0, "unsafe": 1}


def load_activation(path: Path) -> torch.Tensor:
    """Load one activation file and return [num_layers, hidden_size]."""

    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value["activations"]

    activation = value.detach().cpu().float().squeeze()

    if activation.ndim == 1:
        activation = activation.unsqueeze(0)
    elif activation.ndim == 2:
        pass
    elif activation.ndim == 3:
        activation = activation[:, -1, :]
    elif activation.ndim == 4:
        activation = activation[0, :, -1, :]
    else:
        raise ValueError(f"Unexpected activation shape in {path}: {tuple(activation.shape)}")

    return activation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine per-prompt activations into a probe dataset."
    )
    parser.add_argument("--model", required=True, help="Folder name under outputs/<dataset>/")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    args = parser.parse_args()

    labels_path = ROOT / "data" / DATASETS[args.dataset]
    activation_dir = ROOT / "outputs" / args.dataset / args.model / "activations"
    output_dir = ROOT / "analysis" / "prepared"
    output_file = output_dir / f"{args.dataset}_{args.model}.pt"

    with labels_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{labels_path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    ids: list[str] = []
    labels: list[int] = []
    activations: list[torch.Tensor] = []
    activation_shape: tuple[int, ...] | None = None

    for row in rows:
        prompt_id = str(row["id"])
        text_label = str(row["label"]).strip().lower()
        if text_label not in LABEL_MAP:
            raise ValueError(f"Unknown label for {prompt_id}: {row['label']!r}")

        activation_path = activation_dir / f"{prompt_id}.pt"
        if not activation_path.is_file():
            raise FileNotFoundError(f"Missing activation file: {activation_path}")

        activation = load_activation(activation_path)
        if activation_shape is None:
            activation_shape = tuple(activation.shape)
        elif tuple(activation.shape) != activation_shape:
            raise ValueError(
                f"Activation shape mismatch for {prompt_id}: "
                f"expected {activation_shape}, got {tuple(activation.shape)}"
            )

        ids.append(prompt_id)
        labels.append(LABEL_MAP[text_label])
        activations.append(activation)

    if not activations:
        raise RuntimeError("No activation files were loaded.")

    X = torch.stack(activations, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    probe_dataset = {
        "ids": ids,
        "X": X,
        "y": y,
        "label_map": LABEL_MAP,
        "num_examples": len(ids),
        "activation_shape": activation_shape,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe_dataset, output_file)

    print(f"Saved: {output_file}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"y shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
