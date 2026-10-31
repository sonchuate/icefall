#!/usr/bin/env python3
"""Merge LoRA adapters into the weights of the base model.

The output is a plain checkpoint with the same keys as the base model, so it works
with zipformer/decode.py and zipformer/export-onnx.py.

python zipformer_lora/merge_lora.py \
    --checkpoint zipformer_lora/exp_zac/epoch-20.pt \
    --base-model ../../../data/gipformer/model.pt \
    --output zipformer/exp_lora_merged/epoch-1.pt
"""

import argparse
from pathlib import Path

import torch


def merge(checkpoint: Path, base_model: Path, output: Path, key: str, lora_alpha: int):
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)[key]
    base = torch.load(base_model, map_location="cpu", weights_only=False)["model"]

    merged = {name: base[name].clone() for name in base}
    num_adapters = 0
    for name, lora_a in state_dict.items():
        if not name.endswith(".lora_A"):
            continue
        prefix = name[: -len(".lora_A")]
        lora_b = state_dict[f"{prefix}.lora_B"]
        delta = (lora_b @ lora_a).float() * (lora_alpha / lora_a.shape[0])

        # ActivationDropoutAndLinear_lora keeps the layer in a submodule named l.
        weight = f"{prefix}.weight"
        if weight not in merged:
            assert prefix.endswith(".l"), prefix
            weight = f"{prefix[:-2]}.weight"
        merged[weight] += delta
        num_adapters += 1

    assert num_adapters > 0, f"No LoRA adapters found in {checkpoint}"

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged}, output)
    print(f"Merged {num_adapters} adapters into {output}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into the base model."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint saved by zipformer_lora/finetune.py.",
    )
    parser.add_argument(
        "--base-model",
        type=Path,
        required=True,
        help="Checkpoint the fine-tuning started from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path of the merged checkpoint.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="model",
        choices=["model", "model_avg"],
        help="Which weights of the checkpoint to merge.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=4,
        help="Must match lora_alpha used in zipformer_lora/zipformer.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    merge(args.checkpoint, args.base_model, args.output, args.key, args.lora_alpha)
