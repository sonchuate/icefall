#!/usr/bin/env bash
# Export a fine-tuned checkpoint to ONNX, in a folder laid out like the gipformer one.
# Run from egs/custom/ASR with the training environment activated.
#
# Usage:
#   bash export_zac.sh <checkpoint> <name>
#
# Examples:
#   bash export_zac.sh ../../../data/gipformer/best-valid-loss.pt finetune_best
#   bash export_zac.sh zipformer/exp_lora_merged/epoch-1.pt lora_best
#
# The result is written to $GIPFORMER_DIR/<name>. Works for both best-valid-loss.pt
# and the output of zipformer_lora/merge_lora.py.
#
# Options can be overridden by environment variables:
#   GIPFORMER_DIR  folder with tokens.txt, bpe.model, bpe.vocab and config.json
#   WORK_DIR       folder for intermediate files, including the .int8.onnx models
#   KEY            weights to export: model or model_avg

set -eou pipefail

if [ $# -ne 2 ]; then
  echo "Usage: bash $0 <checkpoint> <name>" >&2
  exit 1
fi

CHECKPOINT=$1
NAME=$2
GIPFORMER_DIR=${GIPFORMER_DIR:-../../../data/gipformer}
WORK_DIR=${WORK_DIR:-$HOME/gip_export}
KEY=${KEY:-model}

OUTPUT_DIR=$GIPFORMER_DIR/$NAME
EXP_DIR=$WORK_DIR/$NAME
mkdir -p "$EXP_DIR" "$OUTPUT_DIR"

CHECKPOINT=$CHECKPOINT KEY=$KEY EXP_DIR=$EXP_DIR python - <<'E'
import os
import pathlib
import sys

import torch

# A checkpoint saved by Python 3.13 cannot be opened by older versions without this.
if sys.version_info < (3, 13):
    sys.modules["pathlib._local"] = pathlib

checkpoint = torch.load(os.environ["CHECKPOINT"], map_location="cpu", weights_only=False)
state_dict = {
    k.removeprefix("module."): v.float() if v.dtype.is_floating_point else v
    for k, v in checkpoint[os.environ["KEY"]].items()
}
assert not any("lora" in k for k in state_dict), (
    "This checkpoint still has LoRA adapters. Run zipformer_lora/merge_lora.py first."
)
torch.save({"model": state_dict}, os.environ["EXP_DIR"] + "/epoch-1.pt")
print(f"Saved {len(state_dict)} tensors")
E

python zipformer/export-onnx.py \
  --tokens "$GIPFORMER_DIR/tokens.txt" \
  --use-averaged-model 0 \
  --epoch 1 \
  --avg 1 \
  --exp-dir "$EXP_DIR" \
  --use-ctc 1 \
  --use-transducer 1

for f in encoder decoder joiner; do
  cp "$EXP_DIR/$f-epoch-1-avg-1.onnx" "$OUTPUT_DIR/$f.onnx"
done
cp "$EXP_DIR/epoch-1.pt" "$OUTPUT_DIR/model.pt"
cp "$GIPFORMER_DIR/bpe.model" "$GIPFORMER_DIR/bpe.vocab" \
  "$GIPFORMER_DIR/tokens.txt" "$GIPFORMER_DIR/config.json" "$OUTPUT_DIR/"

echo "Exported to $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
