#!/usr/bin/env bash
# Fine-tune gipformer on metadata.csv. Run from egs/custom/ASR in WSL.
#
# Options can be overridden by environment variables, for example:
#   MAX_DURATION=40 NUM_EPOCHS=5 ./finetune_zac.sh

set -eou pipefail

METADATA_CSV=${METADATA_CSV:-../../../data/metadata.csv}
GIPFORMER_DIR=${GIPFORMER_DIR:-../../../data/gipformer}
EXP_DIR=${EXP_DIR:-zipformer/exp_zac}
WORLD_SIZE=${WORLD_SIZE:-1}
MAX_DURATION=${MAX_DURATION:-60}
NUM_EPOCHS=${NUM_EPOCHS:-10}
BASE_LR=${BASE_LR:-0.001}

python zipformer/finetune.py \
  --world-size "$WORLD_SIZE" \
  --num-epochs "$NUM_EPOCHS" \
  --start-epoch 1 \
  --exp-dir "$EXP_DIR" \
  --do-finetune 1 \
  --finetune-ckpt "$GIPFORMER_DIR/model.pt" \
  --bpe-model "$GIPFORMER_DIR/bpe.model" \
  --metadata-csv "$METADATA_CSV" \
  --use-transducer 1 \
  --use-ctc 1 \
  --warm-step 0 \
  --base-lr "$BASE_LR" \
  --lr-batches 100000 \
  --lr-epochs 100 \
  --max-duration "$MAX_DURATION" \
  --use-fp16 1 \
  --on-the-fly-feats 1 \
  --enable-musan 0 \
  --num-workers 2 \
  --save-every-n 100000 \
  --average-period 20
