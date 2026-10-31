# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

icefall holds speech recipes (ASR, TTS, KWS and others) built on [k2](https://github.com/k2-fsa/k2), [lhotse](https://github.com/lhotse-speech/lhotse) and PyTorch. This checkout is a fork (`sonchuate/icefall`) of `k2-fsa/icefall`. Its own work is a Vietnamese ASR recipe in `egs/custom/ASR`, trained with `pruned_transducer_stateless7` (see "The `custom` recipe" below).

`k2` and `lhotse` are not in `requirements.txt` and must be installed separately. Add the repo root to `PYTHONPATH` so `import icefall` works.

## Commands

Style checks use the versions pinned in `.pre-commit-config.yaml` (`black==22.3.0`, `flake8==5.0.4`, `isort` with the black profile). CI (`.github/workflows/style_check.yml`) runs them from the repo root:

```bash
pre-commit install                                      # enable hooks; they are off by default
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics   # the only flake8 check that fails CI
black --check --diff .
isort --check --diff .
```

Tests use pytest, and recipe tests must run from inside the recipe directory because they use bare imports:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
pytest -v -s ./test                                     # library tests
pytest -v -s test/test_utils.py::test_name              # single test

cd egs/librispeech/ASR/zipformer && pytest -v -s        # one recipe's tests
```

`.github/workflows/test.yml` lists the recipe directories CI covers. Its jobs only run when the repo owner is `k2-fsa` or `csukuangfj`, so they do not run on this fork. `style_check.yml` has no such guard.

Recipes run from `egs/<dataset>/ASR`, not the repo root, because scripts use paths relative to it (`data/`, `shared/`, `<model_dir>/exp`):

```bash
cd egs/yesno/ASR
./prepare.sh                                            # data preparation
./prepare.sh --stage 3 --stop-stage 5                   # selected stages (via shared/parse_options.sh)
python3 ./tdnn/train.py
python3 ./tdnn/decode.py
```

`egs/yesno/ASR` is the smallest end-to-end recipe, and `.github/scripts/<dataset>/ASR/run.sh` shows the exact train, decode and export invocations used in CI.

Docs are Sphinx sources in `docs/source`. `docs/source/contributing/how-to-create-a-recipe.rst` describes the recipe layout.

## Architecture

**`icefall/` is the shared library** and **`egs/<dataset>/<task>/` holds recipes**. Most model code lives in recipes, not the library. `icefall/` provides checkpointing, DDP setup, graph compilers, lexicon handling, k2 decoding, and utilities such as `AttributeDict`. `icefall/shared/` holds scripts that recipes reach through their `shared` symlink.

**Recipe layout.** Each recipe has `prepare.sh`, `local/` (data preparation) and one directory per model variant, such as `zipformer/` or `pruned_transducer_stateless7/`. `egs/librispeech/ASR` is the reference recipe with the most models, and its README tables explain how the variants differ. `zipformer/` is the latest one.

**Model directories are scripts, not packages.** Files sit side by side and import each other by bare name (`from decoder import Decoder`, `import optim`), relying on the script's directory being on `sys.path`. A typical directory has `train.py`, `decode.py`, `export*.py`, `pretrained.py`, `model.py`, `decoder.py`, `joiner.py` and the encoder. Each `train.py` is self-contained: `get_parser()` holds CLI options, `get_params()` returns an `AttributeDict` of non-CLI settings, and the two are merged before training.

**Code is shared between model variants by symlink.** A variant usually links files from an earlier one, for example `pruned_transducer_stateless7/beam_search.py` to `../pruned_transducer_stateless2/beam_search.py`, and datamodules commonly chain back to `tdnn_lstm_ctc/asr_datamodule.py`. Editing a linked file therefore changes every variant that uses it, and a variant that needs different behaviour must replace the link with a real copy.

**Data flow.** Lhotse manifests (recordings + supervisions) become a `CutSet`, then fbank features are computed and stored. `asr_datamodule.py` turns cuts into PyTorch dataloaders using `DynamicBucketingSampler` and `K2SpeechRecognitionDataset`. Batch size is set in seconds of audio through `--max-duration`, not in utterances. Language assets live in `data/lang_bpe_<vocab>/` or `data/lang_phone/`.

## Symlinks on this checkout

The repo tracks about 1,400 symlinks (git mode `120000`), and this Windows checkout has `core.symlinks=false`. Each one appears as a short text file containing its relative target path, such as `../pruned_transducer_stateless2/beam_search.py`.

- Importing such a file raises `SyntaxError`, so anything depending on a link will not run here until the links are real.
- Before reading or editing a very small `.py` file, check whether it is a stub, and follow its target to find the real code.
- Writing code into a stub changes the link target in git. To fork a linked file, replace it with a real copy and commit the mode change.

`egs/custom/ASR` is worse: it was copied from librispeech with the stubs committed as regular files (mode `100644`), so about 290 of them are broken on every OS, including Linux. That includes `egs/custom/ASR/shared` and `pruned_transducer_stateless7/beam_search.py`.

## The `custom` recipe

`egs/custom/ASR` is a copy of `egs/librispeech/ASR`, and most of it is unused and still librispeech-specific. Its `prepare.sh` calls `local/` scripts that do not exist here. Only three preparation scripts are present, run in this order:

```bash
python local/prepare_manifest.py --csv train.csv --output-dir data/manifests/train   # CSV columns: audio_path, text
python local/make_cutsets.py --recordings data/manifests/train/recordings.jsonl.gz \
  --supervisions data/manifests/train/supervisions.jsonl.gz --output data/cuts_train.jsonl.gz
python local/compute_fbank.py --manifest data/cuts_train.jsonl.gz --storage-dir data/fbank/train   # rewrites the manifest in place
```

Training reads `<manifest-dir>/cuts_train.jsonl.gz` and `cuts_eval.jsonl.gz`, and decoding reads `cuts_test.jsonl.gz`. `--manifest-dir` defaults to `data/` in `train.py`. Nothing in `local/` builds the BPE model that `--bpe-model` expects at `data/lang_bpe_500/bpe.model`.

`pruned_transducer_stateless7` has diverged across two recipes, and neither is complete:

- `train.py` is identical in `egs/custom` and `egs/librispeech`. It defines `--manifest-dir` itself, which replaces the datamodule's option.
- `egs/custom/.../asr_datamodule.py` and `encoder_interface.py` are real copies, with `--manifest-dir` removed to avoid an argparse conflict. The librispeech `train.py` still conflicts with its linked datamodule, so it fails at startup. Train from `egs/custom/ASR`.
- `train.py` declares `--manifest-dir` as `str`, but the datamodule's MUSAN branch does `self.args.manifest_dir / "musan_cuts.jsonl.gz"`. With the default `--enable-musan 1` this raises `TypeError`, so pass `--enable-musan 0`.
- The decoding changes that read `cuts_test.jsonl.gz` are only in `egs/librispeech/.../decode.py`. `egs/custom/.../decode.py` is still the upstream version, which calls the LibriSpeech test sets and imports the broken `beam_search.py` stub.
