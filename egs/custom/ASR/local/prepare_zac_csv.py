#!/usr/bin/env python3
"""Build metadata.csv (columns: text, audio_path) from the ZAC train split.

Both the final request and the user turns in `history` are used. Text is normalized
to match the gipformer tokenizer, which is upper case and has no digits or
punctuation. audio_path is relative to the directory of metadata.csv, so the file
works from both Windows and WSL.

python local/prepare_zac_csv.py \
    --train-dir ../../../data/train+public_test/train \
    --output ../../../data/metadata.csv
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

DIGITS = "KHÔNG MỘT HAI BA BỐN NĂM SÁU BẢY TÁM CHÍN".split()
SCALES = ["", "NGHÌN", "TRIỆU", "TỶ"]

# The way these codes are spoken cannot be told from the text, e.g. RV123.
AMBIGUOUS = re.compile(r"^(RV\d+|NHOM\d+|\d+A\d+|A\d+)$", re.IGNORECASE)


def read_below_1000(n: int, full: bool) -> List[str]:
    hundreds, tens, units = n // 100, n // 10 % 10, n % 10
    words = []
    if hundreds or full:
        words += [DIGITS[hundreds], "TRĂM"]
    if tens == 0:
        if units and words:
            words.append("LINH")
    elif tens == 1:
        words.append("MƯỜI")
    else:
        words += [DIGITS[tens], "MƯƠI"]
    if units == 0:
        return words
    if units == 1 and tens > 1:
        words.append("MỐT")
    elif units == 4 and tens > 1:
        words.append("TƯ")
    elif units == 5 and tens > 0:
        words.append("LĂM")
    else:
        words.append(DIGITS[units])
    return words


def read_number(digits: str) -> str:
    if len(digits) >= 8 or (len(digits) > 1 and digits[0] == "0"):
        # Phone numbers and other codes are read digit by digit.
        return " ".join(DIGITS[int(d)] for d in digits)
    n = int(digits)
    if n == 0:
        return DIGITS[0]

    groups = []
    while n:
        groups.append(n % 1000)
        n //= 1000

    words = []
    for i in reversed(range(len(groups))):
        if groups[i]:
            words += read_below_1000(groups[i], full=bool(words))
            words += [SCALES[i]] if SCALES[i] else []
    return " ".join(words)


def normalize(text: str) -> Optional[str]:
    """Return normalized text, or None if its reading is ambiguous."""
    for token in text.split():
        if AMBIGUOUS.match(token.strip(".,;:?!…")):
            return None

    text = text.replace("+", " PLUS ")
    text = re.sub(r"\b(\d{1,2})h\b", r"\1 GIỜ", text)
    text = re.sub(r"\d+", lambda m: f" {read_number(m.group())} ", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    return " ".join(text.upper().split())


def get_pairs(sample: dict) -> List[Tuple[str, str]]:
    pairs = [(sample["audio"], sample["audio_text"])]
    for turn in sample["history"]:
        if turn.get("role") == "user":
            pairs.append((turn["audio"], turn["audio_text"]))
    return pairs


def prepare(train_dir: Path, output: Path):
    with open(train_dir / "train.jsonl", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    rows = []
    num_excluded = 0
    for sample in samples:
        for audio, text in get_pairs(sample):
            normalized = normalize(text)
            if normalized:
                path = os.path.relpath(train_dir / audio, output.parent)
                rows.append((normalized, Path(path).as_posix()))
            else:
                num_excluded += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "audio_path"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {output}, excluded {num_excluded}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Build metadata.csv from the ZAC train split."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Directory containing train.jsonl and audio/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path of the output metadata.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    prepare(args.train_dir, args.output)
