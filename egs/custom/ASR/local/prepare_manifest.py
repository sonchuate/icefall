"""python prepare_manifest.py \
    --csv dev.csv \
    --output-dir data/manifests/dev
"""



import argparse
from pathlib import Path

import pandas as pd
from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
)


def prepare(csv_path: str, output_dir: str):
    df = pd.read_csv(csv_path)

    recordings = []
    supervisions = []

    for idx, row in df.iterrows():
        recording = Recording.from_file(row["audio_path"])

        recordings.append(recording)

        supervisions.append(
            SupervisionSegment(
                id=str(idx),
                recording_id=recording.id,
                start=0.0,
                duration=recording.duration,
                text=row["text"],
                language="vi",
            )
        )

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recording_set.to_file(output_dir / "recordings.jsonl.gz")
    supervision_set.to_file(output_dir / "supervisions.jsonl.gz")

    print(f"Saved manifests to: {output_dir}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare Lhotse manifests from a CSV file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file containing audio_path and text columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for recordings.jsonl.gz and supervisions.jsonl.gz.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    prepare(args.csv, args.output_dir)