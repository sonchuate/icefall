import argparse
from pathlib import Path

from lhotse import CutSet, load_manifest


def make_cutset(
    recordings_path: str,
    supervisions_path: str,
    output_path: str,
):
    recordings = load_manifest(recordings_path)
    supervisions = load_manifest(supervisions_path)

    cuts = CutSet.from_manifests(
        recordings=recordings,
        supervisions=supervisions,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cuts.to_file(output_path)

    print(f"Saved {len(cuts)} cuts to {output_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Create a Lhotse CutSet from recordings and supervisions."
    )

    parser.add_argument(
        "--recordings",
        type=str,
        required=True,
        help="Path to recordings.jsonl.gz",
    )

    parser.add_argument(
        "--supervisions",
        type=str,
        required=True,
        help="Path to supervisions.jsonl.gz",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path of cuts_*.jsonl.gz",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    make_cutset(
        recordings_path=args.recordings,
        supervisions_path=args.supervisions,
        output_path=args.output,
    )