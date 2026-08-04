import argparse
from pathlib import Path

from lhotse import Fbank, FbankConfig, load_manifest


def compute_fbank(
    manifest_path: str,
    storage_dir: str,
    num_jobs: int = 8,
):
    cuts = load_manifest(manifest_path)

    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    cuts = cuts.compute_and_store_features(
        extractor=Fbank(FbankConfig()),
        storage_path=str(storage_dir),
        num_jobs=num_jobs,
    )

    cuts.to_file(manifest_path)

    print(f"Updated manifest: {manifest_path}")
    print(f"Features stored in: {storage_dir}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Compute Fbank features for a Lhotse CutSet."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to cuts_*.jsonl.gz",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        required=True,
        help="Directory to store extracted features.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=8,
        help="Number of parallel workers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    compute_fbank(
        manifest_path=args.manifest,
        storage_dir=args.storage_dir,
        num_jobs=args.num_jobs,
    )