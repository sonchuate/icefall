"""Build train/dev CutSets from a metadata.csv with columns text and audio_path."""

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet


def _make_cuts(rows: List[Dict[str, str]], root: Path) -> CutSet:
    recordings = []
    supervisions = []
    for row in rows:
        recording = Recording.from_file(root / row["audio_path"])
        recordings.append(recording)
        supervisions.append(
            SupervisionSegment(
                id=recording.id,
                recording_id=recording.id,
                start=0.0,
                duration=recording.duration,
                text=row["text"],
                language="vi",
            )
        )

    return CutSet.from_manifests(
        recordings=RecordingSet.from_recordings(recordings),
        supervisions=SupervisionSet.from_segments(supervisions),
    )


def load_metadata_cuts(
    metadata_csv: Path, dev_ratio: float, seed: int
) -> Tuple[CutSet, CutSet]:
    """Split by the parent directory of audio_path, so that all turns of one
    conversation stay in the same split.

    Returns:
      A tuple (train_cuts, dev_cuts).
    """
    metadata_csv = Path(metadata_csv)
    with open(metadata_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for row in rows:
        groups[Path(row["audio_path"]).parent.name].append(row)

    names = sorted(groups)
    random.Random(seed).shuffle(names)
    num_dev = int(len(names) * dev_ratio)

    root = metadata_csv.parent
    dev_rows = [row for name in names[:num_dev] for row in groups[name]]
    train_rows = [row for name in names[num_dev:] for row in groups[name]]
    return _make_cuts(train_rows, root), _make_cuts(dev_rows, root)
