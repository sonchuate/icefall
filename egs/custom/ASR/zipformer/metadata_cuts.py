"""Build train/dev CutSets from a metadata.csv with columns text and audio_path."""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet


def _make_cuts(rows: List[Dict[str, str]], root: Path, repeat: bool) -> CutSet:
    """Rows of the same audio file must have the same text. If repeat is True, an
    audio file listed N times gives N cuts, otherwise it gives one cut.
    """
    texts = {}
    for row in rows:
        path = row["audio_path"]
        assert texts.setdefault(path, row["text"]) == row["text"], (
            f"Different texts for {path}"
        )
    counts = Counter(row["audio_path"] for row in rows)

    recordings = []
    supervisions = []
    repeats = {}
    for path, text in texts.items():
        recording = Recording.from_file(root / path)
        recordings.append(recording)
        repeats[recording.id] = counts[path] if repeat else 1
        supervisions.append(
            SupervisionSegment(
                id=recording.id,
                recording_id=recording.id,
                start=0.0,
                duration=recording.duration,
                text=text,
                language="vi",
            )
        )

    cuts = CutSet.from_manifests(
        recordings=RecordingSet.from_recordings(recordings),
        supervisions=SupervisionSet.from_segments(supervisions),
    )
    return CutSet.from_cuts(
        cut.with_id(f"{cut.id}-repeat{i}")
        for cut in cuts
        for i in range(repeats[cut.recording.id])
    )


def load_metadata_cuts(
    metadata_csv: Path, dev_ratio: float, seed: int
) -> Tuple[CutSet, CutSet]:
    """Split by the parent directory of audio_path, so that all turns of one
    conversation stay in the same split. Repeated rows oversample an utterance in
    train and are ignored in dev.

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
    return _make_cuts(train_rows, root, repeat=True), _make_cuts(
        dev_rows, root, repeat=False
    )
