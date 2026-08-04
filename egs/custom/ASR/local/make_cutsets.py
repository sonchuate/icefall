from lhotse import load_manifest
from lhotse import CutSet

recordings = load_manifest(
    "data/manifests/train/recordings.jsonl.gz"
)

supervisions = load_manifest(
    "data/manifests/train/supervisions.jsonl.gz"
)

cuts = CutSet.from_manifests(
    recordings=recordings,
    supervisions=supervisions
)

cuts.to_file(
    "data/fbank/cuts_train.jsonl.gz"
)