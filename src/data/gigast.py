"""GigaST data: load translation lookup, stream HF audio, join on segment id."""
import json
from datasets import load_dataset


def load_lookup(jsonl_path):
    """Load {sid: translation} from a pre-extracted GigaST jsonl."""
    d = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            d[r["sid"]] = r["text"]
    return d


def stream_pairs(lookup, subset="s", split="train",
                 max_duration=None, limit=None):
    """Yield (audio_array, sr, target) by joining HF GigaSpeech with lookup."""
    gs = load_dataset("speechcolab/gigaspeech", subset,
                      split=split, streaming=True)
    n = 0
    for ex in gs:
        sid = ex["segment_id"]
        if sid not in lookup:
            continue
        audio = ex["audio"]["array"]
        sr = ex["audio"]["sampling_rate"]
        if max_duration is not None and len(audio) / sr > max_duration:
            continue
        yield audio, sr, lookup[sid]
        n += 1
        if limit is not None and n >= limit:
            break
