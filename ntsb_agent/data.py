"""Load the episodes built in 02. One episode per line (.jsonl)."""
import json


def load(data_dir, split):
    episodes = []
    with open(f"{data_dir}/episodes_{split}.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def load_all(data_dir):
    train = load(data_dir, "train")
    val = load(data_dir, "val")
    test = load(data_dir, "test")
    return train, val, test
