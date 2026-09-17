import pickle
import torch

path = "artifacts/exp7_detector_dataset.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

pairs = {}

for item in data:
    key = (
        item["sample_id"],
        item["layer_id"],
        item["patch_start"],
    )

    pairs.setdefault(key, {})[item["label"]] = item["features"].float()

total_changed = 0
total_values = 0
pair_diffs = []

for key, pair in pairs.items():

    if 0 not in pair or 1 not in pair:
        continue

    clean = pair[0]
    stego = pair[1]

    diff = (stego - clean).abs()

    changed = (diff != 0).sum().item()

    total_changed += changed
    total_values += diff.numel()

    pair_diffs.append(diff.mean().item())

print("PAIRED DATASET CHECK")
print("====================")

print("Complete pairs:", len(pair_diffs))
print("Total values:", total_values)
print("Changed values:", total_changed)

if total_values:
    print(
        "Changed ratio:",
        total_changed / total_values
    )

print("\nPAIR-WISE MEAN ABS DIFF")
print("Mean:", sum(pair_diffs) / len(pair_diffs))
print("Min :", min(pair_diffs))
print("Max :", max(pair_diffs))