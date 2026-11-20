# check_balance.py
import json, glob, collections, sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
files = sorted(DATA_DIR.glob("*.json"))
print(f"found {len(files)} files in {DATA_DIR}")

bn, sec, scores = [], [], []
for f in files:
    d = json.load(open(f))
    meta = d.get("metadata", d)          # на случай если метки в корне
    bn.append(meta.get("bottleneck"))
    sec.append(meta.get("security_flaw"))
    if "score" in meta:
        scores.append(float(meta["score"]))

print("bottleneck     :", collections.Counter(bn))
print("security_flaw  :", collections.Counter(sec))
if scores:
    print(f"score: n={len(scores)}  min={min(scores):.3f}  "
          f"max={max(scores):.3f}  mean={sum(scores)/len(scores):.3f}")