import json
from pathlib import Path
from collections import Counter

from heuristics import analyze_topology

DATA_DIR = Path("/tmp/linktelligence-stress-dk9sko8u/data")

BUCKETS = {
    "20-39": (20, 40),
    "40-59": (40, 60),
    "60-79": (60, 80),
    "80-100": (80, 101),
}

stats = {
    name: {
        "total": 0,
        "classes": Counter(),
        "issue_a": 0,
        "issue_b": 0,
        "issue_combinations": Counter(),
        "node_types": Counter(),
    }
    for name in BUCKETS
}

for path in sorted(DATA_DIR.glob("sample_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))

    nodes = data["nodes"]
    edges = data["edges"]
    n = len(nodes)

    bucket_name = next(
        name
        for name, (lo, hi) in BUCKETS.items()
        if lo <= n < hi
    )

    report = analyze_topology(nodes, edges)
    security = report["security"]

    issue_a = security["issue_a"]["present"]
    issue_b = security["issue_b"]["present"]

    current = stats[bucket_name]

    current["total"] += 1
    current["classes"][security["class"]] += 1
    current["issue_a"] += int(issue_a)
    current["issue_b"] += int(issue_b)
    current["issue_combinations"][(issue_a, issue_b)] += 1

    current["node_types"].update(
        node["type"] for node in nodes
    )


for name, current in stats.items():
    total = current["total"]

    if not total:
        continue

    print(f"\n{'=' * 50}")
    print(f"{name} nodes | {total} graphs")
    print("=" * 50)

    print("Security classes:", dict(sorted(current["classes"].items())))

    print(
        f"Issue A (unprotected hosts): "
        f"{current['issue_a']} ({current['issue_a'] / total:.1%})"
    )

    print(
        f"Issue B (printer-router): "
        f"{current['issue_b']} ({current['issue_b'] / total:.1%})"
    )

    print("Combinations (issue A, issue B):")

    for combination, count in sorted(
        current["issue_combinations"].items()
    ):
        print(f"  {combination}: {count} ({count / total:.1%})")

    print("Node types:", dict(sorted(current["node_types"].items())))