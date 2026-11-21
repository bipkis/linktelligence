import json
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np

from generate_data import generate_random_graph, assign_attributes
from heuristics import calculate_labels


SEED = 3030

SIZE_BUCKETS = {
    "5-19": (5, 19),
    "20-39": (20, 39),
    "40-59": (40, 59),
    "60-100": (60, 100),
}

# (issue_a, issue_b)
SCENARIOS = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]

SAMPLES_PER_BUCKET = 500
SAMPLES_PER_SCENARIO = SAMPLES_PER_BUCKET // len(SCENARIOS)


def apply_security_scenario(nodes, edges, rng, issue_a, issue_b):
    """Force the requested security issue combination."""

    node_types = {
        node["id"]: node["type"]
        for node in nodes
    }

    # Issue B: direct printer-router connection.
    if issue_b:
        edge = edges[int(rng.integers(0, len(edges)))]

        node_types[edge["source"]] = "printer"
        node_types[edge["target"]] = "router"

    else:
        # Remove every direct printer-router connection.
        for edge in edges:
            source = edge["source"]
            target = edge["target"]

            if {
                node_types[source],
                node_types[target],
            } == {"printer", "router"}:

                printer_id = (
                    source
                    if node_types[source] == "printer"
                    else target
                )

                node_types[printer_id] = "switch"

    # Issue A: hosts without firewall and IDS/IPS.
    if issue_a:
        for node_id, node_type in node_types.items():
            if node_type in {"firewall", "ids_ips"}:
                node_types[node_id] = "switch"

        if "host" not in node_types.values():
            candidates = [
                node_id
                for node_id, node_type in node_types.items()
                if node_type not in {"router", "printer"}
            ]

            node_id = int(rng.choice(candidates))
            node_types[node_id] = "host"

    else:
        has_protection = any(
            node_type in {"firewall", "ids_ips"}
            for node_type in node_types.values()
        )

        if not has_protection:
            candidates = [
                node_id
                for node_id, node_type in node_types.items()
                if node_type not in {"router", "printer"}
            ]

            node_id = int(rng.choice(candidates))
            node_types[node_id] = "firewall"

    for node in nodes:
        node["type"] = node_types[node["id"]]


def main():
    rng = np.random.default_rng(SEED)

    output_root = Path(
        tempfile.mkdtemp(prefix="linktelligence-balanced-v3-")
    )

    data_dir = output_root / "data"
    data_dir.mkdir(parents=True)

    generation_plan = []

    for bucket, (min_nodes, max_nodes) in SIZE_BUCKETS.items():
        for issue_a, issue_b in SCENARIOS:
            for _ in range(SAMPLES_PER_SCENARIO):
                num_nodes = int(
                    rng.integers(min_nodes, max_nodes + 1)
                )

                generation_plan.append(
                    (bucket, num_nodes, issue_a, issue_b)
                )

    rng.shuffle(generation_plan)

    distributions = Counter()
    size_counts = Counter()

    for i, (bucket, num_nodes, issue_a, issue_b) in enumerate(
        generation_plan
    ):
        graph = generate_random_graph(num_nodes, rng)

        nodes, edges = assign_attributes(graph, rng)

        apply_security_scenario(
            nodes,
            edges,
            rng,
            issue_a,
            issue_b,
        )

        labels = calculate_labels(nodes, edges)

        expected_security = int(issue_a) + int(issue_b)

        if labels["security_flaw"] != expected_security:
            raise RuntimeError(
                f"Scenario mismatch in sample {i + 1}: "
                f"expected {expected_security}, "
                f"got {labels['security_flaw']}"
            )

        sample = {
            "sample_id": i + 1,
            "nodes": nodes,
            "edges": edges,
            "metadata": labels,
        }

        path = data_dir / f"sample_{i + 1:04d}.json"

        path.write_text(
            json.dumps(sample, indent=2),
            encoding="utf-8",
        )

        distributions[(bucket, issue_a, issue_b)] += 1
        size_counts[bucket] += 1

    print(f"Generated {len(generation_plan)} samples.")
    print(f"Dataset: {data_dir}")

    print("\nSize distribution:")
    for bucket, count in sorted(size_counts.items()):
        print(f"  {bucket}: {count}")

    print("\nSecurity scenario distribution:")
    for key, count in sorted(distributions.items()):
        print(f"  {key}: {count}")

    print("\nGeneration completed successfully.")


if __name__ == "__main__":
    main()