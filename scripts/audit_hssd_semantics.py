"""Resolve HSSD scene template hashes to official condensed categories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    with args.metadata.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    hash_column = "Object Hash"
    category_column = next(
        name for name in rows[0] if name.startswith("Semantic Category:\nCONDENSED")
    )
    categories = {row[hash_column]: row[category_column] for row in rows}
    instances = scene.get("object_instances", [])
    resolved = []
    missing = []
    counts: Counter[str] = Counter()
    for instance in instances:
        template = instance["template_name"]
        category = categories.get(template)
        if category:
            counts[category] += 1
        else:
            missing.append(template)
        resolved.append({"template_name": template, "category": category})

    report = {
        "instance_count": len(instances),
        "unique_template_count": len({item["template_name"] for item in instances}),
        "resolved_instance_count": sum(item["category"] is not None for item in resolved),
        "missing_template_hashes": sorted(set(missing)),
        "category_counts": dict(sorted(counts.items())),
        "instances": resolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in report if key != "instances"}))


if __name__ == "__main__":
    main()
