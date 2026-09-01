"""Merge protocol-locked 2D and source-only DINO pilot metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHODS = ("global", "semantic_2d", "global_dino", "semantic_dino")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--metrics-2d", type=Path, required=True)
    parser.add_argument("--warp-2d", type=Path, required=True)
    parser.add_argument("--metrics-dino", type=Path, required=True)
    parser.add_argument("--warp-dino", type=Path, required=True)
    parser.add_argument("--dino-appearance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def check_report(report: dict, pair_id: str, methods: tuple[str, ...], label: str) -> None:
    if report.get("status") != "passed":
        raise RuntimeError(f"{label} has not passed")
    if report.get("pair_id") != pair_id:
        raise RuntimeError(f"{label} pair_id differs from frozen protocol")
    missing = set(methods) - set(report.get("methods", {}))
    if missing:
        raise RuntimeError(f"{label} is missing methods: {sorted(missing)}")


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = load(protocol_path)
    if protocol.get("status") != "frozen":
        raise RuntimeError("protocol is not frozen")
    rules = protocol.get("rules", {})
    if rules.get("appearance_source") != "source_donor_only":
        raise RuntimeError("protocol does not require source-donor-only appearance")
    if rules.get("consistency_views_excluded_from_appearance_training") is not True:
        raise RuntimeError("protocol does not exclude consistency views from training")

    paths = {
        "metrics_2d": args.metrics_2d.resolve(),
        "warp_2d": args.warp_2d.resolve(),
        "metrics_dino": args.metrics_dino.resolve(),
        "warp_dino": args.warp_dino.resolve(),
    }
    reports = {name: load(path) for name, path in paths.items()}
    check_report(reports["metrics_2d"], protocol["pair_id"], ("global", "semantic_2d"), "2D metrics")
    check_report(reports["warp_2d"], protocol["pair_id"], ("global", "semantic_2d"), "2D warp metrics")
    check_report(reports["metrics_dino"], protocol["pair_id"], ("global_dino", "semantic_dino"), "DINO metrics")
    check_report(reports["warp_dino"], protocol["pair_id"], ("global_dino", "semantic_dino"), "DINO warp metrics")
    for key in ("warp_2d", "warp_dino"):
        if reports[key].get("protocol_split") != "target_consistency":
            raise RuntimeError(f"{key} does not evaluate frozen consistency views")

    appearance_hashes = {}
    appearance_root = args.dino_appearance_root.resolve()
    for method in ("global_dino", "semantic_dino"):
        report_path = appearance_root / method / "baseline_report.json"
        appearance = load(report_path)
        if appearance.get("pair_id") != protocol["pair_id"]:
            raise RuntimeError(f"{method} appearance pair_id differs from protocol")
        if appearance.get("target_rgb_accessed") is not False:
            raise RuntimeError(f"{method} does not prove target_rgb_accessed=false")
        if appearance.get("donor_view_count") != len(protocol["selection"]["source_donor"]):
            raise RuntimeError(f"{method} donor count differs from protocol")
        appearance_hashes[method] = sha256(report_path)

    table = {}
    for method in METHODS:
        metric_key = "metrics_dino" if method.endswith("dino") else "metrics_2d"
        warp_key = "warp_dino" if method.endswith("dino") else "warp_2d"
        summary = reports[metric_key]["methods"][method]["summary"]
        warp = reports[warp_key]["methods"][method]
        table[method] = {
            "l1": summary["l1"],
            "psnr_db": summary["psnr_db"],
            "semantic_color_leakage_rate": summary["semantic_color_leakage_rate"],
            "evaluated_pixels": summary["evaluated_pixels"],
            "warp_l1": warp["warp_l1"],
            "warp_rmse": warp["warp_rmse"],
            "warp_correspondences": warp["correspondences"],
        }

    report = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": sha256(protocol_path),
        "appearance_source": "source_donor_only",
        "target_rgb_accessed_by_dino_training": False,
        "method_order": list(METHODS),
        "methods": table,
        "source_reports": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()
        },
        "dino_appearance_report_sha256": appearance_hashes,
        "interpretation_note": (
            "Warp metrics must be interpreted together with held-out visual error; "
            "a spatially constant method can have zero warp error by construction."
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "methods": table}))


if __name__ == "__main__":
    main()
