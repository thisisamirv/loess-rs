import json
from pathlib import Path
from statistics import mean, median, stdev
import csv
import math

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_time_value(entry: dict):
    """Robustly pick a numeric timing from an entry.
    Prefer mean_time_ms, then median_time_ms, then max_time_ms, then any numeric field.
    Returns (value_ms: float or None, size: int or None).
    """
    for key in ("mean_time_ms", "median_time_ms", "max_time_ms"):
        if key in entry:
            try:
                return float(entry[key]), entry.get("size")
            except Exception:
                pass
    # fallback: search for first numeric value
    for k, v in entry.items():
        if isinstance(v, (int, float)):
            # ignore small integer metadata like iteration counts if name-like keys present
            if k in ("iterations", "size", "runs"):
                continue
            try:
                return float(v), entry.get("size")
            except Exception:
                pass
    return None, entry.get("size")

def build_map(entries):
    # allow entries that might already be a dict of results
    out = {}
    for e in entries:
        name = e.get("name") or e.get("id") or e.get("test") or None
        if not name:
            # generate fallback unique name if missing
            name = json.dumps(e, sort_keys=True)
        out[name] = e
    return out

def compare_category(rust_entries, R_entries):
    rust_map = build_map(rust_entries)
    R_map = build_map(R_entries)
    common = sorted(set(rust_map.keys()) & set(R_map.keys()))
    rows = []
    speedups = []
    for name in common:
        r_entry = rust_map[name]
        s_entry = R_map[name]
        r_val, r_size = pick_time_value(r_entry)
        s_val, s_size = pick_time_value(s_entry)

        row = {
            "name": name,
            "rust_value_ms": r_val,
            "R_value_ms": s_val,
            "rust_size": r_size,
            "R_size": s_size,
            "notes": []
        }

        if r_val is None or s_val is None:
            row["notes"].append("missing_metric")
            rows.append(row)
            continue

        # core comparisons
        if r_val == 0 or s_val == 0:
            speedup = None
        else:
            speedup = s_val / r_val  # >1 => R faster by this factor
        row["speedup_R_over_rust"] = speedup
        if speedup is not None:
            row["log2_speedup"] = math.log2(speedup) if speedup > 0 else None
            row["percent_change_R_vs_rust"] = ((s_val - r_val) / r_val) * 100.0
            speedups.append(speedup)

        # absolute diffs
        row["absolute_diff_ms"] = None if r_val is None or s_val is None else (s_val - r_val)
        row["abs_percent_vs_rust"] = None if r_val == 0 else abs(row["absolute_diff_ms"]) / r_val * 100.0

        # per-point normalization if size available and >0
        size = r_size or s_size
        if size:
            try:
                size_i = int(size)
                row["rust_ms_per_point"] = r_val / size_i
                row["R_ms_per_point"] = s_val / size_i
                row["speedup_per_point"] = None if row["rust_ms_per_point"] == 0 else row["R_ms_per_point"] / row["rust_ms_per_point"]
            except Exception:
                row["notes"].append("bad_size")

        rows.append(row)
    summary = {
        "compared": len(common),
        "mean_speedup": mean(speedups) if speedups else None,
        "median_speedup": median(speedups) if speedups else None,
        "count_with_metrics": len(speedups),
    }
    return rows, summary

def main():
    repo_root = Path(__file__).resolve().parent
    # walk up to workspace root (same heuristic as other scripts)
    workspace = repo_root
    for _ in range(6):
        if (workspace / "output").exists():
            break
        if workspace.parent == workspace:
            break
        workspace = workspace.parent
    out_dir = workspace / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    rust_path = out_dir / "rust_benchmark.json"
    r_path = out_dir / "r_benchmark.json"

    if not rust_path.exists() or not r_path.exists():
        missing = []
        if not rust_path.exists():
            missing.append(str(rust_path))
        if not r_path.exists():
            missing.append(str(r_path))
        print("Missing files:", ", ".join(missing))
        return

    rust = load_json(rust_path)
    r_data = load_json(r_path)

    all_keys = sorted(set(rust.keys()) | set(r_data.keys()))
    comparison = {}
    overall_speedups = []

    # detailed rows for CSV
    csv_rows = []
    csv_fieldnames = [
        "category","name","rust_value_ms","R_value_ms","speedup_R_over_rust",
        "log2_speedup","percent_change_R_vs_rust","absolute_diff_ms","abs_percent_vs_rust",
        "rust_size","R_size","rust_ms_per_point","R_ms_per_point","speedup_per_point","notes"
    ]

    for key in all_keys:
        rust_entries = rust.get(key, [])
        r_entries = r_data.get(key, [])
        rows, summary = compare_category(rust_entries, r_entries)
        comparison[key] = {"rows": rows, "summary": summary}
        if summary["median_speedup"] is not None:
            overall_speedups.append(summary["median_speedup"])
        for row in rows:
            csv_rows.append({
                "category": key,
                "name": row.get("name"),
                "rust_value_ms": row.get("rust_value_ms"),
                "R_value_ms": row.get("R_value_ms"),
                "speedup_R_over_rust": row.get("speedup_R_over_rust"),
                "log2_speedup": row.get("log2_speedup"),
                "percent_change_R_vs_rust": row.get("percent_change_R_vs_rust"),
                "absolute_diff_ms": row.get("absolute_diff_ms"),
                "abs_percent_vs_rust": row.get("abs_percent_vs_rust"),
                "rust_size": row.get("rust_size"),
                "R_size": row.get("R_size"),
                "rust_ms_per_point": row.get("rust_ms_per_point"),
                "R_ms_per_point": row.get("R_ms_per_point"),
                "speedup_per_point": row.get("speedup_per_point"),
                "notes": ";".join(row.get("notes", []))
            })

    print("\nBenchmark comparison (R_ms / rust_ms):")
    for key, data in comparison.items():
        s = data["summary"]
        print(f"- {key}: compared={s['compared']}, median_speedup={s['median_speedup']}, mean_speedup={s['mean_speedup']}")

    # Top wins and regressions across all categories
    all_rows = [r for cat in comparison.values() for r in cat["rows"] if r.get("speedup_R_over_rust") is not None]
    if all_rows:
        sorted_by_speed = sorted(all_rows, key=lambda r: r["speedup_R_over_rust"] or 0, reverse=True)
        sorted_by_regression = sorted(all_rows, key=lambda r: r["speedup_R_over_rust"] or 0)

        print("\nTop 10 Rust wins (largest rust_ms / R_ms):")
        for r in sorted_by_speed[:10]:
            print(f"  {r['name']}: rust={r['rust_value_ms']:.4f}ms, R={r['R_value_ms']:.4f}ms, speedup={r['speedup_R_over_rust']:.2f}x")

        print("\nTop 10 regressions (R faster than Rust):")
        for r in sorted_by_regression[:10]:
            if r["speedup_R_over_rust"] < 1.0:
                print(f"  {r['name']}: rust={r['rust_value_ms']:.4f}ms, R={r['R_value_ms']:.4f}ms, speedup={r['speedup_R_over_rust']:.2f}x")

    # Print detailed per-category rows to console
    print("\nDetailed per-category results:")
    for cat, data in comparison.items():
        rows = data["rows"]
        if not rows:
            continue
        print(f"\nCategory: {cat} (compared={data['summary']['compared']})")
        # header
        print(f"{'name':60} {'rust_ms':>10} {'R_ms':>10} {'speedup':>8} {'%chg':>8} {'notes'}")
        for r in rows:
            name = (r.get("name") or "")[:60].ljust(60)
            rust_v = r.get("rust_value_ms")
            R_v = r.get("R_value_ms")
            sp = r.get("speedup_R_over_rust")
            pct = r.get("percent_change_R_vs_rust")
            notes = ";".join(r.get("notes", []))
            rust_s = f"{rust_v:.4f}" if isinstance(rust_v, (int, float)) else "N/A"
            R_s = f"{R_v:.4f}" if isinstance(R_v, (int, float)) else "N/A"
            sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "N/A"
            pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "N/A"
            print(f"{name} {rust_s:>10} {R_s:>10} {sp_s:>8} {pct_s:>8} {notes}")

if __name__ == "__main__":
    main()
