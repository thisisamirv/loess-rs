#!/usr/bin/env python3
"""
Convert Criterion benchmark results to JSON format for comparison.

Criterion stores results in:
- target/criterion/<group>/<bench_id>/<param>/new/estimates.json (parameterized)
- target/criterion/<group>/<bench_name>/new/estimates.json (non-parameterized)

This script extracts timing data and writes it to benchmarks/output/rust_benchmark.json.

Usage: python3 convert_criterion.py
"""

import json
from pathlib import Path
from typing import Dict, List


def parse_estimates(estimates_file: Path) -> dict | None:
    """Parse a criterion estimates.json file and return timing in ms."""
    try:
        with open(estimates_file) as f:
            estimates = json.load(f)
        
        # Criterion stores in nanoseconds
        mean_ns = estimates.get("mean", {}).get("point_estimate", 0)
        std_ns = estimates.get("std_dev", {}).get("point_estimate", 0)
        median_ns = estimates.get("median", {}).get("point_estimate", 0)
        
        # Try to get min/max from sample data
        sample_file = estimates_file.parent / "sample.json"
        min_ns = mean_ns
        max_ns = mean_ns
        if sample_file.exists():
            try:
                with open(sample_file) as f:
                    sample = json.load(f)
                times = sample.get("times", [])
                if times:
                    min_ns = min(times)
                    max_ns = max(times)
            except Exception:
                pass
        
        return {
            "mean_time_ms": mean_ns / 1_000_000,
            "std_time_ms": std_ns / 1_000_000,
            "median_time_ms": median_ns / 1_000_000,
            "min_time_ms": min_ns / 1_000_000,
            "max_time_ms": max_ns / 1_000_000,
        }
    except Exception as e:
        print(f"Error parsing {estimates_file}: {e}")
        return None


def find_criterion_results(criterion_dir: Path) -> Dict[str, List[dict]]:
    """Find all criterion benchmark results and organize by category."""
    results: Dict[str, List[dict]] = {}
    
    if not criterion_dir.exists():
        print(f"Criterion directory not found: {criterion_dir}")
        return results
    
    # Walk through criterion output structure
    for group_dir in criterion_dir.iterdir():
        if not group_dir.is_dir() or group_dir.name == "report":
            continue
        
        category = group_dir.name
        if category not in results:
            results[category] = []
        
        for bench_dir in group_dir.iterdir():
            if not bench_dir.is_dir() or bench_dir.name == "report":
                continue
            
            bench_id = bench_dir.name
            
            # Check if this has 'new/' directly (non-parameterized)
            # or has parameter subdirectories
            new_dir = bench_dir / "new"
            if new_dir.exists() and (new_dir / "estimates.json").exists():
                # Non-parameterized benchmark (e.g., pathological/clustered)
                timing = parse_estimates(new_dir / "estimates.json")
                if timing:
                    result = {
                        "name": bench_id,
                        "size": 5000,  # Default size for pathological
                        "iterations": 10,
                        **timing
                    }
                    results[category].append(result)
            else:
                # Parameterized benchmark - look for param subdirectories
                for param_dir in bench_dir.iterdir():
                    if not param_dir.is_dir() or param_dir.name in ("report", "new", "base", "change"):
                        continue
                    
                    param = param_dir.name
                    estimates_file = param_dir / "new" / "estimates.json"
                    
                    if estimates_file.exists():
                        timing = parse_estimates(estimates_file)
                        if timing:
                            # Try to parse param as size
                            try:
                                size = int(param)
                            except ValueError:
                                size = 0
                            
                            # Create aligned name: category_param (e.g., scale_1000)
                            # Match Python naming convention
                            if category == "scalability":
                                name = f"scale_{param}"
                            elif category == "financial":
                                name = f"financial_{param}"
                            elif category == "scientific":
                                name = f"scientific_{param}"
                            elif category == "genomic":
                                name = f"genomic_{param}"
                            elif category == "fraction":
                                name = f"fraction_{param}"
                            elif category == "iterations":
                                name = f"iterations_{param}"
                            elif category == "polynomial_degrees":
                                # bench_id is "degree", param is "linear", etc.
                                name = f"{bench_id}_{param}"
                            elif category == "distance_metrics":
                                # bench_id is "metric", param is "euclidean", etc.
                                name = f"{bench_id}_{param}"
                            else:
                                name = f"{bench_id}_{param}"
                            
                            result = {
                                "name": name,
                                "size": size,
                                "iterations": 10,
                                **timing
                            }
                            results[category].append(result)
    
    # Sort results within each category by name
    for category in results:
        results[category].sort(key=lambda x: x["name"])
    
    return results


def main():
    script_dir = Path(__file__).resolve().parent
    criterion_dir = script_dir / "loess-rs" / "target" / "criterion"
    output_dir = script_dir / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading criterion results from: {criterion_dir}")
    
    results = find_criterion_results(criterion_dir)
    
    if not results:
        print("No criterion results found. Run 'cargo bench' first.")
        return 1
    
    # Write to output file
    output_file = output_dir / "rust_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    total_benchmarks = sum(len(v) for v in results.values())
    print(f"\nExported {total_benchmarks} benchmarks across {len(results)} categories:")
    for category, benchmarks in sorted(results.items()):
        print(f"  {category}: {len(benchmarks)} benchmarks")
        for b in benchmarks:
            print(f"    - {b['name']}: {b['mean_time_ms']:.3f} ms")
    
    return 0


if __name__ == "__main__":
    exit(main())
