import json
import os
import glob
import numpy as np
import sys

SCIKIT_DIR = "output/scikit"
LOESS_RS_DIR = "output/loess_rs"

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compare_scenarios():
    scikit_files = glob.glob(os.path.join(SCIKIT_DIR, "*.json"))
    scikit_files.sort()
    
    if not scikit_files:
        print(f"No files found in {SCIKIT_DIR}")
        return

    print(f"{'SCENARIO':<30} | {'STATUS':<15} | {'MAX DIFF':<15} | {'RMSE':<15}")
    print("-" * 85)

    failures = []

    for s_path in scikit_files:
        filename = os.path.basename(s_path)
        l_path = os.path.join(LOESS_RS_DIR, filename)
        
        name = filename.replace(".json", "")

        if not os.path.exists(l_path):
            print(f"{name:<30} | {'MISSING':<15} | {'-':<15} | {'-':<15}")
            failures.append(name)
            continue

        try:
            s_data = load_json(s_path)
            l_data = load_json(l_path)
            
            # Extract fitted values
            # Structure: {"result": {"fitted": [...]}}
            s_fitted = np.array(s_data['result']['fitted'])
            l_fitted = np.array(l_data['result']['fitted'])
            
            if s_fitted.shape != l_fitted.shape:
                print(f"{name:<30} | {'SHAPE MISMATCH':<15} | {'-':<15} | {'-':<15}")
                failures.append(name)
                continue

            # Compare
            diff = np.abs(s_fitted - l_fitted)
            max_diff = np.max(diff)
            rmse = np.sqrt(np.mean(diff**2))
            
            # Tolerance for floating point differences across implementations
            # 1e-6 is a reasonable target for numerical consistency
            tolerance = 1e-6
            
            if max_diff < tolerance:
                status = "MATCH"
            else:
                status = "MISMATCH"
                failures.append(name)

            print(f"{name:<30} | {status:<15} | {max_diff:<15.2e} | {rmse:<15.2e}")

        except Exception as e:
            print(f"{name:<30} | {'ERROR':<15} | {str(e):<15} | {'-':<15}")
            failures.append(name)

    print("-" * 85)
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll scenarios passed!")
        sys.exit(0)

if __name__ == "__main__":
    compare_scenarios()