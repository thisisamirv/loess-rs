import json
import numpy as np
from scipy.stats import pearsonr

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_values(val1, val2, tol=1e-6, name='value'):
    if val1 is None or val2 is None:
        return val1 == val2, None

    def to_numeric_array(v):
        # handle scalars
        if isinstance(v, (int, float, np.floating, np.integer)):
            return np.array([float(v)])
        if isinstance(v, str):
            try:
                return np.array([float(v)])
            except:
                if v.lower() in ('na', 'nan', 'none', 'null'):
                    return np.array([np.nan])
                return None
        # try iterable
        try:
            out = []
            for x in v:
                if x is None:
                    out.append(np.nan)
                    continue
                if isinstance(x, (int, float, np.floating, np.integer)):
                    out.append(float(x))
                    continue
                if isinstance(x, str):
                    if x.lower() in ('na', 'nan', 'none', 'null'):
                        out.append(np.nan)
                        continue
                    try:
                        out.append(float(x))
                        continue
                    except:
                        out.append(np.nan)
                        continue
                try:
                    out.append(float(x))
                except:
                    out.append(np.nan)
            return np.array(out, dtype=float)
        except TypeError:
            return None

    a1 = to_numeric_array(val1)
    a2 = to_numeric_array(val2)

    # If unable to coerce either side to numeric arrays, fall back to direct equality
    if a1 is None or a2 is None:
        try:
            return val1 == val2, None
        except Exception:
            return False, "Non-comparable non-numeric values"

    if a1.shape != a2.shape:
        return False, "Shapes differ"

    # positions where both finite numbers exist
    finite_both = np.isfinite(a1) & np.isfinite(a2)
    # positions where both are NaN (treated as equal)
    both_nan = ~np.isfinite(a1) & ~np.isfinite(a2)

    # if all positions are either matching finite values within tol or both NaN, it's a match
    if np.all(both_nan | (finite_both & np.isclose(a1, a2, atol=tol, rtol=1e-8))):
        return True, None, 0.0

    # compute max diff over positions where both finite
    if np.any(finite_both):
        max_diff = np.max(np.abs(a1[finite_both] - a2[finite_both]))
        return False, f"Max diff: {max_diff}", max_diff
    # no comparable numeric positions
    return False, "Non-comparable (no overlapping numeric entries)", None

def compare_diagnostics(diag1, diag2, tol=1e-6):
    if diag1 is None or diag2 is None:
        return diag1 == diag2, None
    results = {}
    for key in set(diag1.keys()) | set(diag2.keys()):
        v1 = diag1.get(key)
        v2 = diag2.get(key)
        if v1 is None or v2 is None:
            results[key] = (v1 == v2, None, None)
        else:
            match, diff_msg, diff_val = compare_values([v1], [v2], tol, key)
            results[key] = (match, diff_msg, diff_val)
    all_match = all(m[0] for m in results.values())
    return all_match, results

def compare_results(results_dict, tol=1e-6, loose_tol=0.01):
    implementations = list(results_dict.keys())
    scenarios = list(results_dict[implementations[0]].keys())
    
    for scenario in scenarios:
        print(f"\n============= Scenario: {scenario} =============")
        data = {impl: results_dict[impl][scenario] for impl in implementations}
        
        # Compare 'y' (smoothed values)
        print("\nComparing smoothed 'y' values:")
        base_y = np.array(data[implementations[0]]['y'])
        for impl in implementations[1:]:
            match, diff, max_diff = compare_values(base_y, data[impl]['y'], tol)
            if match:
                status = "MATCH"
            elif max_diff is not None and max_diff < loose_tol:
                status = f"ACCEPTABLE ({diff})"
            else:
                status = f"MISMATCH ({diff})"
            print(f"{implementations[0]} vs {impl}: {status}")
        
        # Pairwise correlations for 'y'
        print("\nPearson correlations for 'y':")
        for i in range(len(implementations)):
            for j in range(i+1, len(implementations)):
                y1 = np.array(data[implementations[i]]['y'])
                y2 = np.array(data[implementations[j]]['y'])
                corr, _ = pearsonr(y1, y2)
                print(f"{implementations[i]} vs {implementations[j]}: {corr:.6f}")
        
        # Compare fraction_used
        base_frac = data[implementations[0]]['fraction_used']
        header_printed = False
        for impl in implementations[1:]:
            other_frac = data[impl]['fraction_used']
            if base_frac is None and other_frac is None:
                continue
            if not header_printed:
                print("\nFraction used:")
                header_printed = True
            match, diff, _ = compare_values([base_frac], [other_frac], tol)
            status = "MATCH" if match else f"MISMATCH ({diff})"
            print(f"{implementations[0]} vs {impl}: {status}")
        
        # Compare iterations_used (handling None)
        base_iter = data[implementations[0]]['iterations_used']
        header_printed = False
        for impl in implementations[1:]:
            other_iter = data[impl]['iterations_used']
            if base_iter is None and other_iter is None:
                continue
            if not header_printed:
                print("\nIterations used:")
                header_printed = True
            if base_iter == other_iter:
                print(f"{implementations[0]} vs {impl}: MATCH ({base_iter})")
            else:
                print(f"{implementations[0]} vs {impl}: MISMATCH ({base_iter} {implementations[0]} vs {other_iter} {impl})")
        
        # Compare cv_scores
        base_cv = data[implementations[0]]['cv_scores']
        header_printed = False
        for impl in implementations[1:]:
            other_cv = data[impl]['cv_scores']
            if base_cv is None and other_cv is None:
                continue
            if not header_printed:
                print("\nCV scores:")
                header_printed = True
            match, diff, max_diff = compare_values(base_cv, other_cv, tol)
            if match:
                status = "MATCH"
            elif max_diff is not None and max_diff < loose_tol:
                status = f"ACCEPTABLE ({diff})"
            else:
                status = f"MISMATCH ({diff})"
            print(f"{implementations[0]} vs {impl}: {status}")
        
        # Compare diagnostics
        base_diag = data[implementations[0]]['diagnostics']
        header_printed = False
        for impl in implementations[1:]:
            other_diag = data[impl]['diagnostics']
            if base_diag is None and other_diag is None:
                continue
            if not header_printed:
                print("\nDiagnostics:")
                header_printed = True
            match, details = compare_diagnostics(base_diag, other_diag, tol)
            
            # Determine overall status
            overall_status = "MATCH"
            if not match:
                is_acceptable = True
                if details:
                    for key, (m, _, val) in details.items():
                        if not m:
                            if val is None or val >= loose_tol:
                                is_acceptable = False
                                break
                    overall_status = "ACCEPTABLE" if is_acceptable else "MISMATCH"
                else:
                    overall_status = "MISMATCH"

            print(f"{implementations[0]} vs {impl}: {overall_status}")
            if not match:
                if details is None:
                    print("  Details unavailable: one diagnostics value is None or incomparable")
                else:
                    for key, (m, d, val) in details.items():
                        if not m:
                            status_label = "MISMATCH"
                            if val is not None and val < loose_tol:
                                status_label = "ACCEPTABLE"
                            print(f"  {key}: {status_label} ({d})")
        
        # Compare residuals
        extra = 'residuals'
        base_extra = data[implementations[0]].get(extra)
        if base_extra is not None:
            print(f"\n{extra.capitalize()}:")
            for impl in implementations[1:]:
                match, diff, max_diff = compare_values(base_extra, data[impl].get(extra), tol)
                if match:
                    status = "MATCH"
                elif max_diff is not None and max_diff < loose_tol:
                    status = f"ACCEPTABLE ({diff})"
                else:
                    status = f"MISMATCH ({diff})"
                print(f"{implementations[0]} vs {impl}: {status}")

        # Compare robustness_weights with relaxed tolerance
        extra = 'robustness_weights'
        base_extra = data[implementations[0]].get(extra)
        if base_extra is not None:
            print(f"\n{extra.capitalize()}:")
            # Relaxed tolerances for robustness weights
            rob_tol = 1e-4
            rob_loose_tol = 0.05
            for impl in implementations[1:]:
                match, diff, max_diff = compare_values(base_extra, data[impl].get(extra), rob_tol)
                if match:
                    status = "MATCH"
                elif max_diff is not None and max_diff < rob_loose_tol:
                    status = f"ACCEPTABLE ({diff})"
                else:
                    status = f"MISMATCH ({diff})"
                print(f"{implementations[0]} vs {impl}: {status}")

if __name__ == "__main__":
    files = {
        'skmisc': 'output/skmisc_validate.json',
        'this_crate': 'output/rust_validate.json'
    }
    
    results_dict = {}
    for name, path in files.items():
        try:
            results_dict[name] = load_json(path)
            print(f"Loaded {name} from {path}")
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping {name}")
    
    if len(results_dict) < 2:
        print("Need at least two results to compare")
    else:
        compare_results(results_dict, tol=1e-6)