import numpy as np
from skmisc.loess import loess
import json
import os

OUTPUT_DIR = "output/scikit/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_scenario(name, x, y, frac, deg, iter, notes="", **kwargs):
    print(f"Running scenario: {name}")
    
    # Configure LOESS
    control = {'iterations': iter}
    l = loess(x, y, span=frac, degree=deg, control=control, **kwargs)
    l.fit()
    
    fitted = l.predict(x, stderror=False).values
    
    # Sanitize for JSON (numpy types)
    x_list = x.tolist()
    y_list = y.tolist()
    fitted_list = fitted.tolist()
    
    data = {
        "name": name,
        "notes": notes,
        "input": {
            "x": x_list,
            "y": y_list
        },
        "params": {
            "fraction": frac,
            "degree": deg,
            "iterations": iter,
            "extra": kwargs
        },
        "result": {
            "fitted": fitted_list
        }
    }
    
    path = os.path.join(OUTPUT_DIR, f"{name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def generate_data(n=100, kind='linear', noise=0.0, range_min=0.0, range_max=1.0, outlier_ratio=0.0):
    np.random.seed(42) # Fixed seed for reproducibility
    x = np.linspace(range_min, range_max, n)
    
    if kind == 'linear':
        y = 2 * x + 1
    elif kind == 'quadratic':
        y = x**2
    elif kind == 'sine':
        y = np.sin(4 * x)
    elif kind == 'step':
        y = np.where(x < (range_min + range_max)/2, 0.0, 1.0)
    elif kind == 'constant':
        y = np.full_like(x, 5.0)
    else:
        y = x.copy()
        
    # Add noise
    if noise > 0:
        y += np.random.normal(0, noise, n)
        
    # Add outliers
    if outlier_ratio > 0:
        n_out = int(n * outlier_ratio)
        indices = np.random.choice(n, n_out, replace=False)
        y[indices] += 10.0 # Significant outlier
        
    return x, y

def main():
    # 1. Tiny Linear
    x, y = generate_data(n=10, kind='linear')
    run_scenario("01_tiny_linear", x, y, frac=0.8, deg=1, iter=0)
    
    # 2. Small Quadratic
    x, y = generate_data(n=50, kind='quadratic')
    run_scenario("02_small_quadratic", x, y, frac=0.5, deg=2, iter=0)
    
    # 3. Sine Standard
    x, y = generate_data(n=100, kind='sine', noise=0.1)
    run_scenario("03_sine_standard", x, y, frac=0.3, deg=1, iter=0)
    
    # 4. Sine Robust
    x, y = generate_data(n=100, kind='sine', outlier_ratio=0.05)
    run_scenario("04_sine_robust", x, y, frac=0.3, deg=1, iter=4)
    
    # 5. Degree 0
    x, y = generate_data(n=100, kind='sine')
    run_scenario("05_degree_0", x, y, frac=0.2, deg=0, iter=0)
    
    # 6. Large scale
    x, y = generate_data(n=500, kind='sine')
    run_scenario("06_large_scale", x, y, frac=0.1, deg=1, iter=0)
    
    # 7. High Smoothness
    x, y = generate_data(n=100, kind='linear', noise=0.5)
    run_scenario("07_high_smoothness", x, y, frac=0.9, deg=1, iter=0)
    
    # 8. Low Smoothness
    x, y = generate_data(n=100, kind='sine')
    run_scenario("08_low_smoothness", x, y, frac=0.05, deg=1, iter=0, surface='direct')
    
    # 9. Quadratic Robust
    x, y = generate_data(n=100, kind='quadratic', outlier_ratio=0.1)
    run_scenario("09_quadratic_robust", x, y, frac=0.5, deg=2, iter=4)
    
    # 10. Constant Function
    x, y = generate_data(n=50, kind='constant')
    run_scenario("10_constant", x, y, frac=0.5, deg=1, iter=0)
    
    # 11. Step Function
    x, y = generate_data(n=100, kind='step')
    run_scenario("11_step_func", x, y, frac=0.4, deg=1, iter=0)
    
    # 12. End-effects Left
    x, y = generate_data(n=50, kind='linear', noise=0.1)
    run_scenario("12_end_effects_left", x, y, frac=0.3, deg=1, iter=0, notes="Check left boundary")
    
    # 13. End-effects Right (same data, just naming)
    run_scenario("13_end_effects_right", x, y, frac=0.3, deg=1, iter=0, notes="Check right boundary")
    
    # 14. Sparse Data
    x, y = generate_data(n=20, range_max=100.0, kind='linear', noise=1.0)
    run_scenario("14_sparse_data", x, y, frac=0.6, deg=1, iter=0)
    
    # 15. Dense Data
    x, y = generate_data(n=1000, kind='sine', noise=0.1)
    run_scenario("15_dense_data", x, y, frac=0.01, deg=1, iter=0, surface='direct')
    
    # 16. Degree 2 Sine
    x, y = generate_data(n=100, kind='sine')
    run_scenario("16_degree_2_sine", x, y, frac=0.4, deg=2, iter=0)
    
    # 17. Robust Degree 0
    x, y = generate_data(n=100, kind='linear', outlier_ratio=0.05)
    run_scenario("17_robust_degree_0", x, y, frac=0.4, deg=0, iter=4)
    
    # 18. Iter 2 Check
    x, y = generate_data(n=100, kind='sine', outlier_ratio=0.05)
    run_scenario("18_iter_2", x, y, frac=0.4, deg=1, iter=2)
    
    # 19. Interpolate Exact
    x, y = generate_data(n=50, kind='linear')
    run_scenario("19_interpolate_exact", x, y, frac=0.5, deg=1, iter=0)
    
    # 20. Zero Variance
    x, y = generate_data(n=10, kind='constant') # all 5.0
    run_scenario("20_zero_variance", x, y, frac=0.5, deg=1, iter=0)

if __name__ == "__main__":
    main()
