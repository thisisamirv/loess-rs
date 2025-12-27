import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import json
from pathlib import Path

class LowessBuilder:
    def __init__(self):
        self._fraction = 0.67
        self._iterations = 3
        self._delta = None
        self._weight_function = 'tricube'  # Fixed, but for compatibility
        self._robustness_method = 'bisquare'  # Fixed
        self.interval_level = None
        self.interval_type = None
        self.cv_fractions = None
        self.cv_method = None
        self.k = 5
        self.auto_convergence = None
        self._max_iterations = 20
        self.compute_diagnostics = False
        self.compute_residuals = False
        self.compute_robustness_weights = False
        self._zero_weight_fallback = 'use_local_mean'  # Not used

    def fraction(self, fraction):
        self._fraction = fraction
        return self

    def iterations(self, iterations):
        self._iterations = iterations
        return self

    def delta(self, delta):
        self._delta = delta
        return self

    def weight_function(self, weight_function):
        # Fixed to tricube, ignore others
        self._weight_function = weight_function
        return self

    def robustness_method(self, robustness_method):
        # Fixed to bisquare, ignore others
        self._robustness_method = robustness_method
        return self

    def with_confidence_intervals(self, level):
        # Not supported, set for compatibility
        self.interval_level = level
        self.interval_type = 'confidence'
        return self

    def with_prediction_intervals(self, level):
        self.interval_level = level
        self.interval_type = 'prediction'
        return self

    def with_both_intervals(self, level):
        self.interval_level = level
        self.interval_type = 'both'
        return self

    def cross_validate(self, fractions):
        self.cv_fractions = fractions
        self.cv_method = 'simple'
        return self

    def cross_validate_kfold(self, fractions, k):
        self.cv_fractions = fractions
        self.cv_method = 'kfold'
        self.k = k
        return self

    def cross_validate_loocv(self, fractions):
        self.cv_fractions = fractions
        self.cv_method = 'loocv'
        return self

    def auto_converge(self, tolerance):
        self.auto_convergence = tolerance
        return self

    def max_iterations(self, max_iter):
        self._max_iterations = max_iter
        return self

    def with_diagnostics(self):
        self.compute_diagnostics = True
        return self

    def with_residuals(self):
        self.compute_residuals = True
        return self

    def with_robustness_weights(self):
        self.compute_robustness_weights = True
        return self

    def with_all_diagnostics(self):
        self.compute_diagnostics = True
        self.compute_residuals = True
        self.compute_robustness_weights = True
        return self

    def zero_weight_fallback(self, policy):
        self._zero_weight_fallback = policy
        return self

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        n = len(x)
        idx = np.argsort(x)
        x_sorted = x[idx]
        y_sorted = y[idx]
        delta = self._delta if self._delta is not None else 0.0
        fraction = self._fraction
        iterations = self._iterations
        cv_scores = None

        if self.cv_fractions is not None:
            fracs = self.cv_fractions
            scores = np.zeros(len(fracs))
            cv_method = self.cv_method
            for f_idx, f in enumerate(fracs):
                if cv_method == 'simple':
                    res = lowess(y_sorted, x_sorted, frac=f, it=self._iterations, delta=delta)
                    y_smooth = res[:, 1]
                    scores[f_idx] = np.sqrt(np.mean((y_sorted - y_smooth)**2))
                elif cv_method == 'kfold':
                    k = self.k
                    fold_size = n // k
                    fold_scores = np.zeros(k)
                    for ff in range(k):
                        start = ff * fold_size
                        end = start + fold_size if ff < k - 1 else n
                        train_idx = np.concatenate((np.arange(0, start), np.arange(end, n)))
                        test_idx = np.arange(start, end)
                        x_train = x_sorted[train_idx]
                        y_train = y_sorted[train_idx]
                        x_test = x_sorted[test_idx]
                        y_test = y_sorted[test_idx]
                        res = lowess(y_train, x_train, frac=f, it=self._iterations, delta=delta)
                        # Interpolate or predict at x_test
                        y_pred = np.interp(x_test, res[:, 0], res[:, 1])
                        fold_scores[ff] = np.sqrt(np.mean((y_test - y_pred)**2))
                    scores[f_idx] = np.mean(fold_scores)
                elif cv_method == 'loocv':
                    loo_score = 0
                    for ff in range(n):
                        train_idx = np.delete(np.arange(n), ff)
                        x_train = x_sorted[train_idx]
                        y_train = y_sorted[train_idx]
                        x_test = x_sorted[ff:ff+1]
                        y_test = y_sorted[ff]
                        res = lowess(y_train, x_train, frac=f, it=self._iterations, delta=delta)
                        y_pred = np.interp(x_test, res[:, 0], res[:, 1])
                        loo_score += (y_test - y_pred[0])**2
                    scores[f_idx] = np.sqrt(loo_score / n)
            best_idx = np.argmin(scores)
            fraction = fracs[best_idx]
            cv_scores = scores

        if self.auto_convergence is not None:
            tol = self.auto_convergence
            max_it = self._max_iterations
            res = lowess(y_sorted, x_sorted, frac=fraction, it=0, delta=delta)
            y_smooth = res[:, 1]
            iterations_used = 0
            while iterations_used < max_it:
                iterations_used += 1
                res_old = y_smooth.copy()
                res = lowess(y_sorted, x_sorted, frac=fraction, it=iterations_used, delta=delta)
                y_smooth = res[:, 1]
                if np.max(np.abs(y_smooth - res_old)) < tol:
                    break
        else:
            iterations_used = None
            res = lowess(y_sorted, x_sorted, frac=fraction, it=iterations, delta=delta)
            y_smooth = res[:, 1]

        residuals = y_sorted - y_smooth
        median_res = np.median(residuals)
        residual_sd = np.median(np.abs(residuals - median_res)) * 1.4826 if len(residuals) > 0 else 0

        diagnostics = None
        if self.compute_diagnostics:
            rmse = np.sqrt(np.mean(residuals**2)) if len(residuals) > 0 else 0
            mae = np.mean(np.abs(residuals)) if len(residuals) > 0 else 0
            r_squared = 1 - np.var(residuals) / np.var(y_sorted) if len(residuals) > 0 and np.var(y_sorted) > 0 else 0
            aic = None  # Not computed
            aicc = None
            effective_df = None
            diagnostics = {
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared,
                'aic': aic,
                'aicc': aicc,
                'effective_df': effective_df,
                'residual_sd': residual_sd
            }

        robustness_weights = None
        if self.compute_robustness_weights and iterations > 0:
            median_res = np.median(residuals)
            mad = np.median(np.abs(residuals - median_res))
            if mad == 0:
                robustness_weights = np.ones(len(residuals))
            else:
                scaled = np.abs(residuals) / (6 * mad)
                robustness_weights = np.where(scaled < 1, (1 - scaled**2)**2, 0)

        result = {
            'x': x_sorted.tolist(),
            'y': y_smooth.tolist(),
            'residuals': residuals.tolist() if self.compute_residuals else None,
            'robustness_weights': robustness_weights.tolist() if robustness_weights is not None else None,
            'diagnostics': diagnostics,
            'iterations_used': iterations_used,
            'fraction_used': fraction,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None
        }
        return result

# Main code

np.random.seed(42)

n = 100
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + 0.5 * np.random.randn(n)
outlier_idx = np.random.choice(n, 10, replace=False)
y[outlier_idx] += 10 * np.random.randn(10)

scenarios = [
    ('basic', LowessBuilder()),
    ('small_fraction', LowessBuilder().fraction(0.2)),
    ('no_robust', LowessBuilder().iterations(0)),
    ('more_robust', LowessBuilder().iterations(5)),
    ('auto_converge', LowessBuilder().auto_converge(1e-4)),
    ('cross_validate', LowessBuilder().cross_validate([0.2, 0.4, 0.6])),
    ('kfold_cv', LowessBuilder().cross_validate_kfold([0.2, 0.4, 0.6], 5)),
    ('loocv', LowessBuilder().cross_validate_loocv([0.2, 0.4, 0.6])),
    ('delta_zero', LowessBuilder().delta(0)),
    ('with_all_diagnostics', LowessBuilder().with_all_diagnostics()),
]

results = {}
for name, builder in scenarios:
    result = builder.fit(x, y)
    results[name] = result

# save results to a JSON file
script_dir = Path(__file__).resolve().parent
validation_dir = script_dir.parent

out_dir = validation_dir / "output"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "statsmodels_validate.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {out_path}")
