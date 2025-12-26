//! Cross-validation for LOESS bandwidth selection.
//!
//! ## Purpose
//!
//! This module provides cross-validation tools for selecting the optimal
//! smoothing fraction (bandwidth) in LOESS regression. It implements
//! generic k-fold and leave-one-out cross-validation strategies.
//!
//! ## Design notes
//!
//! * **Generic Strategy**: Supports both k-fold and leave-one-out (LOOCV).
//! * **Interpolation**: Uses linear interpolation for minimizing prediction error.
//! * **Optimization**: Selects the fraction that minimizes RMSE.
//!
//! ## Key concepts
//!
//! * **K-Fold**: Partitions data into k subsamples (train on k-1, test on 1).
//! * **LOOCV**: Extreme case where k equals sample size (n iterations).
//! * **Interpolation**: Linear interpolation handles test points outside training set.
//!
//! ## Invariants
//!
//! * Training and test sets are disjoint in each fold.
//! * The best fraction minimizes RMSE across all folds.
//! * Interpolation uses constant extrapolation at boundaries.
//!
//! ## Non-goals
//!
//! * This module does not perform the actual smoothing (done via callback).
//! * This module does not provide confidence intervals for CV scores.

// Feature-gated dependencies
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering;
use num_traits::Float;

// ============================================================================
// Internal PRNG
// ============================================================================

/// Minimal PRNG for no-std shuffling.
///
/// Uses an LCG (Linear Congruential Generator) with constants from PCG/MQL.
#[derive(Debug, Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        // LCG constants for 64-bit state
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }
}

// ============================================================================
// Internal CV Kind (for storage)
// ============================================================================

/// Internal representation of CV method for storage (no lifetime needed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CVKind {
    /// K-fold cross-validation with k folds.
    KFold(usize),
    /// Leave-one-out cross-validation.
    #[allow(clippy::upper_case_acronyms)]
    LOOCV,
}

// ============================================================================
// Cross-Validation Configuration
// ============================================================================

/// Cross-validation configuration combining strategy, fractions, and seed.
#[derive(Debug, Clone)]
pub struct CVConfig<'a, T> {
    /// The CV strategy kind.
    pub(crate) kind: CVKind,
    /// Candidate smoothing fractions to evaluate.
    pub(crate) fractions: &'a [T],
    /// Random seed for reproducible fold shuffling (K-Fold only).
    pub(crate) seed: Option<u64>,
}

impl<'a, T> CVConfig<'a, T> {
    /// Set the random seed for reproducible K-Fold cross-validation.
    ///
    /// The seed controls shuffling of data indices before fold assignment.
    /// Using the same seed produces identical fold assignments across runs.
    ///
    /// # Note
    ///
    /// This only affects K-Fold CV. LOOCV is deterministic and ignores the seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get the fractions slice.
    pub fn fractions(&self) -> &[T] {
        self.fractions
    }

    /// Get the CV kind for internal use.
    pub(crate) fn kind(&self) -> CVKind {
        self.kind
    }

    /// Get the seed for internal use.
    pub(crate) fn get_seed(&self) -> Option<u64> {
        self.seed
    }
}

/// Create a K-fold cross-validation configuration.
#[allow(non_snake_case)]
pub fn KFold<T>(k: usize, fractions: &[T]) -> CVConfig<'_, T> {
    CVConfig {
        kind: CVKind::KFold(k),
        fractions,
        seed: None,
    }
}

/// Create a leave-one-out cross-validation configuration.
#[allow(non_snake_case)]
pub fn LOOCV<T>(fractions: &[T]) -> CVConfig<'_, T> {
    CVConfig {
        kind: CVKind::LOOCV,
        fractions,
        seed: None,
    }
}

// ============================================================================
// Cross-Validation Execution
// ============================================================================

impl CVKind {
    // ========================================================================
    // Public API
    // ========================================================================

    /// Run cross-validation to select the best fraction.
    pub fn run<T, F>(
        self,
        x: &[T],
        y: &[T],
        fractions: &[T],
        seed: Option<u64>,
        smoother: F,
    ) -> (T, Vec<T>)
    where
        T: Float,
        F: Fn(&[T], &[T], T) -> Vec<T> + Copy,
    {
        match self {
            CVKind::KFold(k) => Self::kfold_cross_validation(x, y, fractions, k, seed, smoother),
            CVKind::LOOCV => Self::leave_one_out_cross_validation(x, y, fractions, smoother),
        }
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /// Build a data subset from a list of indices into provided scratch buffers.
    pub fn build_subset_inplace<T: Float>(
        x: &[T],
        y: &[T],
        indices: &[usize],
        tx: &mut Vec<T>,
        ty: &mut Vec<T>,
    ) {
        tx.clear();
        ty.clear();
        for &i in indices {
            tx.push(x[i]);
            ty.push(y[i]);
        }
    }

    /// Build a data subset from a list of indices.
    pub fn build_subset_from_indices<T: Float>(
        x: &[T],
        y: &[T],
        indices: &[usize],
    ) -> (Vec<T>, Vec<T>) {
        let mut tx = Vec::with_capacity(indices.len());
        let mut ty = Vec::with_capacity(indices.len());
        Self::build_subset_inplace(x, y, indices, &mut tx, &mut ty);
        (tx, ty)
    }

    /// Interpolate prediction at a new x value given fitted training values.
    ///
    /// # Implementation notes
    ///
    /// * Uses binary search for O(log n) bracketing
    /// * Handles duplicate x-values by averaging y-values
    /// * Constant extrapolation prevents unbounded predictions
    pub fn interpolate_prediction<T: Float>(x_train: &[T], y_train: &[T], x_new: T) -> T {
        let n = x_train.len();

        // Edge case: empty training set
        if n == 0 {
            return T::zero();
        }

        // Edge case: single training point
        if n == 1 {
            return y_train[0];
        }

        // Boundary handling: constant extrapolation
        if x_new <= x_train[0] {
            return y_train[0];
        }
        if x_new >= x_train[n - 1] {
            return y_train[n - 1];
        }

        // Binary search for bracketing points
        let mut left = 0;
        let mut right = n - 1;

        while right - left > 1 {
            let mid = (left + right) / 2;
            if x_train[mid] <= x_new {
                left = mid;
            } else {
                right = mid;
            }
        }

        // Linear interpolation between left and right
        let x0 = x_train[left];
        let x1 = x_train[right];
        let y0 = y_train[left];
        let y1 = y_train[right];

        let denom = x1 - x0;
        if denom <= T::zero() {
            // Duplicate x-values: return average of y-values
            return (y0 + y1) / T::from(2.0).unwrap();
        }

        let alpha = (x_new - x0) / denom;
        y0 + alpha * (y1 - y0)
    }

    /// Predict values at multiple new x points using linear interpolation.
    ///
    /// # Implementation notes
    ///
    /// * Leverages sorted order of `x_new` for O(n_train + n_new) linear scan.
    /// * Falls back to repeated binary search if `x_new` is not monotonic (not recommended).
    pub fn interpolate_prediction_batch<T: Float>(
        x_train: &[T],
        y_train: &[T],
        x_new: &[T],
        y_pred: &mut [T],
    ) {
        let n_train = x_train.len();
        let n_new = x_new.len();

        if n_new == 0 {
            return;
        }

        if n_train == 0 {
            y_pred.fill(T::zero());
            return;
        }

        if n_train == 1 {
            y_pred.fill(y_train[0]);
            return;
        }

        let mut left = 0;
        for i in 0..n_new {
            let xi = x_new[i];

            // Boundary handling: constant extrapolation
            if xi <= x_train[0] {
                y_pred[i] = y_train[0];
                continue;
            }
            if xi >= x_train[n_train - 1] {
                y_pred[i] = y_train[n_train - 1];
                continue;
            }

            // Linear scan forward to find bracket
            while left + 1 < n_train && x_train[left + 1] <= xi {
                left += 1;
            }

            let right = left + 1;
            let x0 = x_train[left];
            let x1 = x_train[right];
            let y0 = y_train[left];
            let y1 = y_train[right];

            let denom = x1 - x0;
            if denom <= T::zero() {
                y_pred[i] = (y0 + y1) / T::from(2.0).unwrap();
            } else {
                let alpha = (xi - x0) / denom;
                y_pred[i] = y0 + alpha * (y1 - y0);
            }
        }
    }

    // ========================================================================
    // Internal Cross-Validation Implementations
    // ========================================================================

    /// Select the best fraction based on cross-validation scores.
    fn select_best_fraction<T: Float>(fractions: &[T], scores: &[T]) -> (T, Vec<T>) {
        if fractions.is_empty() {
            return (T::zero(), Vec::new());
        }

        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        (fractions[best_idx], scores.to_vec())
    }

    /// Perform k-fold cross-validation.
    fn kfold_cross_validation<T, F>(
        x: &[T],
        y: &[T],
        fractions: &[T],
        k: usize,
        seed: Option<u64>,
        smoother: F,
    ) -> (T, Vec<T>)
    where
        T: Float,
        F: Fn(&[T], &[T], T) -> Vec<T>,
    {
        let n = x.len();
        if n < k || k < 2 {
            return (
                fractions.first().copied().unwrap_or(T::zero()),
                vec![T::zero(); fractions.len()],
            );
        }

        let fold_size = n / k;
        let mut cv_scores = vec![T::zero(); fractions.len()];

        // Generate indices and optionally shuffle if seed is provided
        let mut indices: Vec<usize> = (0..n).collect();
        if let Some(s) = seed {
            let mut rng = SimpleRng::new(s);
            for i in (1..n).rev() {
                let j = (rng.next_u32() as usize) % (i + 1);
                indices.swap(i, j);
            }
        }

        // Pre-allocate scratch buffers for training set
        let mut train_x = Vec::with_capacity(n);
        let mut train_y = Vec::with_capacity(n);

        for (frac_idx, &frac) in fractions.iter().enumerate() {
            // Store RMSE for each fold, then compute mean
            let mut fold_rmses = Vec::with_capacity(k);

            for fold in 0..k {
                // Define test set for this fold
                let test_start = fold * fold_size;
                let test_end = if fold == k - 1 {
                    n // Last fold includes remainder
                } else {
                    (fold + 1) * fold_size
                };

                // Build training set using (potentially shuffled) indices
                train_x.clear();
                train_y.clear();

                for &idx in &indices[0..test_start] {
                    train_x.push(x[idx]);
                    train_y.push(y[idx]);
                }
                for &idx in &indices[test_end..n] {
                    train_x.push(x[idx]);
                    train_y.push(y[idx]);
                }

                // Training data MUST be sorted for LOESS
                // We need to sort by x and permute y accordingly
                let mut train_data: Vec<(T, T)> = train_x
                    .iter()
                    .zip(train_y.iter())
                    .map(|(&xi, &yi)| (xi, yi))
                    .collect();
                train_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

                let (sorted_tx, sorted_ty): (Vec<T>, Vec<T>) = train_data.into_iter().unzip();

                // Fit smoother on training data
                let train_smooth = smoother(&sorted_tx, &sorted_ty, frac);

                // Compute RMSE on test set using batched interpolation
                // Test points from original buffers (must be collected from indices too)
                let mut test_x = Vec::with_capacity(test_end - test_start);
                let mut test_y = Vec::with_capacity(test_end - test_start);
                for &idx in &indices[test_start..test_end] {
                    test_x.push(x[idx]);
                    test_y.push(y[idx]);
                }

                let mut predictions = vec![T::zero(); test_x.len()];
                Self::interpolate_prediction_batch(
                    &sorted_tx,
                    &train_smooth,
                    &test_x,
                    &mut predictions,
                );

                let mut fold_error = T::zero();
                let mut fold_count = T::zero();
                for (i, &predicted) in predictions.iter().enumerate() {
                    let actual = test_y[i];
                    let error = actual - predicted;
                    fold_error = fold_error + error * error;
                    fold_count = fold_count + T::one();
                }

                // Compute RMSE for this fold
                if fold_count > T::zero() {
                    fold_rmses.push((fold_error / fold_count).sqrt());
                }
            }

            // Compute mean of fold RMSEs (matches sklearn's cross_val_score)
            if !fold_rmses.is_empty() {
                let sum: T = fold_rmses.iter().copied().fold(T::zero(), |a, b| a + b);
                cv_scores[frac_idx] = sum / T::from(fold_rmses.len()).unwrap();
            } else {
                cv_scores[frac_idx] = T::infinity();
            }
        }

        Self::select_best_fraction(fractions, &cv_scores)
    }

    /// Perform leave-one-out cross-validation (LOOCV).
    fn leave_one_out_cross_validation<T, F>(
        x: &[T],
        y: &[T],
        fractions: &[T],
        smoother: F,
    ) -> (T, Vec<T>)
    where
        T: Float,
        F: Fn(&[T], &[T], T) -> Vec<T>,
    {
        let n = x.len();
        let mut cv_scores = vec![T::zero(); fractions.len()];

        // Pre-allocate scratch buffers for training set
        let mut train_x = Vec::with_capacity(n - 1);
        let mut train_y = Vec::with_capacity(n - 1);

        for (frac_idx, &frac) in fractions.iter().enumerate() {
            let mut total_error = T::zero();

            for i in 0..n {
                // Build training set (all points except i) using scratch buffers
                train_x.clear();
                train_y.clear();
                for j in 0..i {
                    train_x.push(x[j]);
                    train_y.push(y[j]);
                }
                for j in (i + 1)..n {
                    train_x.push(x[j]);
                    train_y.push(y[j]);
                }

                // Fit smoother on training data
                let train_smooth = smoother(&train_x, &train_y, frac);

                // Predict at held-out point
                let predicted = Self::interpolate_prediction(&train_x, &train_smooth, x[i]);
                let error = y[i] - predicted;
                total_error = total_error + error * error;
            }

            // Compute RMSE for this fraction
            cv_scores[frac_idx] = (total_error / T::from(n).unwrap()).sqrt();
        }

        Self::select_best_fraction(fractions, &cv_scores)
    }
}
