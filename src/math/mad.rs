//! Median Absolute Deviation (MAD) computation for robust scale estimation.
//!
//! ## Purpose
//!
//! This module provides robust scale estimation using the Median Absolute
//! Deviation (MAD), which is resistant to outliers.
//!
//! ## Design notes
//!
//! * **Algorithm**: Uses Quickselect for O(n) median finding.
//! * **Memory**: Reuses allocated buffers to minimize memory allocations.
//! * **Formula**: MAD = median(|r_i - median(r)|).
//!
//! ## Key concepts
//!
//! * **Robustness**: 50% breakdown point (safe against 50% outliers).
//! * **Efficiency**: Avoids full sorting (O(n log n)) in favor of selection (O(n)).
//!
//! ## Invariants
//!
//! * MAD >= 0 for any input.
//! * Handles even and odd population sizes correctly.
//!
//! ## Non-goals
//!
//! * This module does not provide weighted MAD variants.
//! * This module does not handle non-finite values (NaN/Inf).

// External dependencies
use core::cmp::Ordering::Equal;
use num_traits::Float;

// ============================================================================
// MAD Computation
// ============================================================================

/// Compute the Median Absolute Deviation (MAD) in-place, avoiding extra allocations.
///
/// # Formula
///
/// Calculated as:
/// ```text
/// MAD = median(|r_i - median(r)|)
/// ```
///
/// # Safety
///
/// This function modifies the provided `vals` slice.
#[inline]
pub fn compute_mad<T: Float>(vals: &mut [T]) -> T {
    if vals.is_empty() {
        return T::zero();
    }

    // Step 1: Compute median of residuals
    let median: T = median_inplace(vals);

    // Step 2: Compute absolute deviations from median
    for val in vals.iter_mut() {
        *val = (*val - median).abs();
    }

    // Step 3: Return median of absolute deviations
    median_inplace(vals)
}

/// Internal helper function to compute median in-place using Quickselect.
#[inline]
fn median_inplace<T: Float>(vals: &mut [T]) -> T {
    let n = vals.len();
    if n == 0 {
        return T::zero();
    }

    let mid = n / 2;

    if n % 2 == 0 {
        // Even length: average of two middle values
        vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Equal));
        let upper = vals[mid];

        // Find the largest value in the lower half
        let lower = vals[..mid].iter().copied().fold(T::neg_infinity(), T::max);

        (lower + upper) / T::from(2.0).unwrap()
    } else {
        // Odd length: middle value
        vals.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Equal));
        vals[mid]
    }
}
