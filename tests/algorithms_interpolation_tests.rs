#![cfg(feature = "dev")]
//! Tests for interpolation and delta optimization algorithms.
//!
//! These tests verify the core interpolation utilities used in LOESS for:
//! - Delta calculation for optimization (skipping dense regions)
//!
//! ## Test Organization
//!
//! 1. **Delta Calculation** - Computing optimal delta for interpolation

use approx::assert_relative_eq;

use loess_rs::internals::algorithms::interpolation::calculate_delta;

// ============================================================================
// Delta Calculation Tests
// ============================================================================

/// Test delta calculation with None (automatic) and empty input.
///
/// Verifies:
/// - Empty input returns delta of 0.0
/// - Automatic delta is 1% of x range
#[test]
fn test_calculate_delta_automatic() {
    // Empty x => delta zero
    let empty: Vec<f64> = vec![];
    let delta = calculate_delta::<f64>(None, &empty).unwrap();
    assert_relative_eq!(delta, 0.0, epsilon = 1e-12);

    // Range-based default: 1% of (last - first)
    let xs = vec![1.0f64, 2.0, 5.0];
    let delta = calculate_delta::<f64>(None, &xs).unwrap();
    let expected = 0.01 * (5.0 - 1.0);
    assert_relative_eq!(delta, expected, epsilon = 1e-12);
}

/// Test delta calculation with explicit valid and invalid values.
///
/// Verifies:
/// - Valid provided delta is used as-is
/// - Negative delta produces error
#[test]
fn test_calculate_delta_explicit() {
    let xs = vec![0.0f64, 1.0];

    // Valid provided delta
    let delta = calculate_delta(Some(0.2f64), &xs).unwrap();
    assert_relative_eq!(delta, 0.2, epsilon = 1e-12);
}

/// Test delta calculation with extremely large x range.
#[test]
fn test_calculate_delta_extreme_range() {
    let xs = vec![0.0f64, 1e20];
    let delta = calculate_delta::<f64>(None, &xs).unwrap();
    // Default 1% of 1e20 = 1e18
    assert_relative_eq!(delta, 1e18f64, epsilon = 1e6);
}
