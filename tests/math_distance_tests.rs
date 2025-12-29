#![cfg(feature = "dev")]

use approx::assert_relative_eq;
use loess_rs::internals::math::distance::DistanceMetric;

// ============================================================================
// Euclidean Distance Tests
// ============================================================================

#[test]
fn test_euclidean_distance_1d() {
    let a = [1.0];
    let b = [4.0];
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 3.0);
}

#[test]
fn test_euclidean_distance_2d() {
    let a = [0.0, 0.0];
    let b = [3.0, 4.0];
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 5.0);
}

#[test]
fn test_euclidean_distance_3d() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 8.0];
    // diffs: 3, 4, 5. sum_sq: 9+16+25=50. sqrt(50) approx 7.071
    let dist = DistanceMetric::euclidean(&a, &b);
    assert_relative_eq!(dist, 50.0f64.sqrt());
}

// ============================================================================
// Normalization Tests
// ============================================================================

#[test]
fn test_robust_scales_simple() {
    // 3 points, 2D
    // n < 10, so fallback to 1.0
    let points = [1.0, 10.0, 2.0, 5.0, 5.0, 20.0];
    let dims = 2;

    let scales = DistanceMetric::compute_robust_scales(&points, dims);
    assert_eq!(scales, vec![1.0, 1.0]);
}

#[test]
fn test_robust_scales_outliers() {
    // 20 points, 1D
    // 18 points in [0, 1], 2 outliers at 1000.0
    // 10% trim = ceil(0.1 * 20) = 2 points.
    // robust logic should trim the top 2 (outliers) and bottom 2.
    // remaining 16 points are in [0, 1].
    // std dev should be small (~0.3), scale should be large (~3.0).
    // if it used min-max (0 to 1000), scale would be 0.001.

    let mut points = Vec::new();
    for i in 0..18 {
        points.push(i as f64 / 18.0);
    }
    points.push(1000.0);
    points.push(1000.0);

    let dims = 1;
    let scales = DistanceMetric::compute_robust_scales(&points, dims);

    assert!(scales[0] > 1.0); // Should be much larger than 0.001
}

#[test]
fn test_normalized_distance() {
    let a = [0.0, 10.0];
    let b = [10.0, 20.0];
    let scales = [0.1, 0.05]; // range x: 10, range y: 20

    // diffs: 10, 10
    // scaled: 1.0, 0.5
    // sum_sq: 1.0 + 0.25 = 1.25
    // sqrt(1.25) approx 1.118

    let dist = DistanceMetric::normalized(&a, &b, &scales);
    assert_relative_eq!(dist, 1.25f64.sqrt());
}

#[test]
fn test_manhattan_distance() {
    let a = [1.0, 2.0];
    let b = [4.0, 6.0];
    // |1-4| + |2-6| = 3 + 4 = 7
    let dist = DistanceMetric::manhattan(&a, &b);
    assert_relative_eq!(dist, 7.0);
}

#[test]
fn test_chebyshev_distance() {
    let a = [1.0, 2.0];
    let b = [4.0, 7.0];
    // |1-4|=3, |2-7|=5. max(3, 5) = 5
    let dist = DistanceMetric::chebyshev(&a, &b);
    assert_relative_eq!(dist, 5.0);
}

#[test]
fn test_minkowski_distance() {
    let a = [1.0, 2.0];
    let b = [4.0, 6.0];
    let p = 3.0;
    // |3|^3 + |4|^3 = 27 + 64 = 91. 91^(1/3) approx 4.4979
    let dist: f64 = DistanceMetric::minkowski(&a, &b, p);
    assert_relative_eq!(dist, 91.0f64.powf(1.0 / 3.0));
}

#[test]
fn test_weighted_distance() {
    let a = [1.0, 2.0];
    let b = [2.0, 3.0];
    let weights = [4.0, 1.0]; // Weight X more heavily

    // diffs: 1, 1
    // weighted sq: 4*(1)^2 + 1*(1)^2 = 5
    // dist = sqrt(5) approx 2.236

    let dist = DistanceMetric::weighted(&a, &b, &weights);
    assert_relative_eq!(dist, 5.0f64.sqrt());
}
