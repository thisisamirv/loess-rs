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
fn test_compute_ranges() {
    // 3 points, 2D
    let points = [
        1.0, 10.0, // p1
        2.0, 5.0, // p2
        5.0, 20.0, // p3
    ];
    let dims = 2;

    let (mins, maxs) = DistanceMetric::compute_ranges(&points, dims);

    assert_eq!(mins, vec![1.0, 5.0]);
    assert_eq!(maxs, vec![5.0, 20.0]);
}

#[test]
fn test_normalization_scales() {
    let mins = [0.0, 10.0];
    let maxs = [10.0, 20.0];
    // range 1: 10, scale 1/10 = 0.1
    // range 2: 10, scale 1/10 = 0.1

    let scales = DistanceMetric::compute_normalization_scales(&mins, &maxs);
    assert_relative_eq!(scales[0], 0.1);
    assert_relative_eq!(scales[1], 0.1);
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
