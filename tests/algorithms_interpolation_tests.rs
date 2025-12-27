#![cfg(feature = "dev")]
//! Tests for interpolation algorithms.
//!
//! These tests verify the `InterpolationSurface` and its multilinear interpolation
//! capabilities used for efficient nD LOESS evaluation.
//!
//! ## Test Organization
//!
//! 1. **Surface Construction** - Verifies cell subdivision and vertex creation
//! 2. **Interpolation Accuracy** - Verifies 1D (linear) and 2D (bilinear) interpolation
//! 3. **Adaptive Subdivision** - Verifies finding high-variance regions
//! 4. **Edge Cases** - Boundary conditions and degenerate inputs

use approx::assert_relative_eq;

use loess_rs::internals::algorithms::interpolation::InterpolationSurface;
use loess_rs::internals::engine::executor::LoessDistanceCalculator;
use loess_rs::internals::math::distance::DistanceMetric;
use loess_rs::internals::math::neighborhood::{KDTree, Neighborhood};

// ============================================================================
// Helper Functions & Mocks
// ============================================================================

fn create_mock_dist_calc() -> LoessDistanceCalculator<'static, f64> {
    LoessDistanceCalculator {
        metric: DistanceMetric::Euclidean,
        scales: &[], // Not used for these tests
    }
}

// ============================================================================
// Surface Construction Tests
// ============================================================================

/// Test building a simple 1D surface.
///
/// Verifies that vertices are created at bounds.
#[test]
fn test_build_simple_1d() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let fraction = 0.5;
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    // Simple fitter that just returns the x-coordinate (identity)
    let fitter = |vertex: &[f64], _: &Neighborhood<f64>| -> Option<f64> { Some(vertex[0]) };

    let surface = InterpolationSurface::build(
        &x, &y, dimensions, fraction, &dist_calc, &kdtree, 10, // Max vertices
        fitter, 0.2,
    );

    assert!(surface.vertex_values.len() >= 2);
    // Determine min/max from input to check bounds
    let min_x = 0.0;
    let max_x = 4.0;

    // Bounds are expanded by 0.5%
    let range = 4.0;
    let margin = range * 0.005;
    let effective_min = min_x - margin;
    let effective_max = max_x + margin;

    // Check root cell correctness
    let root = &surface.cells[surface.root];
    assert_relative_eq!(root.lower[0], effective_min, epsilon = 1e-10);
    assert_relative_eq!(root.upper[0], effective_max, epsilon = 1e-10);
}

/// Test building a simple 2D surface.
#[test]
fn test_build_simple_2d() {
    // 4 points in a square
    let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let y = vec![0.0, 1.0, 1.0, 2.0]; // x + y
    let dimensions = 2;
    let fraction = 1.0;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    let fitter =
        |vertex: &[f64], _: &Neighborhood<f64>| -> Option<f64> { Some(vertex[0] + vertex[1]) };

    let surface = InterpolationSurface::build(
        &x, &y, dimensions, fraction, &dist_calc, &kdtree, 20, fitter, 0.2,
    );

    // Initial cell has 4 vertices (2^2)
    assert!(surface.vertex_values.len() >= 4);
}

// ============================================================================
// Interpolation Accuracy Tests
// ============================================================================

/// Test exact 1D linear interpolation.
///
/// Linear interpolation of a linear function should be exact.
#[test]
fn test_interpolate_1d_linear() {
    let x = vec![0.0, 2.0, 4.0];
    let y = vec![0.0, 2.0, 4.0]; // y = x
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    let fitter = |vertex: &[f64], _: &Neighborhood<f64>| -> Option<f64> { Some(vertex[0]) };

    let surface = InterpolationSurface::build(
        &x, &y, dimensions, 0.5, &dist_calc, &kdtree, 10, fitter, 0.2,
    );

    // Test points
    assert_relative_eq!(surface.evaluate(&[1.0]), 1.0, epsilon = 1e-10);
    assert_relative_eq!(surface.evaluate(&[3.0]), 3.0, epsilon = 1e-10);
    assert_relative_eq!(surface.evaluate(&[0.5]), 0.5, epsilon = 1e-10);
}

/// Test exact 2D bilinear interpolation.
///
/// Bilinear interpolation of f(x,y) = ax + by + c should be exact.
#[test]
fn test_interpolate_2d_bilinear() {
    // Grid
    let x = vec![0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0];
    let y: Vec<f64> = x.chunks(2).map(|p| 2.0 * p[0] + 3.0 * p[1] + 1.0).collect();
    let dimensions = 2;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    let fitter = |vertex: &[f64], _: &Neighborhood<f64>| -> Option<f64> {
        Some(2.0 * vertex[0] + 3.0 * vertex[1] + 1.0)
    };

    let surface = InterpolationSurface::build(
        &x, &y, dimensions, 1.0, &dist_calc, &kdtree, 20, fitter, 0.2,
    );

    // Evaluate at center (1, 1) -> 2(1) + 3(1) + 1 = 6
    assert_relative_eq!(surface.evaluate(&[1.0, 1.0]), 6.0, epsilon = 1e-10);

    // Evaluate at (0.5, 1.5) -> 2(0.5) + 3(1.5) + 1 = 1 + 4.5 + 1 = 6.5
    assert_relative_eq!(surface.evaluate(&[0.5, 1.5]), 6.5, epsilon = 1e-10);
}

// ============================================================================
// Adaptive Subdivision Tests
// ============================================================================

/// Test that adaptive subdivision occurs.
///
/// With max_vertices high enough, it should split cells.
#[test]
fn test_adaptive_subdivision() {
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| v * v).collect(); // Nonlinear
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    // Fitter returns x^2
    let fitter =
        |vertex: &[f64], _: &Neighborhood<f64>| -> Option<f64> { Some(vertex[0] * vertex[0]) };

    let surface = InterpolationSurface::build(
        &x, &y, dimensions, 0.3, &dist_calc, &kdtree,
        100, // Allow many vertices to force subdivision
        fitter, 0.2,
    );

    // Should have more than just the initial 2 vertices
    assert!(surface.vertex_values.len() > 2);
    // Should have created child cells
    assert!(surface.cells.len() > 1);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/// Test surface evaluation outside bounds.
///
/// Should clamp to the nearest cell/edge.
#[test]
fn test_interpolate_boundary_clamping() {
    let x = vec![0.0, 2.0];
    let y = vec![0.0, 2.0];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();
    let fitter = |vertex: &[f64], _: &Neighborhood<f64>| -> Option<f64> { Some(vertex[0]) };

    let surface = InterpolationSurface::build(
        &x, &y, dimensions, 0.5, &dist_calc, &kdtree, 10, fitter, 0.2,
    );

    // Far outside right (should be clamped to upper bound value)
    // Upper bound is ~2.01 (0.5% margin)
    // Value should be ~2.01
    let val_far = surface.evaluate(&[10.0]);
    assert!(val_far > 2.0);
    assert!(val_far < 2.1); // Margin check
}

/// Test build handles identical implementation results when fitter fails.
///
/// If fitter returns None, build should fallback to global mean.
#[test]
fn test_fitter_fallback() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![2.0, 2.0, 2.0];
    let dimensions = 1;

    let kdtree = KDTree::new(&x, dimensions);
    let dist_calc = create_mock_dist_calc();

    // Broken fitter always returns None
    let fitter = |_: &[f64], _: &Neighborhood<f64>| -> Option<f64> { None };

    let surface = InterpolationSurface::build(
        &x, &y, dimensions, 0.5, &dist_calc, &kdtree, 10, fitter, 0.2,
    );

    // Should use mean (2.0)
    assert_relative_eq!(surface.evaluate(&[1.0]), 2.0, epsilon = 1e-10);
}
