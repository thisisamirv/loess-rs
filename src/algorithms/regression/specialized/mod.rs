//! Specialized Solvers
//!
//! ## Purpose
//!
//! This module provides high-performance, specialized solvers and accumulation
//! logic for common low-dimensional cases (1D and 2D linear/quadratic regression).

// External dependencies
use num_traits::Float;

/// Accumulators for specialized solvers.
pub mod accumulators;

/// Implementations of specialized solvers.
pub mod impls;

// ============================================================================
// Specialized Solvers Trait
// ============================================================================

/// Trait for specialized low-dimensional linear solvers.
/// Extracted from FloatLinalg to keep specialized logic near regression code.
pub trait SolverLinalg: Float + 'static {
    /// Solve 2x2 system of linear equations Ax = b.
    fn solve_2x2(a: [Self; 4], b: [Self; 2]) -> Option<[Self; 2]>;

    /// Solve 3x3 system of linear equations Ax = b.
    fn solve_3x3(a: [Self; 9], b: [Self; 3]) -> Option<[Self; 3]>;

    /// Solve 6x6 system of linear equations Ax = b.
    fn solve_6x6(a: [Self; 36], b: [Self; 6]) -> Option<[Self; 6]>;

    /// Accumulate Normal Equations for 1D Linear Case.
    fn accumulate_1d_linear(
        x: &[Self],
        y: &[Self],
        indices: &[usize],
        weights: &[Self],
        query: Self,
        xtwx: &mut [Self; 4],
        xtwy: &mut [Self; 2],
    );

    /// Accumulate Normal Equations for 2D Linear Case.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_2d_linear(
        x: &[Self],
        y: &[Self],
        indices: &[usize],
        weights: &[Self],
        query_x: Self,
        query_y: Self,
        xtwx: &mut [Self; 9],
        xtwy: &mut [Self; 3],
    );

    /// Accumulate Normal Equations for 2D Quadratic Case.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_2d_quadratic(
        x: &[Self],
        y: &[Self],
        indices: &[usize],
        weights: &[Self],
        query_x: Self,
        query_y: Self,
        xtwx: &mut [Self; 36],
        xtwy: &mut [Self; 6],
    );
}
