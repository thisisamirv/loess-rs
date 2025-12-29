//! Workspace for reusable LOESS buffers.
//!
//! This module provides a pre-allocated workspace to minimize dynamic memory
//! allocations during local regression fitting and neighborhood searches.

use core::fmt::Debug;
use num_traits::Float;

use crate::algorithms::regression::FittingBuffer;
use crate::math::neighborhood::{Neighborhood, NeighborhoodSearchBuffer};

/// A workspace containing pre-allocated buffers for LOESS operations.
///
/// Reusing a workspace across multiple smoothing points significantly reduces
/// allocation overhead in both point-wise and interpolation modes.
pub struct LoessWorkspace<T: Float> {
    /// Buffer for KD-tree search state.
    pub search_buffer: NeighborhoodSearchBuffer<T>,
    /// Buffer for neighbor indices and distances.
    pub neighborhood: Neighborhood<T>,
    /// Buffer for regression fitting (WLS matrices).
    pub fitting_buffer: FittingBuffer<T>,
}

impl<T: Float + Debug + Send + Sync + 'static> LoessWorkspace<T> {
    /// Create a new workspace with capacities matching the expected fit parameters.
    ///
    /// - `k`: Expected number of neighbors (window size).
    /// - `n_coeffs`: Expected number of polynomial coefficients.
    pub fn new(k: usize, n_coeffs: usize) -> Self {
        Self {
            search_buffer: NeighborhoodSearchBuffer::new(k),
            neighborhood: Neighborhood::with_capacity(k),
            fitting_buffer: FittingBuffer::new(k, n_coeffs),
        }
    }

    /// Clear workspace buffers for the next fitting point.
    ///
    /// Note: Does not deallocate, only resets logical lengths.
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.search_buffer.clear();
        self.neighborhood.indices.clear();
        self.neighborhood.distances.clear();
        // FittingBuffer fields are cleared internally by RegressionContext when used.
    }
}
