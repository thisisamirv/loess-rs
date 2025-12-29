//! Memory management and buffer recycling for LOESS operations.
//!
//! ## Purpose
//!
//! This module provides a centralized, reusable workspace (`LoessBuffer`) to minimize
//! dynamic memory allocations during repeated LOESS fitting. By allocating buffers once
//! and recycling them across multiple query points or cross-validation folds, we significantly
//! reduce allocator pressure and improve cache locality.
//!
//! ## Design notes
//!
//! * **Centralized Ownership**: The `LoessBuffer` struct holds all necessary scratch space
//!   for the entire pipeline (neighborhood search, regression weights, solver matrices, etc.).
//! * **Lazy Expansion**: Buffers are grown on demand via `ensure_capacity` but never shrunk,
//!   stabilizing at the maximum required size for the dataset.
//! * **Generic injection**: The workspace is generic over the `Neighborhood` storage to
//!   allow decoupling from specific spatial index implementations.
//!
//! ## Key concepts
//!
//! * **LoessBuffer**: The top-level struct passed through the executor pipeline.
//! * **NeighborhoodSearchBuffer**: Reusable heap and vector for K-nearest neighbor searches.
//! * **FittingBuffer**: Matrices and vectors for the weighted least squares solver.
//! * **ExecutorBuffer**: Buffers for global operations like robustness iteration and normalization.
//! * **CVBuffer**: Scratch space specifically for cross-validation data subsets.
//!
//! ## Invariants
//!
//! * Buffers are only logically cleared (e.g., `vec.clear()`), not deallocated, between iterations.
//! * Capability is monotonically increasing; `ensure_capacity` only reallocates if current capacity is insufficient.
//!
//! ## Non-goals
//!
//! * Thread-local automatic caching (buffers are explicitly passed to allow parallel execution with one buffer per thread).
//! * Dynamic shrinking or aggressive memory reclamation (performance is prioritized over minimal footprint).

use core::fmt::Debug;
use num_traits::Float;

#[cfg(not(feature = "std"))]
use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::collections::BinaryHeap;
#[cfg(feature = "std")]
use std::vec::Vec;

// ============================================================================
// Main Workspace
// ============================================================================

/// A struct containing pre-allocated buffers for LOESS operations.
pub struct LoessBuffer<T: Float, N, NH> {
    /// Buffer for KD-tree search state.
    pub search_buffer: NeighborhoodSearchBuffer<N>,
    /// Buffer for neighbor indices and distances.
    pub neighborhood: NH,
    /// Buffer for regression fitting (WLS matrices).
    pub fitting_buffer: FittingBuffer<T>,
    /// Buffer for global executor state.
    pub executor_buffer: ExecutorBuffer<T>,
    /// Buffer for cross-validation.
    pub cv_buffer: CVBuffer<T>,
}

impl<T: Float + Debug + Send + Sync + 'static, N, NH> LoessBuffer<T, N, NH>
where
    N: Ord,
    NH: NeighborhoodStorage,
{
    /// Create a new workspace with capacities matching the expected fit parameters.
    ///
    /// - `n`: Total number of points.
    /// - `dims`: Predictor dimensions.
    /// - `k`: Expected number of neighbors (window size).
    /// - `n_coeffs`: Expected number of polynomial coefficients.
    pub fn new(n: usize, dims: usize, k: usize, n_coeffs: usize) -> Self {
        Self {
            search_buffer: NeighborhoodSearchBuffer::new(k),
            neighborhood: NH::with_capacity(k),
            fitting_buffer: FittingBuffer::<T>::new(k, n_coeffs),
            executor_buffer: ExecutorBuffer::<T>::new(n, dims),
            cv_buffer: CVBuffer::<T>::new(n, dims),
        }
    }

    /// Ensure all buffers have enough capacity for the given problem size.
    pub fn ensure_capacity(&mut self, n_total: usize, dims: usize, k: usize, n_coeffs: usize) {
        if self.neighborhood.capacity() < k {
            self.neighborhood = NH::with_capacity(k);
            self.search_buffer = NeighborhoodSearchBuffer::new(k);
        }
        if self.fitting_buffer.weights.capacity() < k
            || self.fitting_buffer.xtw_x.capacity() < n_coeffs * n_coeffs
        {
            self.fitting_buffer = FittingBuffer::new(k, n_coeffs);
        }

        self.executor_buffer.ensure_capacity(n_total, dims);
        self.cv_buffer.ensure_capacity(n_total, dims);
    }
}

// ============================================================================
// Traits
// ============================================================================

/// Trait for neighborhood storage that can be injected into the workspace.
pub trait NeighborhoodStorage {
    /// Create a new neighborhood storage with given capacity.
    fn with_capacity(k: usize) -> Self;
    /// Get the current capacity of the storage.
    fn capacity(&self) -> usize;
}

// ============================================================================
// Internal Buffers
// ============================================================================

/// Persistent buffers for KD-tree search to avoid allocations.
pub struct NeighborhoodSearchBuffer<N> {
    pub(crate) heap: BinaryHeap<N>,
    pub(crate) sort_vec: Vec<N>,
}

impl<N: Ord> NeighborhoodSearchBuffer<N> {
    /// Create a new search buffer with capacity k.
    pub fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k),
            sort_vec: Vec::with_capacity(k),
        }
    }

    /// Clear all internal buffers for reuse.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.sort_vec.clear();
    }
}

/// Persistent buffers for local regression to avoid allocations.
pub struct FittingBuffer<T: Float> {
    /// Weights for each neighbor.
    pub weights: Vec<T>,
    /// Normal matrix X'WX.
    pub xtw_x: Vec<T>,
    /// Normal vector X'WY.
    pub xtw_y: Vec<T>,
    /// Column norms for equilibration.
    pub col_norms: Vec<T>,
}

impl<T: Float> FittingBuffer<T> {
    /// Create a new fitting buffer with estimated capacities.
    pub fn new(k: usize, n_coeffs: usize) -> Self {
        Self {
            weights: Vec::with_capacity(k),
            xtw_x: Vec::with_capacity(n_coeffs * n_coeffs),
            xtw_y: Vec::with_capacity(n_coeffs),
            col_norms: Vec::with_capacity(n_coeffs),
        }
    }
}

/// Persistent buffers for global executor state.
pub struct ExecutorBuffer<T: Float> {
    /// Minimum values for each dimension.
    pub mins: Vec<T>,
    /// Maximum values for each dimension.
    pub maxs: Vec<T>,
    /// Normalization scales for each dimension.
    pub scales: Vec<T>,
    /// Robustness weights for iterative refinement.
    pub robustness_weights: Vec<T>,
    /// Residuals for iterative refinement.
    pub residuals: Vec<T>,
    /// Sorted residuals for median computation.
    pub sorted_residuals: Vec<T>,
}

impl<T: Float> ExecutorBuffer<T> {
    /// Create a new executor buffer with given capacities.
    pub fn new(n: usize, dims: usize) -> Self {
        Self {
            mins: Vec::with_capacity(dims),
            maxs: Vec::with_capacity(dims),
            scales: Vec::with_capacity(dims),
            robustness_weights: Vec::with_capacity(n),
            residuals: Vec::with_capacity(n),
            sorted_residuals: Vec::with_capacity(n),
        }
    }

    /// Ensure buffers have enough capacity for given dimensions and points.
    pub fn ensure_capacity(&mut self, n: usize, dims: usize) {
        if self.mins.capacity() < dims {
            self.mins = Vec::with_capacity(dims);
            self.maxs = Vec::with_capacity(dims);
            self.scales = Vec::with_capacity(dims);
        }
        if self.robustness_weights.capacity() < n {
            self.robustness_weights = Vec::with_capacity(n);
            self.residuals = Vec::with_capacity(n);
            self.sorted_residuals = Vec::with_capacity(n);
        }
    }
}

/// Persistent buffers for cross-validation subsets.
pub struct CVBuffer<T: Float> {
    /// Training subset x-values.
    pub train_x: Vec<T>,
    /// Training subset y-values.
    pub train_y: Vec<T>,
    /// Test subset x-values.
    pub test_x: Vec<T>,
    /// Test subset y-values.
    pub test_y: Vec<T>,
}

impl<T: Float> CVBuffer<T> {
    /// Create a new CV buffer with given capacities.
    pub fn new(n: usize, dims: usize) -> Self {
        Self {
            train_x: Vec::with_capacity(n * dims),
            train_y: Vec::with_capacity(n),
            test_x: Vec::with_capacity(n * dims),
            test_y: Vec::with_capacity(n),
        }
    }

    /// Ensure buffers have enough capacity for given dimensions and points.
    pub fn ensure_capacity(&mut self, n: usize, dims: usize) {
        if self.train_x.capacity() < n * dims {
            self.train_x = Vec::with_capacity(n * dims);
            self.test_x = Vec::with_capacity(n * dims);
        }
        if self.train_y.capacity() < n {
            self.train_y = Vec::with_capacity(n);
            self.test_y = Vec::with_capacity(n);
        }
    }
}
