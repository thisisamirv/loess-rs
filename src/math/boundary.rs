//! Boundary padding strategies for local regression.
//!
//! ## Purpose
//!
//! This module implements boundary padding strategies to reduce smoothing bias at
//! data edges. By providing context beyond the original boundaries, local
//! regression can perform better near the start and end of the dataset.
//!
//! ## Design notes
//!
//! * **Strategy Pattern**: Uses `BoundaryPolicy` enum to select the padding method.
//! * **Allocation**: Creates new vectors for padded data (necessary for extension).
//!
//! ## Key concepts
//!
//! * **Boundary Effect**: The tendency for local regression to have higher bias at edges.
//! * **Padding strategies**: `Extend` (repeat edge), `Reflect` (mirror), `Zero` (pad 0).
//!
//! ## Invariants
//!
//! * Padding length is limited to half the window size or `n - 1`.
//! * Original data is preserved in the middle of the value range.
//!
//! ## Non-goals
//!
//! * This module does not perform in-place modification of input data.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering::Equal;
use core::iter::repeat_n;
use num_traits::Float;

/// Policy for handling boundaries at the start and end of a data stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryPolicy {
    /// Replicate edge values to provide context for boundary points.
    Extend,

    /// Mirror values across the boundary.
    Reflect,

    /// Use zero padding beyond data boundaries.
    Zero,

    /// No boundary padding (standard LOESS behavior).
    #[default]
    NoBoundary,
}

impl BoundaryPolicy {
    /// Apply the boundary policy to pad input data (1D or nD).
    ///
    /// Returns augmented x, augmented y, and a mapping from augmented index to original index.
    pub fn apply<T: Float>(
        &self,
        x: &[T],
        y: &[T],
        dimensions: usize,
        window_size: usize,
    ) -> (Vec<T>, Vec<T>, Vec<usize>) {
        let n = x.len() / dimensions;
        let d = dimensions;

        // Mapping from augmented index to original index
        let mapping: Vec<usize> = (0..n).collect();

        // 1D specific full implementation
        if d == 1 {
            // Check for NoBoundary policy first
            if *self == BoundaryPolicy::NoBoundary {
                return (x.to_vec(), y.to_vec(), mapping);
            }

            let pad_len = (window_size / 2).min(n - 1);
            if pad_len == 0 {
                return (x.to_vec(), y.to_vec(), mapping);
            }

            let total_len = n + 2 * pad_len;
            let mut px = Vec::with_capacity(total_len);
            let mut py = Vec::with_capacity(total_len);

            // Prepend padding
            match self {
                BoundaryPolicy::Extend => {
                    let x0 = x[0];
                    let y0 = y[0];
                    let dx = x[1] - x[0];
                    for i in (1..=pad_len).rev() {
                        px.push(x0 - T::from(i).unwrap() * dx);
                        py.push(y0);
                    }
                }
                BoundaryPolicy::Reflect => {
                    let x0 = x[0];
                    for i in (1..=pad_len).rev() {
                        px.push(x0 - (x[i] - x0));
                        py.push(y[i]);
                    }
                }
                BoundaryPolicy::Zero => {
                    let x0 = x[0];
                    let dx = x[1] - x[0];
                    for i in (1..=pad_len).rev() {
                        px.push(x0 - T::from(i).unwrap() * dx);
                        py.push(T::zero());
                    }
                }
                BoundaryPolicy::NoBoundary => unreachable!(),
            }

            // Add original data
            px.extend_from_slice(x);
            py.extend_from_slice(y);

            // Append padding
            match self {
                BoundaryPolicy::Extend => {
                    let xn = x[n - 1];
                    let yn = y[n - 1];
                    let dx = x[n - 1] - x[n - 2];
                    for i in 1..=pad_len {
                        px.push(xn + T::from(i).unwrap() * dx);
                        py.push(yn);
                    }
                }
                BoundaryPolicy::Reflect => {
                    let xn = x[n - 1];
                    for i in 1..=pad_len {
                        px.push(xn + (xn - x[n - 1 - i]));
                        py.push(y[n - 1 - i]);
                    }
                }
                BoundaryPolicy::Zero => {
                    let xn = x[n - 1];
                    let dx = x[n - 1] - x[n - 2];
                    for i in 1..=pad_len {
                        px.push(xn + T::from(i).unwrap() * dx);
                        py.push(T::zero());
                    }
                }
                BoundaryPolicy::NoBoundary => unreachable!(),
            }

            let n_padded = px.len();
            let mut full_mapping = Vec::with_capacity(n_padded);
            full_mapping.extend(repeat_n(0, pad_len));
            full_mapping.extend(0..n);
            full_mapping.extend(repeat_n(n - 1, n_padded - n - pad_len));

            return (px, py, full_mapping);
        }

        if *self == BoundaryPolicy::Extend
            || *self == BoundaryPolicy::NoBoundary
            || (d > 1 && *self == BoundaryPolicy::Zero)
        {
            // TODO: Zero padding in nD is tricky without a clear "outside".
            // We'll skip extending for now or implement if needed.
            return (x.to_vec(), y.to_vec(), mapping);
        }

        if *self == BoundaryPolicy::Reflect && d > 1 {
            let mut px = x.to_vec();
            let mut py = y.to_vec();
            let mut p_mapping = mapping;

            let pad_count = (window_size / 2).min(n / 2).max(1);

            for j in 0..d {
                // Find indices of points with smallest and largest values in dimension j
                let mut indices: Vec<usize> = (0..n).collect();
                indices.sort_by(|&a, &b| x[a * d + j].partial_cmp(&x[b * d + j]).unwrap_or(Equal));

                // Min boundary reflection
                let min_val = x[indices[0] * d + j];
                for &idx in indices.iter().take(pad_count) {
                    let mut new_point = vec![T::zero(); d];
                    for k in 0..d {
                        if k == j {
                            new_point[k] = min_val - (x[idx * d + j] - min_val);
                        } else {
                            new_point[k] = x[idx * d + k];
                        }
                    }
                    px.extend_from_slice(&new_point);
                    py.push(y[idx]);
                    p_mapping.push(idx);
                }

                // Max boundary reflection
                let max_val = x[indices[n - 1] * d + j];
                for &idx in indices.iter().skip(n - pad_count).take(pad_count) {
                    let mut new_point = vec![T::zero(); d];
                    for k in 0..d {
                        if k == j {
                            new_point[k] = max_val + (max_val - x[idx * d + j]);
                        } else {
                            new_point[k] = x[idx * d + k];
                        }
                    }
                    px.extend_from_slice(&new_point);
                    py.push(y[idx]);
                    p_mapping.push(idx);
                }
            }
            return (px, py, p_mapping);
        }

        // Fallback to original
        (x.to_vec(), y.to_vec(), mapping)
    }
}
