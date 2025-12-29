//! Interpolation utilities for LOESS smoothing.
//!
//! ## Purpose
//!
//! This module provides an interpolation surface for efficient nD LOESS evaluation
//! by fitting at cell vertices and using n-linear interpolation.
//!
//! ## Design notes
//!
//! * **Vertex-based**: Fits are computed only at cell vertices.
//! * **Multilinear**: Uses n-linear interpolation within hypercube cells.
//! * **Adaptive**: Cell subdivision based on data density.
//!
//! ## Key concepts
//!
//! * **Cell**: A hypercube region with 2^d vertices for surface interpolation.
//! * **Vertex**: A corner point where local regression is computed.
//! * **Interpolation**: Weighted average of vertex fits based on position.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::fmt::Debug;
use core::option::Option;
use num_traits::Float;

// Internal dependencies
use crate::algorithms::regression::FittingBuffer;
use crate::engine::workspace::LoessWorkspace;
use crate::math::neighborhood::{KDTree, Neighborhood, PointDistance};

// ============================================================================
// Surface Cell
// ============================================================================

/// A cell in the spatial partition with references to its vertices.
#[derive(Debug, Clone)]
pub struct SurfaceCell<T: Float> {
    /// Lower bounds for each dimension.
    pub lower: Vec<T>,
    /// Upper bounds for each dimension.
    pub upper: Vec<T>,
    /// Indices of the 2^d vertices (corners) in the vertices array.
    pub vertex_indices: Vec<usize>,
    /// Child cell indices (for tree structure), None if leaf.
    pub children: Option<(usize, usize)>,
    /// Split dimension (if not a leaf).
    pub split_dim: Option<usize>,
    /// Split value (if not a leaf).
    pub split_val: Option<T>,
    /// Start index in point index array (for O(1) point counting).
    pub point_lo: usize,
    /// End index in point index array, inclusive (for O(1) point counting).
    pub point_hi: usize,
}

// ============================================================================
// Interpolation Surface
// ============================================================================

/// Pre-computed surface for efficient LOESS evaluation.
///
/// This structure enables fast evaluation by:
/// 1. Building a spatial partition (KD-tree-like cell structure)
/// 2. Fitting local regression only at cell vertices
/// 3. Interpolating within cells using n-linear interpolation
#[derive(Debug, Clone)]
pub struct InterpolationSurface<T: Float> {
    /// Fitted data at each vertex: [val, ∂/∂x₁, ∂/∂x₂, ...] for each vertex.
    /// Layout: vertex 0 data, vertex 1 data, ... with (d+1) values per vertex.
    pub vertex_data: Vec<T>,
    /// Vertex coordinates (stored for refitting).
    pub vertices: Vec<Vec<T>>,
    /// Spatial cells for lookup.
    pub cells: Vec<SurfaceCell<T>>,
    /// Root cell index.
    pub root: usize,
    /// Number of dimensions.
    pub dimensions: usize,
}

impl<T: Float + Debug + Send + Sync + 'static> InterpolationSurface<T> {
    /// Build an interpolation surface from data.
    ///
    /// This creates a spatial partition and interpolates between pre-computed vertex fits.
    /// The fitter closure performs local regression at each vertex.
    #[allow(clippy::too_many_arguments)]
    pub fn build<D, F>(
        x: &[T],
        y: &[T],
        dimensions: usize,
        fraction: T,
        window_size: usize,
        dist_calc: &D,
        kdtree: &KDTree<T>,
        max_vertices: usize,
        mut fitter: F,
        workspace: &mut LoessWorkspace<T>,
        cell_size: T,
    ) -> Self
    where
        D: PointDistance<T>,
        F: FnMut(&[T], &Neighborhood<T>, &mut FittingBuffer<T>) -> Option<Vec<T>>,
    {
        let n = x.len() / dimensions;

        // Compute bounding box
        let mut lower = vec![T::infinity(); dimensions];
        let mut upper = vec![T::neg_infinity(); dimensions];

        for i in 0..n {
            for d in 0..dimensions {
                let val = x[i * dimensions + d];
                if val < lower[d] {
                    lower[d] = val;
                }
                if val > upper[d] {
                    upper[d] = val;
                }
            }
        }

        // Expand bounding box slightly (0.5%)
        for d in 0..dimensions {
            let range = upper[d] - lower[d];
            let margin = range * T::from(0.005).unwrap();
            lower[d] = lower[d] - margin;
            upper[d] = upper[d] + margin;
        }

        // Build initial cells and vertices
        let mut vertices: Vec<Vec<T>> = Vec::new();
        let mut cells: Vec<SurfaceCell<T>> = Vec::new();

        // Create root cell with bounding box corners as vertices
        let num_corners = 1usize << dimensions; // 2^d corners
        let mut root_vertex_indices = Vec::with_capacity(num_corners);

        for corner_idx in 0..num_corners {
            let mut vertex = Vec::with_capacity(dimensions);
            for d in 0..dimensions {
                // Use lower or upper based on bit pattern
                if (corner_idx >> d) & 1 == 0 {
                    vertex.push(lower[d]);
                } else {
                    vertex.push(upper[d]);
                }
            }
            root_vertex_indices.push(vertices.len());
            vertices.push(vertex);
        }

        let root_cell = SurfaceCell {
            lower: lower.clone(),
            upper: upper.clone(),
            vertex_indices: root_vertex_indices,
            children: None,
            split_dim: None,
            split_val: None,
            point_lo: 0,
            point_hi: n.saturating_sub(1),
        };
        cells.push(root_cell);

        // Create point index array for O(1) point counting during subdivision
        let mut pi: Vec<usize> = (0..n).collect();

        // Cleveland's subdivision parameters:
        // new_cell = span * cell
        // Then: fc = floor(n * new_cell) = floor(n * span * cell)
        let fc = (T::from(n).unwrap() * cell_size * fraction)
            .floor()
            .to_usize()
            .unwrap_or(1)
            .max(1);

        // Disable the minimum cell diameter check
        let fd = T::zero();

        // Subdivide cells using Cleveland's stopping criteria
        Self::subdivide_cells(
            &mut cells,
            &mut vertices,
            &mut pi,
            0,
            x,
            dimensions,
            max_vertices,
            fc, // min points per cell
            fd, // min cell diameter (0 = disabled)
        );

        // Fit at each vertex - store value + d partial derivatives
        // Layout: [v0_val, v0_dx1, ..., v0_dxd, v1_val, v1_dx1, ..., v1_dxd, ...]
        let stride = dimensions + 1; // d+1 values per vertex
        let mut vertex_data = vec![T::zero(); vertices.len() * stride];

        for (v_idx, vertex) in vertices.iter().enumerate() {
            // Find neighbors for this vertex using workspace buffers
            kdtree.find_k_nearest(
                vertex,
                window_size,
                dist_calc,
                None,
                &mut workspace.search_buffer,
                &mut workspace.neighborhood,
            );
            let neighborhood = &workspace.neighborhood;

            let base_idx = v_idx * stride;

            if neighborhood.is_empty() {
                // Fallback: use mean of all y values, zero derivatives
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                vertex_data[base_idx] = mean;
                // Derivatives remain zero
                continue;
            }

            // Fit local regression at this vertex using injected fitter
            // Returns [value, d/dx1, d/dx2, ..., d/dxd]
            if let Some(coeffs) = fitter(
                vertex,
                &workspace.neighborhood,
                &mut workspace.fitting_buffer,
            ) {
                for (i, &c) in coeffs.iter().take(stride).enumerate() {
                    vertex_data[base_idx + i] = c;
                }
            } else {
                // Fallback to mean, zero derivatives
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                vertex_data[base_idx] = mean;
            }
        }

        Self {
            vertex_data,
            vertices,
            cells,
            root: 0,
            dimensions,
        }
    }

    /// Refit vertex values without rebuilding the cell structure.
    ///
    /// This is used during robustness iterations to update vertex fits
    /// with new robustness weights, avoiding the expensive cell subdivision.
    pub fn refit_values<D, F>(
        &mut self,
        y: &[T],
        kdtree: &KDTree<T>,
        window_size: usize,
        dist_calc: &D,
        mut fitter: F,
        workspace: &mut LoessWorkspace<T>,
    ) where
        D: PointDistance<T>,
        F: FnMut(&[T], &Neighborhood<T>, &mut FittingBuffer<T>) -> Option<Vec<T>>,
    {
        let n = y.len() / self.dimensions;
        let stride = self.dimensions + 1; // d+1 values per vertex

        for (v_idx, vertex) in self.vertices.iter().enumerate() {
            // Find neighbors for this vertex using workspace buffers
            kdtree.find_k_nearest(
                vertex,
                window_size,
                dist_calc,
                None,
                &mut workspace.search_buffer,
                &mut workspace.neighborhood,
            );
            let neighborhood = &workspace.neighborhood;

            let base_idx = v_idx * stride;

            if neighborhood.is_empty() {
                // Fallback: use mean of all y values, zero derivatives
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                self.vertex_data[base_idx] = mean;
                for i in 1..stride {
                    self.vertex_data[base_idx + i] = T::zero();
                }
                continue;
            }

            // Fit local regression at this vertex using injected fitter
            if let Some(coeffs) = fitter(
                vertex,
                &workspace.neighborhood,
                &mut workspace.fitting_buffer,
            ) {
                for (i, &c) in coeffs.iter().take(stride).enumerate() {
                    self.vertex_data[base_idx + i] = c;
                }
            } else {
                // Fallback to mean, zero derivatives
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                self.vertex_data[base_idx] = mean;
                for i in 1..stride {
                    self.vertex_data[base_idx + i] = T::zero();
                }
            }
        }
    }

    /// Subdivide cells recursively using Cleveland's stopping criteria.
    ///
    /// Uses O(1) point counting via index ranges in the `pi` array.
    /// Points are partitioned during subdivision like scikit-misc's ehg106/ehg124.
    ///
    /// Stops subdividing when:
    /// - `points_in_cell <= fc` (minimum points per cell)
    /// - `cell_diameter <= fd` (minimum cell diameter)
    /// - `vertices.len() >= max_vertices` (hard limit)
    #[allow(clippy::too_many_arguments)]
    fn subdivide_cells(
        cells: &mut Vec<SurfaceCell<T>>,
        vertices: &mut Vec<Vec<T>>,
        pi: &mut [usize], // Point index array (reordered in-place)
        cell_idx: usize,
        x: &[T],
        dimensions: usize,
        max_vertices: usize,
        fc: usize, // min points per cell (Cleveland's fc)
        fd: T,     // min cell diameter (Cleveland's fd)
    ) {
        // Stop if we have reached the vertex limit
        if vertices.len() >= max_vertices {
            return;
        }

        let cell = &cells[cell_idx];
        let lo = cell.point_lo;
        let hi = cell.point_hi;

        // O(1) point count using index ranges!
        let points_in_cell = if hi >= lo { hi - lo + 1 } else { 0 };

        // Compute cell diameter (Euclidean distance of diagonal)
        let mut diam_sq = T::zero();
        for d in 0..dimensions {
            let range = cell.upper[d] - cell.lower[d];
            diam_sq = diam_sq + range * range;
        }
        let diam = diam_sq.sqrt();

        // Cleveland's stopping criteria:
        // leaf = (points <= fc) OR (diameter <= fd)
        let is_leaf = points_in_cell <= fc || diam <= fd;

        if is_leaf || points_in_cell == 0 {
            return;
        }

        // Additional safety: don't subdivide if we'd exceed vertex limits
        let num_new_vertices = 1usize << (dimensions - 1);
        if vertices.len() + num_new_vertices > max_vertices {
            return;
        }

        // Find dimension with largest spread in point data (like Cleveland's ehg129)
        let mut best_dim = 0;
        let mut best_spread = T::zero();
        for d in 0..dimensions {
            let mut min_val = T::infinity();
            let mut max_val = T::neg_infinity();
            for &idx in &pi[lo..=hi] {
                let val = x[idx * dimensions + d];
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            let spread = max_val - min_val;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }

        // Find median split point
        let m = (lo + hi) / 2;

        // Partition points around median using Floyd-Rivest style selection
        Self::partition_by_dim(pi, lo, hi, m, x, best_dim, dimensions);

        let split_val = x[pi[m] * dimensions + best_dim];

        // Create child cells with inherited bounds
        let left_lower = cells[cell_idx].lower.clone();
        let mut left_upper = cells[cell_idx].upper.clone();
        left_upper[best_dim] = split_val;

        let mut right_lower = cells[cell_idx].lower.clone();
        let right_upper = cells[cell_idx].upper.clone();
        right_lower[best_dim] = split_val;

        // Create vertices for new split plane
        let start_vertex_idx = vertices.len();

        for corner_idx in 0..num_new_vertices {
            let mut vertex = vec![T::zero(); dimensions];
            vertex[best_dim] = split_val;

            let mut bit_pos = 0;
            for (d, v_d) in vertex.iter_mut().enumerate().take(dimensions) {
                if d != best_dim {
                    if (corner_idx >> bit_pos) & 1 == 0 {
                        *v_d = cells[cell_idx].lower[d];
                    } else {
                        *v_d = cells[cell_idx].upper[d];
                    }
                    bit_pos += 1;
                }
            }
            vertices.push(vertex);
        }

        // Build vertex indices for child cells
        let num_corners = 1usize << dimensions;
        let mut left_vertices = vec![0; num_corners];
        let mut right_vertices = vec![0; num_corners];

        let parent_vertices = cells[cell_idx].vertex_indices.clone();

        for child_corner_idx in 0..num_corners {
            let dim_bit = (child_corner_idx >> best_dim) & 1;

            // Calculate "compressed" index (skipping best_dim) for new vertices
            let mask = (1 << best_dim) - 1;
            let lower_bits = child_corner_idx & mask;
            let upper_bits = child_corner_idx >> (best_dim + 1);
            let compressed_idx = (upper_bits << best_dim) | lower_bits;

            // Left Child
            if dim_bit == 0 {
                left_vertices[child_corner_idx] = parent_vertices[child_corner_idx];
            } else {
                left_vertices[child_corner_idx] = start_vertex_idx + compressed_idx;
            }

            // Right Child
            if dim_bit == 0 {
                right_vertices[child_corner_idx] = start_vertex_idx + compressed_idx;
            } else {
                right_vertices[child_corner_idx] = parent_vertices[child_corner_idx];
            }
        }

        let left_idx = cells.len();
        let right_idx = cells.len() + 1;

        // Child cells inherit point ranges from partition
        cells.push(SurfaceCell {
            lower: left_lower,
            upper: left_upper,
            vertex_indices: left_vertices,
            children: None,
            split_dim: None,
            split_val: None,
            point_lo: lo,
            point_hi: m,
        });

        cells.push(SurfaceCell {
            lower: right_lower,
            upper: right_upper,
            vertex_indices: right_vertices,
            children: None,
            split_dim: None,
            split_val: None,
            point_lo: m + 1,
            point_hi: hi,
        });

        // Update parent
        cells[cell_idx].children = Some((left_idx, right_idx));
        cells[cell_idx].split_dim = Some(best_dim);
        cells[cell_idx].split_val = Some(split_val);

        // Recurse
        Self::subdivide_cells(
            cells,
            vertices,
            pi,
            left_idx,
            x,
            dimensions,
            max_vertices,
            fc,
            fd,
        );
        Self::subdivide_cells(
            cells,
            vertices,
            pi,
            right_idx,
            x,
            dimensions,
            max_vertices,
            fc,
            fd,
        );
    }

    /// Partition pi[lo..=hi] so that pi[m] contains the median element along dimension dim.
    /// Uses a quickselect-style algorithm similar to Floyd-Rivest.
    fn partition_by_dim(
        pi: &mut [usize],
        lo: usize,
        hi: usize,
        k: usize, // target position
        x: &[T],
        dim: usize,
        dimensions: usize,
    ) {
        if lo >= hi {
            return;
        }

        let mut left = lo;
        let mut right = hi;

        while left < right {
            // Partition around pivot at position k
            let pivot_val = x[pi[k] * dimensions + dim];

            // Move pivot to end
            pi.swap(k, right);

            let mut store_idx = left;
            for i in left..right {
                if x[pi[i] * dimensions + dim] < pivot_val {
                    pi.swap(i, store_idx);
                    store_idx += 1;
                }
            }

            // Move pivot to final position
            pi.swap(store_idx, right);

            // Narrow search range
            if store_idx == k {
                return;
            } else if store_idx < k {
                left = store_idx + 1;
            } else {
                right = store_idx.saturating_sub(1);
            }
        }
    }

    /// Evaluate the surface at a query point using multilinear interpolation.
    pub fn evaluate(&self, query: &[T]) -> T {
        // Find the leaf cell containing the query point
        let cell_idx = self.find_cell(query);
        let cell = &self.cells[cell_idx];

        // Multilinear interpolation
        self.interpolate_in_cell(cell, query)
    }

    /// Find the leaf cell containing a query point.
    fn find_cell(&self, query: &[T]) -> usize {
        let mut current = self.root;

        loop {
            let cell = &self.cells[current];

            match cell.children {
                Some((left, right)) => {
                    let split_dim = cell.split_dim.unwrap();
                    let split_val = cell.split_val.unwrap();

                    if query[split_dim] <= split_val {
                        current = left;
                    } else {
                        current = right;
                    }
                }
                Option::None => {
                    // Leaf cell
                    return current;
                }
            }
        }
    }

    /// Hermite basis functions as used in scikit-misc ehg128.
    #[inline]
    fn hermite_phi0(h: T) -> T {
        // (1-h)^2 * (1+2h)
        let one = T::one();
        let two = T::from(2.0).unwrap();
        (one - h) * (one - h) * (one + two * h)
    }

    #[inline]
    fn hermite_phi1(h: T) -> T {
        // h^2 * (3-2h)
        let two = T::from(2.0).unwrap();
        let three = T::from(3.0).unwrap();
        h * h * (three - two * h)
    }

    #[inline]
    fn hermite_psi0(h: T) -> T {
        // h * (1-h)^2
        let one = T::one();
        h * (one - h) * (one - h)
    }

    #[inline]
    fn hermite_psi1(h: T) -> T {
        // h^2 * (h-1)
        let one = T::one();
        h * h * (h - one)
    }

    /// Perform Hermite interpolation within a cell using value + derivatives.
    /// Matches scikit-misc's ehg128 algorithm.
    fn interpolate_in_cell(&self, cell: &SurfaceCell<T>, query: &[T]) -> T {
        let d = self.dimensions;
        let stride = d + 1; // d+1 values per vertex: [value, d/dx1, d/dx2, ..., d/dxd]

        // Get vertex indices for lower and upper corners
        // In a 1D case: vertex 0 is lower, vertex 1 is upper
        // In nD: we use tensor interpolation dimension by dimension

        // For simplicity in 1D case (most common in LOESS)
        if d == 1 {
            // Get two vertices
            if cell.vertex_indices.len() < 2 {
                // Fallback to weighted average
                return self.fallback_interpolation(cell);
            }

            let v0_idx = cell.vertex_indices[0];
            let v1_idx = cell.vertex_indices[1];

            // Get vertex data: [value, derivative]
            let g0_val = self.vertex_data[v0_idx * stride];
            let g0_deriv = self.vertex_data[v0_idx * stride + 1];
            let g1_val = self.vertex_data[v1_idx * stride];
            let g1_deriv = self.vertex_data[v1_idx * stride + 1];

            // Compute h = normalized position in cell
            let range = cell.upper[0] - cell.lower[0];
            if range <= T::epsilon() {
                return g0_val;
            }
            let h = (query[0] - cell.lower[0]) / range;
            let h = h.max(T::zero()).min(T::one());

            // Hermite basis functions
            let phi0 = Self::hermite_phi0(h);
            let phi1 = Self::hermite_phi1(h);
            let psi0 = Self::hermite_psi0(h);
            let psi1 = Self::hermite_psi1(h);

            // Hermite interpolation matching scikit-misc:
            // result = phi0*g0_val + phi1*g1_val + (psi0*g0_deriv + psi1*g1_deriv) * range
            return phi0 * g0_val + phi1 * g1_val + (psi0 * g0_deriv + psi1 * g1_deriv) * range;
        }

        // For higher dimensions, use tensor Hermite interpolation
        // This is more complex - scikit-misc uses recursive tensor product
        // For now, fallback to multilinear for nD (can be enhanced later)
        self.hermite_tensor_interpolation(cell, query)
    }

    /// Fallback interpolation when cell has insufficient vertices.
    fn fallback_interpolation(&self, cell: &SurfaceCell<T>) -> T {
        let stride = self.dimensions + 1;
        let sum: T = cell
            .vertex_indices
            .iter()
            .filter_map(|&idx| {
                let base = idx * stride;
                self.vertex_data.get(base).copied()
            })
            .fold(T::zero(), |a, b| a + b);
        let count = T::from(cell.vertex_indices.len()).unwrap();
        if count > T::zero() {
            sum / count
        } else {
            T::zero()
        }
    }

    /// Tensor Hermite interpolation for nD case.
    /// Uses dimension-by-dimension interpolation like scikit-misc.
    fn hermite_tensor_interpolation(&self, cell: &SurfaceCell<T>, query: &[T]) -> T {
        let d = self.dimensions;
        let stride = d + 1;
        let num_corners = 1usize << d;

        // Get all corner data
        let mut g: Vec<Vec<T>> = Vec::with_capacity(num_corners);
        for &v_idx in &cell.vertex_indices {
            let base = v_idx * stride;
            let data: Vec<T> = (0..stride)
                .filter_map(|i| self.vertex_data.get(base + i).copied())
                .collect();
            if data.len() == stride {
                g.push(data);
            } else {
                // Fallback for missing data
                let mut default = vec![T::zero(); stride];
                default[0] = self.vertex_data.get(base).copied().unwrap_or(T::zero());
                g.push(default);
            }
        }

        // Ensure we have 2^d corners
        while g.len() < num_corners {
            g.push(vec![T::zero(); stride]);
        }

        // Tensor interpolation: process dimension by dimension
        let mut lg = num_corners;

        for dim in (0..d).rev() {
            let range = cell.upper[dim] - cell.lower[dim];
            let h = if range > T::epsilon() {
                let t = (query[dim] - cell.lower[dim]) / range;
                t.max(T::zero()).min(T::one())
            } else {
                T::zero()
            };

            let phi0 = Self::hermite_phi0(h);
            let phi1 = Self::hermite_phi1(h);
            let psi0 = Self::hermite_psi0(h);
            let psi1 = Self::hermite_psi1(h);

            lg /= 2;
            let (lower, upper) = g.split_at_mut(lg);
            for (row_curr, row_next) in lower.iter_mut().zip(upper.iter()) {
                // Value interpolation with derivative terms
                row_curr[0] = phi0 * row_curr[0]
                    + phi1 * row_next[0]
                    + (psi0 * row_curr[dim + 1] + psi1 * row_next[dim + 1]) * range;

                // Interpolate partial derivatives for remaining dimensions
                for (val, &next_val) in row_curr.iter_mut().zip(row_next.iter()).skip(1).take(dim) {
                    *val = phi0 * *val + phi1 * next_val;
                }
            }
        }

        g[0][0]
    }
}
