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
    /// Fitted values at each vertex (the y0 coefficient).
    pub vertex_values: Vec<T>,
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
        fitter: F,
        cell_size: T,
    ) -> Self
    where
        D: PointDistance<T>,
        F: Fn(&[T], &Neighborhood<T>) -> Option<T>,
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

        // Fit at each vertex
        let mut vertex_values = vec![T::zero(); vertices.len()];

        for (v_idx, vertex) in vertices.iter().enumerate() {
            // Find neighbors for this vertex
            let neighborhood = kdtree.find_k_nearest(vertex, window_size, dist_calc, None);

            if neighborhood.is_empty() {
                // Fallback: use mean of all y values
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                vertex_values[v_idx] = mean;
                continue;
            }

            // Fit local regression at this vertex using injected fitter
            if let Some(val) = fitter(vertex, &neighborhood) {
                vertex_values[v_idx] = val;
            } else {
                // Fallback to mean
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                vertex_values[v_idx] = mean;
            }
        }

        Self {
            vertex_values,
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
        fitter: F,
    ) where
        D: PointDistance<T>,
        F: Fn(&[T], &Neighborhood<T>) -> Option<T>,
    {
        let n = y.len() / self.dimensions;

        for (v_idx, vertex) in self.vertices.iter().enumerate() {
            // Find neighbors for this vertex
            let neighborhood = kdtree.find_k_nearest(vertex, window_size, dist_calc, None);

            if neighborhood.is_empty() {
                // Fallback: use mean of all y values
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                self.vertex_values[v_idx] = mean;
                continue;
            }

            // Fit local regression at this vertex using injected fitter
            if let Some(val) = fitter(vertex, &neighborhood) {
                self.vertex_values[v_idx] = val;
            } else {
                // Fallback to mean
                let mean = y.iter().copied().fold(T::zero(), |a, b| a + b) / T::from(n).unwrap();
                self.vertex_values[v_idx] = mean;
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

    /// Cubic Hermite blending function (smoothstep).
    /// Maps [0, 1] to [0, 1] with zero derivative at endpoints.
    /// f(t) = 3t^2 - 2t^3
    fn smoothstep(t: T) -> T {
        let three = T::from(3.0).unwrap();
        let two = T::from(2.0).unwrap();
        t * t * (three - two * t)
    }

    /// Perform multilinear interpolation within a cell.
    fn interpolate_in_cell(&self, cell: &SurfaceCell<T>, query: &[T]) -> T {
        let d = self.dimensions;

        // Optimization: Use a stack-allocated buffer for common low-dimensional cases
        let mut weights_stack = [T::zero(); 16];
        let mut weights_heap = Vec::new();

        if d <= 16 {
            for (dim, item) in query.iter().enumerate().take(d) {
                let range = cell.upper[dim] - cell.lower[dim];
                if range > T::epsilon() {
                    let t = (*item - cell.lower[dim]) / range;
                    let t_clamped = t.max(T::zero()).min(T::one());
                    weights_stack[dim] = Self::smoothstep(t_clamped);
                } else {
                    weights_stack[dim] = T::zero();
                }
            }
        } else {
            weights_heap.reserve(d);
            for (dim, item) in query.iter().enumerate().take(d) {
                let range = cell.upper[dim] - cell.lower[dim];
                if range > T::epsilon() {
                    let t = (*item - cell.lower[dim]) / range;
                    let t_clamped = t.max(T::zero()).min(T::one());
                    weights_heap.push(Self::smoothstep(t_clamped));
                } else {
                    weights_heap.push(T::zero());
                }
            }
        };

        let weights: &[T] = if d <= 16 {
            &weights_stack[..d]
        } else {
            &weights_heap
        };

        // Multilinear interpolation: weighted sum over all 2^d corners
        let num_corners = 1usize << d;
        let mut result = T::zero();
        let mut total_weight = T::zero();

        for corner_idx in 0..num_corners {
            // Compute weight for this corner
            let mut corner_weight = T::one();
            for (dim, weight) in weights.iter().enumerate().take(d) {
                let is_upper = (corner_idx >> dim) & 1 == 1;
                if is_upper {
                    corner_weight = corner_weight * *weight;
                } else {
                    corner_weight = corner_weight * (T::one() - *weight);
                }
            }

            // Find the vertex for this corner pattern
            // We need to match the corner pattern to the vertex in the cell
            if corner_idx < cell.vertex_indices.len() {
                let vertex_idx = cell.vertex_indices[corner_idx];
                if vertex_idx < self.vertex_values.len() {
                    result = result + corner_weight * self.vertex_values[vertex_idx];
                    total_weight = total_weight + corner_weight;
                }
            }
        }

        if total_weight > T::epsilon() {
            result / total_weight
        } else {
            // Fallback: average of all vertex values in cell
            let sum: T = cell
                .vertex_indices
                .iter()
                .filter_map(|&idx| self.vertex_values.get(idx).copied())
                .fold(T::zero(), |a, b| a + b);
            let count = T::from(cell.vertex_indices.len()).unwrap();
            sum / count
        }
    }
}
