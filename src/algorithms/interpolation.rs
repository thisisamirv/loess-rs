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
use num_traits::Float;

// Internal dependencies
use crate::math::neighborhood::{KDTree, Neighborhood, PointDistance};
use crate::primitives::window::Window;

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
        let window_size = Window::calculate_span(n, fraction);

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
        };
        cells.push(root_cell);

        // Compute subdivision thresholds
        let mut thresholds = Vec::with_capacity(dimensions);
        let range_fraction = fraction * cell_size;
        for d in 0..dimensions {
            let range = upper[d] - lower[d];
            thresholds.push(range * range_fraction);
        }

        // Subdivide cells
        Self::subdivide_cells(
            &mut cells,
            &mut vertices,
            0,
            x,
            dimensions,
            max_vertices,
            window_size,
            &thresholds,
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
            cells,
            root: 0,
            dimensions,
        }
    }

    /// Subdivide cells recursively.
    #[allow(clippy::too_many_arguments)]
    fn subdivide_cells(
        cells: &mut Vec<SurfaceCell<T>>,
        vertices: &mut Vec<Vec<T>>,
        cell_idx: usize,
        x: &[T],
        dimensions: usize,
        max_vertices: usize,
        target_window: usize,
        thresholds: &[T],
    ) {
        // Stop if we have enough vertices
        if vertices.len() >= max_vertices {
            return;
        }

        let n = x.len() / dimensions;

        // Count points in this cell
        let cell = &cells[cell_idx];
        let mut points_in_cell = 0;
        for i in 0..n {
            let mut inside = true;
            for d in 0..dimensions {
                let val = x[i * dimensions + d];
                if val < cell.lower[d] || val > cell.upper[d] {
                    inside = false;
                    break;
                }
            }
            if inside {
                points_in_cell += 1;
            }
        }

        // Check if we need to split based on geometry (cell size)
        let mut force_split = false;
        for (d, &threshold) in thresholds.iter().enumerate().take(dimensions) {
            let range = cell.upper[d] - cell.lower[d];
            if range > threshold {
                force_split = true;
                break;
            }
        }

        // Don't subdivide if cell has few points relative to window size
        if (!force_split && points_in_cell <= target_window) || vertices.len() >= max_vertices {
            return;
        }

        // Find dimension with largest range
        let mut best_dim = 0;
        let mut best_range = T::zero();
        for d in 0..dimensions {
            let range = cell.upper[d] - cell.lower[d];
            if range > best_range {
                best_range = range;
                best_dim = d;
            }
        }

        // Split at midpoint
        let split_val = (cell.lower[best_dim] + cell.upper[best_dim]) / T::from(2.0).unwrap();

        // Create child cells
        let left_lower = cell.lower.clone();
        let mut left_upper = cell.upper.clone();
        left_upper[best_dim] = split_val;

        let mut right_lower = cell.lower.clone();
        let right_upper = cell.upper.clone();
        right_lower[best_dim] = split_val;

        // Create vertices for new split plane
        let num_new_vertices = 1usize << (dimensions - 1); // 2^(d-1) new vertices on split plane
        let start_vertex_idx = vertices.len();

        for corner_idx in 0..num_new_vertices {
            let mut vertex = vec![T::zero(); dimensions];
            vertex[best_dim] = split_val;

            let mut bit_pos = 0;
            for (d, v_d) in vertex.iter_mut().enumerate().take(dimensions) {
                if d != best_dim {
                    if (corner_idx >> bit_pos) & 1 == 0 {
                        *v_d = cell.lower[d];
                    } else {
                        *v_d = cell.upper[d];
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

        let parent_vertices = &cells[cell_idx].vertex_indices;

        for child_corner_idx in 0..num_corners {
            let dim_bit = (child_corner_idx >> best_dim) & 1;

            // Calculate "compressed" index (skipping best_dim) for new vertices
            let mask = (1 << best_dim) - 1;
            let lower_bits = child_corner_idx & mask;
            let upper_bits = child_corner_idx >> (best_dim + 1);
            let compressed_idx = (upper_bits << best_dim) | lower_bits;

            // Left Child
            if dim_bit == 0 {
                // Comes from parent (same index because bit is 0)
                left_vertices[child_corner_idx] = parent_vertices[child_corner_idx];
            } else {
                // Comes from split (new vertex)
                left_vertices[child_corner_idx] = start_vertex_idx + compressed_idx;
            }

            // Right Child
            if dim_bit == 0 {
                // Comes from split (new vertex)
                right_vertices[child_corner_idx] = start_vertex_idx + compressed_idx;
            } else {
                // Comes from parent (same index because bit is 1)
                right_vertices[child_corner_idx] = parent_vertices[child_corner_idx];
            }
        }

        let left_idx = cells.len();
        let right_idx = cells.len() + 1;

        cells.push(SurfaceCell {
            lower: left_lower,
            upper: left_upper,
            vertex_indices: left_vertices,
            children: None,
            split_dim: None,
            split_val: None,
        });

        cells.push(SurfaceCell {
            lower: right_lower,
            upper: right_upper,
            vertex_indices: right_vertices,
            children: None,
            split_dim: None,
            split_val: None,
        });

        // Update parent
        cells[cell_idx].children = Some((left_idx, right_idx));
        cells[cell_idx].split_dim = Some(best_dim);
        cells[cell_idx].split_val = Some(split_val);

        // Recurse
        Self::subdivide_cells(
            cells,
            vertices,
            left_idx,
            x,
            dimensions,
            max_vertices,
            target_window,
            thresholds,
        );
        Self::subdivide_cells(
            cells,
            vertices,
            right_idx,
            x,
            dimensions,
            max_vertices,
            target_window,
            thresholds,
        );
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
                None => {
                    // Leaf cell
                    return current;
                }
            }
        }
    }

    /// Perform multilinear interpolation within a cell.
    fn interpolate_in_cell(&self, cell: &SurfaceCell<T>, query: &[T]) -> T {
        let d = self.dimensions;

        // For each dimension, compute the interpolation weight
        let mut weights = Vec::with_capacity(d);
        for (dim, item) in query.iter().enumerate().take(d) {
            let range = cell.upper[dim] - cell.lower[dim];
            if range > T::epsilon() {
                let t = (*item - cell.lower[dim]) / range;
                // Clamp to [0, 1] for points slightly outside
                let t = t.max(T::zero()).min(T::one());
                weights.push(t);
            } else {
                weights.push(T::zero());
            }
        }

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
