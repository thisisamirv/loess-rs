//! KD-Tree for efficient k-nearest neighbor search in nD space.
//!
//! ## Purpose
//!
//! This module implements a KD-tree to optimize nD neighborhood searches.
//! By organizing points in a spatial hierarchy, we can reduce the search
//! time from O(n) to approximately O(log n) per query.
//!
//! ## Design notes
//!
//! * **Static Construction**: The tree is built once and then used for queries.
//! * **Flattened Representation**: Tree nodes are stored in a linear vector.
//! * **Trait-based Distance**: Supports generic distance metrics.
//!
//! ## Key concepts
//!
//! * **Splitting Plane**: The dimension and value used to split points at each node.
//! * **Pruning**: Skipping branches that cannot possibly contain nearer neighbors.
//!
//! ## Invariants
//!
//! * Tree depth is bound by O(log n).
//! * Queries always return the exact nearest neighbors (no approximation).
//!
//! ## Non-goals
//!
//! * This module does not support dynamic insertions or deletions.
//! * This module does not support approximate nearest neighbor search.

// Feature-gated imports
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

// External dependencies
use core::cmp::Ordering::Equal;
use num_traits::Float;

// ============================================================================
// Distance Trait
// ============================================================================

/// Trait for distance calculations used in KD-tree search.
pub trait PointDistance<T: Float> {
    /// Compute distance between two points.
    fn distance(&self, a: &[T], b: &[T]) -> T;

    /// Compute distance along a single dimension (for pruning).
    fn split_distance(&self, dim: usize, split_val: T, query_val: T) -> T;
}

// ============================================================================
// Neighborhood Structure
// ============================================================================

/// Result of k-nearest neighbor search in nD space.
#[derive(Debug, Clone)]
pub struct Neighborhood<T> {
    /// Indices of the k nearest neighbors (sorted by distance, ascending)
    pub indices: Vec<usize>,

    /// Distances to each neighbor (same order as indices)
    pub distances: Vec<T>,

    /// Maximum distance in the neighborhood (bandwidth)
    pub max_distance: T,
}

impl<T: Float> Neighborhood<T> {
    /// Create a new empty neighborhood.
    pub fn new() -> Self {
        Self {
            indices: Vec::new(),
            distances: Vec::new(),
            max_distance: T::zero(),
        }
    }

    /// Number of neighbors in this neighborhood.
    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if the neighborhood is empty.
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

impl<T: Float> Default for Neighborhood<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// KD-Tree Implementation
// ============================================================================

/// A node in the KD-tree.
#[derive(Debug, Clone)]
struct KDNode<T: Float> {
    /// Index of the point in the original flattened data array.
    index: usize,
    /// Left child index in the nodes vector.
    left: Option<usize>,
    /// Right child index in the nodes vector.
    right: Option<usize>,
    /// Splitting dimension.
    split_dim: usize,
    /// Point coordinates (cached for faster distance checks).
    point: Vec<T>,
}

/// KD-tree for spatial indexing of nD points.
#[derive(Debug, Clone)]
pub struct KDTree<T: Float> {
    nodes: Vec<KDNode<T>>,
    root: Option<usize>,
}

impl<T: Float> KDTree<T> {
    /// Build a KD-tree from a flattened data array.
    pub fn new(points: &[T], dimensions: usize) -> Self {
        let n = points.len() / dimensions;
        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(n);

        let root = Self::build_recursive(points, dimensions, &mut indices, 0, &mut nodes);

        Self { nodes, root }
    }

    /// Find k-nearest neighbors using the KD-tree.
    ///
    /// Uses the tree structure for efficient pruning during search.
    pub fn find_k_nearest<D: PointDistance<T>>(
        &self,
        query: &[T],
        k: usize,
        dist_calc: &D,
        exclude_self: Option<usize>,
    ) -> Neighborhood<T> {
        if k == 0 || self.root.is_none() {
            return Neighborhood::new();
        }

        let mut heap = Vec::with_capacity(k);
        let root_idx = self.root.unwrap();

        self.search_recursive(root_idx, query, k, dist_calc, exclude_self, &mut heap);

        // Sort by distance ascending
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Equal));

        Neighborhood {
            indices: heap.iter().map(|h| h.0).collect(),
            distances: heap.iter().map(|h| h.1).collect(),
            max_distance: heap.last().map(|h| h.1).unwrap_or(T::zero()),
        }
    }

    fn build_recursive(
        points: &[T],
        dims: usize,
        indices: &mut [usize],
        depth: usize,
        nodes: &mut Vec<KDNode<T>>,
    ) -> Option<usize> {
        if indices.is_empty() {
            return None;
        }

        let axis = depth % dims;

        // Partial sort to find median
        indices.sort_by(|&a, &b| {
            points[a * dims + axis]
                .partial_cmp(&points[b * dims + axis])
                .unwrap_or(Equal)
        });

        let median_idx = indices.len() / 2;
        let point_idx = indices[median_idx];

        let (left_indices, right_indices_with_median) = indices.split_at_mut(median_idx);
        let right_indices = &mut right_indices_with_median[1..];

        // Placeholder for current node to get its index
        let current_node_idx = nodes.len();

        let point_data = points[point_idx * dims..(point_idx + 1) * dims].to_vec();

        nodes.push(KDNode {
            index: point_idx,
            left: None,
            right: None,
            split_dim: axis,
            point: point_data,
        });

        let left = Self::build_recursive(points, dims, left_indices, depth + 1, nodes);
        let right = Self::build_recursive(points, dims, right_indices, depth + 1, nodes);

        // Update the node we just pushed
        nodes[current_node_idx].left = left;
        nodes[current_node_idx].right = right;

        Some(current_node_idx)
    }

    #[allow(clippy::too_many_arguments)]
    fn search_recursive<D: PointDistance<T>>(
        &self,
        node_idx: usize,
        query: &[T],
        k: usize,
        dist_calc: &D,
        exclude_self: Option<usize>,
        heap: &mut Vec<(usize, T)>,
    ) {
        let node = &self.nodes[node_idx];

        // Use injected distance calculator
        let dist = dist_calc.distance(&node.point, query);

        if exclude_self != Some(node.index) {
            if heap.len() < k {
                heap.push((node.index, dist));
                heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Equal));
            // Max-heap
            } else if dist < heap[0].1 {
                heap[0] = (node.index, dist);
                heap.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Equal));
            }
        }

        let axis = node.split_dim;
        let scaled_diff = dist_calc.split_distance(axis, node.point[axis], query[axis]);
        let diff = query[axis] - node.point[axis];

        let (nearer, farther) = if diff <= T::zero() {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        if let Some(next_idx) = nearer {
            self.search_recursive(next_idx, query, k, dist_calc, exclude_self, heap);
        }

        if let Some(next_idx) = farther {
            // Can a point in the farther subtree be closer than the current worst in heap?
            let can_improve = heap.len() < k || scaled_diff < heap[0].1;
            if can_improve {
                self.search_recursive(next_idx, query, k, dist_calc, exclude_self, heap);
            }
        }
    }
}
