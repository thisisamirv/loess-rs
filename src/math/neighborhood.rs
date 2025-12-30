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
use core::cmp::Ordering::{self, Equal};
use num_traits::Float;

// Internal dependencies
use crate::primitives::buffer::{NeighborhoodSearchBuffer, NeighborhoodStorage};

// ============================================================================
// Selection Algorithm (Floyd-Rivest)
// ============================================================================

/// Selects the k-th smallest element in `arr` and partitions the array around it.
///
/// Post-condition:
/// - `arr[k]` contains the k-th smallest element.
/// - All elements `arr[0..k]` are <= `arr[k]`.
/// - All elements `arr[k+1..]` are >= `arr[k]`.
pub fn floyd_rivest_select<T, F>(arr: &mut [T], k: usize, mut cmp: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    assert!(k < arr.len(), "k must be less than arr.len()");
    if arr.is_empty() {
        return;
    }
    floyd_rivest_recursive(arr, 0, arr.len() - 1, k, &mut cmp);
}

/// Recursive helper for Floyd-Rivest selection.
///
/// Adapted from CACM Algorithm 489.
fn floyd_rivest_recursive<T, F>(
    arr: &mut [T],
    mut left: usize,
    mut right: usize,
    k: usize,
    cmp: &mut F,
) where
    F: FnMut(&T, &T) -> Ordering,
{
    while right > left {
        // Use sampling heuristic for large ranges to pick a better pivot
        if right - left > 600 {
            let n = (right - left + 1) as f64;
            let i = (k - left + 1) as f64;
            let z = n.ln();
            let s = 0.5 * (2.0 * z / 3.0).exp();
            let sign = if i - n / 2.0 < 0.0 { -1.0 } else { 1.0 };
            let sd = 0.5 * (z * s * (n - s) / n).sqrt() * sign;

            let new_left = (left as f64).max((k as f64) - i * s / n + sd) as usize;
            let new_right = (right as f64).min((k as f64) + (n - i) * s / n + sd) as usize;

            floyd_rivest_recursive(arr, new_left, new_right, k, cmp);
        }

        // Partition around arr[k] (which was updated by recursive call if sampled)
        let t_idx = k;

        // Swap pivot to left to start partition
        arr.swap(left, t_idx);

        // Ensure arr[left] <= arr[right] to simplify partition
        if cmp(&arr[right], &arr[left]) == Ordering::Less {
            arr.swap(left, right);
        }

        if arr[left + 1..=right - 1].is_empty() {
            // Only 2 elements, and we just sorted them. simple case done.
        }

        // Hoare-like partition
        let mut i_ptr = left + 1;
        let mut j_ptr = right - 1;

        loop {
            // Find element >= pivot from left
            while i_ptr <= j_ptr && cmp(&arr[i_ptr], &arr[left]) == Ordering::Less {
                i_ptr += 1;
            }
            // Find element <= pivot from right
            while j_ptr >= i_ptr && cmp(&arr[j_ptr], &arr[left]) == Ordering::Greater {
                j_ptr -= 1;
            }

            if i_ptr >= j_ptr {
                break;
            }

            arr.swap(i_ptr, j_ptr);
            i_ptr += 1;
            // j_ptr might underflow if usize, but logic guarantees j_ptr >= i_ptr start
            j_ptr = j_ptr.saturating_sub(1);
        }

        // Swap pivot into correct place (j_ptr)
        arr.swap(left, j_ptr);
        let pivot_pos = j_ptr;

        // Adjust bounds
        if k <= pivot_pos {
            if pivot_pos == 0 {
                break;
            } // prevent underflow
            right = pivot_pos - 1;
        }
        if k >= pivot_pos {
            left = pivot_pos + 1;
        }
    }
}

// ============================================================================
// Helper Structures
// ============================================================================

/// Helper structure for max-heap in KD-tree search.
/// Orders by distance (the second field).
#[derive(Debug, Clone, Copy)]
pub struct NodeDistance<T>(pub usize, pub T);

impl<T: PartialEq> PartialEq for NodeDistance<T> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<T: PartialEq> Eq for NodeDistance<T> {}

impl<T: PartialOrd> PartialOrd for NodeDistance<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Ord for NodeDistance<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.partial_cmp(&other.1).unwrap_or(Equal)
    }
}

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

    /// Pre-allocate buffers for a neighborhood of size k.
    pub fn with_capacity(k: usize) -> Self {
        Self {
            indices: Vec::with_capacity(k),
            distances: Vec::with_capacity(k),
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

impl<T: Float> NeighborhoodStorage for Neighborhood<T> {
    fn with_capacity(k: usize) -> Self {
        Self::with_capacity(k)
    }

    fn capacity(&self) -> usize {
        self.indices.capacity()
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

/// Sentinel value indicating no child node.
const NO_CHILD: usize = usize::MAX;

/// A node in the KD-tree.
/// Uses sentinel values instead of Option for better performance.
#[derive(Debug, Clone, Copy)]
struct KDNode<T: Float> {
    /// Index of the point in the original flattened data array.
    index: usize,
    /// Left child index (NO_CHILD if none).
    left: usize,
    /// Right child index (NO_CHILD if none).
    right: usize,
    /// Splitting dimension.
    split_dim: u8,
    /// Splitting value at this node.
    split_val: T,
}

/// KD-tree for spatial indexing of nD points.
#[derive(Debug, Clone)]
pub struct KDTree<T: Float> {
    nodes: Vec<KDNode<T>>,
    points: Vec<T>,
    dimensions: usize,
    root: usize,
}

impl<T: Float> KDTree<T> {
    /// Build a KD-tree from a flattened data array.
    pub fn new(points: &[T], dimensions: usize) -> Self {
        let n = points.len() / dimensions;
        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(n);

        let root = Self::build_recursive(points, dimensions, &mut indices, 0, &mut nodes);

        Self {
            nodes,
            points: points.to_vec(),
            dimensions,
            root,
        }
    }

    /// Optimized search that reuses an existing NeighborhoodSearchBuffer and Neighborhood.
    pub fn find_k_nearest<D: PointDistance<T>>(
        &self,
        query: &[T],
        k: usize,
        dist_calc: &D,
        exclude_self: Option<usize>,
        buffer: &mut NeighborhoodSearchBuffer<NodeDistance<T>>,
        neighborhood: &mut Neighborhood<T>,
    ) {
        if k == 0 || self.root == NO_CHILD {
            neighborhood.max_distance = T::zero();
            neighborhood.indices.clear();
            neighborhood.distances.clear();
            return;
        }

        buffer.clear();
        self.search_iterative(query, k, dist_calc, exclude_self, buffer);

        // Copy to sort vector
        buffer.sort_vec.clear();
        for &nd in buffer.heap.iter() {
            buffer.sort_vec.push(nd);
        }

        // Update neighborhood in-place
        neighborhood.indices.clear();
        neighborhood.distances.clear();
        for nd in &buffer.sort_vec {
            neighborhood.indices.push(nd.0);
            neighborhood.distances.push(nd.1);
        }
        // Since heap is a MaxHeap, the largest distance (bandwidth) is at the top
        neighborhood.max_distance = buffer.heap.peek().map(|nd| nd.1).unwrap_or(T::zero());
    }

    fn build_recursive(
        points: &[T],
        dims: usize,
        indices: &mut [usize],
        depth: usize,
        nodes: &mut Vec<KDNode<T>>,
    ) -> usize {
        if indices.is_empty() {
            return NO_CHILD;
        }

        let axis = depth % dims;

        // Floyd-Rivest selection (O(N)) to find median and partition
        let median_idx = indices.len() / 2;
        floyd_rivest_select(indices, median_idx, |&a, &b| {
            points[a * dims + axis]
                .partial_cmp(&points[b * dims + axis])
                .unwrap_or(Equal)
        });

        let median_idx = indices.len() / 2;
        let point_idx = indices[median_idx];
        let split_val = points[point_idx * dims + axis];

        let (left_indices, right_indices_with_median) = indices.split_at_mut(median_idx);
        let right_indices = &mut right_indices_with_median[1..];

        // Placeholder for current node
        let current_node_idx = nodes.len();

        nodes.push(KDNode {
            index: point_idx,
            left: NO_CHILD,
            right: NO_CHILD,
            split_dim: axis as u8,
            split_val,
        });

        let left = Self::build_recursive(points, dims, left_indices, depth + 1, nodes);
        let right = Self::build_recursive(points, dims, right_indices, depth + 1, nodes);

        // Update the node we just pushed
        nodes[current_node_idx].left = left;
        nodes[current_node_idx].right = right;

        current_node_idx
    }

    /// Iterative k-nearest neighbor search using explicit stack.
    /// This avoids function call overhead and improves cache locality.
    #[inline]
    fn search_iterative<D: PointDistance<T>>(
        &self,
        query: &[T],
        k: usize,
        dist_calc: &D,
        exclude_self: Option<usize>,
        buffer: &mut NeighborhoodSearchBuffer<NodeDistance<T>>,
    ) {
        let d = self.dimensions;
        let heap = &mut buffer.heap;
        let stack = &mut buffer.stack;

        // Track max distance
        let mut max_dist = T::infinity();

        stack.push(self.root);

        while let Some(node_idx) = stack.pop() {
            let node = &self.nodes[node_idx];

            // Process current node
            if exclude_self != Some(node.index) {
                let offset = node.index * d;
                let node_point = &self.points[offset..offset + d];
                let dist = dist_calc.distance(query, node_point);

                if heap.len() < k {
                    heap.push(NodeDistance(node.index, dist));
                    if heap.len() == k {
                        // Heap is now full, cache the max distance
                        max_dist = heap.peek().map(|nd| nd.1).unwrap_or(T::infinity());
                    }
                } else if dist < max_dist {
                    // Replace the worst neighbor
                    if let Some(mut top) = heap.peek_mut() {
                        *top = NodeDistance(node.index, dist);
                    }
                    // Update cached max distance
                    max_dist = heap.peek().map(|nd| nd.1).unwrap_or(T::infinity());
                }
            }

            // Determine near/far children
            let split_dim = node.split_dim as usize;
            let diff = query[split_dim] - node.split_val;

            let (near, far) = if diff <= T::zero() {
                (node.left, node.right)
            } else {
                (node.right, node.left)
            };

            // Check if we need to explore the far subtree
            // We push far first so near is processed first (LIFO)
            if far != NO_CHILD {
                let dist_to_plane =
                    dist_calc.split_distance(split_dim, node.split_val, query[split_dim]);
                // Only explore far if heap is not full OR plane is closer than worst neighbor
                if heap.len() < k || dist_to_plane < max_dist {
                    stack.push(far);
                }
            }

            // Always explore near subtree if it exists
            if near != NO_CHILD {
                stack.push(near);
            }
        }
    }
}
