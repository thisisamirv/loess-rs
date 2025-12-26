//! Configuration types for chunked and incremental data processing.
//!
//! ## Purpose
//!
//! This module defines configuration types used by the streaming and online
//! adapters to control boundary handling, overlap merging, and update modes.
//!
//! ## Design notes
//!
//! * **Encapsulated**: Configuration is captured in simple enums with clear defaults.
//! * **Trait Implementation**: All types implement `Debug`, `Clone`, `Copy`, `PartialEq`, and `Eq`.
//! * **Re-exported**: Accessible via the crate root and the `api` module.
//!
//! ## Key concepts
//!
//! 1. **Boundary Handling**: Extended, Reflected, or Zero padding.
//! 2. **Merging**: Averaging or selecting values for overlapping chunks.
//! 3. **Incremental Updates**: Full or incremental window processing.
//!
//! ## Invariants
//!
//! * Defaults represent the most robust and widely applicable settings.
//! * Policies are validated at builder construction or adapter initialization.
//!
//! ## Non-goals
//!
//! * This module does not implement the partitioning logic itself (handled by `adapters`).

/// Policy for handling boundaries at the start and end of a data stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryPolicy {
    /// Replicate edge values to provide context for boundary points.
    #[default]
    Extend,

    /// Mirror values across the boundary.
    Reflect,

    /// Use zero padding beyond data boundaries.
    Zero,
}

/// Strategy for merging overlapping regions between streaming chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MergeStrategy {
    /// Arithmetic mean of overlapping smoothed values: `(v1 + v2) / 2`.
    Average,

    /// Distance-based weights that favor values from the center of each chunk:
    /// v1 * (1 - alpha) + v2 * alpha where `alpha` is the relative position within the overlap.
    #[default]
    WeightedAverage,

    /// Use the value from the first chunk in processing order.
    TakeFirst,

    /// Use the value from the last chunk in processing order.
    TakeLast,
}

/// Update mode for online LOESS processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpdateMode {
    /// Recompute all points in the window from scratch.
    Full,

    /// Optimized incremental update.
    #[default]
    Incremental,
}
