//! Layer 2: Math
//!
//! # Purpose
//!
//! This layer provides pure mathematical functions used throughout LOESS:
//! - Kernel functions for distance-based weighting
//! - Robust statistics (MAD)
//!
//! These are reusable mathematical building blocks with no algorithm-specific logic.
//!
//! # Architecture
//!
//! ```text
//! Layer 7: API
//!   ↓
//! Layer 6: Adapters
//!   ↓
//! Layer 5: Engine
//!   ↓
//! Layer 4: Evaluation
//!   ↓
//! Layer 3: Algorithms
//!   ↓
//! Layer 2: Math ← You are here
//!   ↓
//! Layer 1: Primitives
//! ```

/// Kernel (weight) functions for distance-based weighting.
pub mod kernel;

/// Median Absolute Deviation (MAD) computation.
pub mod mad;

/// Boundary padding utilities.
pub mod boundary;
