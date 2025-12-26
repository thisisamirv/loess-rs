#![cfg(feature = "dev")]
//! Tests for partition configuration types.
//!
//! These tests verify the enum types used for configuring streaming and online adapters:
//! - BoundaryPolicy
//! - MergeStrategy  
//! - UpdateMode
//!
//! ## Test Organization
//!
//! 1. **Trait Implementations** - Debug, Clone, Copy, PartialEq, Eq, Default
//! 2. **Enum Variants** - All variants are accessible and distinct
//! 3. **Default Values** - Verify correct defaults

use loess_rs::internals::primitives::partition::{BoundaryPolicy, MergeStrategy, UpdateMode};

// ============================================================================
// BoundaryPolicy Tests
// ============================================================================

/// Test BoundaryPolicy default.
#[test]
fn test_boundary_policy_default() {
    assert_eq!(BoundaryPolicy::default(), BoundaryPolicy::Extend);
}

/// Test BoundaryPolicy variants are distinct.
#[test]
fn test_boundary_policy_variants() {
    let extend = BoundaryPolicy::Extend;
    let reflect = BoundaryPolicy::Reflect;
    let zero = BoundaryPolicy::Zero;

    assert_ne!(extend, reflect);
    assert_ne!(extend, zero);
    assert_ne!(reflect, zero);
}

/// Test BoundaryPolicy Clone and Copy.
#[test]
fn test_boundary_policy_clone_copy() {
    let policy = BoundaryPolicy::Reflect;
    let cloned = policy;
    let copied = policy;

    assert_eq!(policy, cloned);
    assert_eq!(policy, copied);
}

/// Test BoundaryPolicy Debug.
#[test]
fn test_boundary_policy_debug() {
    let policy = BoundaryPolicy::Extend;
    let debug_str = format!("{:?}", policy);
    assert!(debug_str.contains("Extend"));
}

// ============================================================================
// MergeStrategy Tests
// ============================================================================

/// Test MergeStrategy default.
#[test]
fn test_merge_strategy_default() {
    assert_eq!(MergeStrategy::default(), MergeStrategy::WeightedAverage);
}

/// Test MergeStrategy variants are distinct.
#[test]
fn test_merge_strategy_variants() {
    let average = MergeStrategy::Average;
    let weighted = MergeStrategy::WeightedAverage;
    let first = MergeStrategy::TakeFirst;
    let last = MergeStrategy::TakeLast;

    assert_ne!(average, weighted);
    assert_ne!(average, first);
    assert_ne!(average, last);
    assert_ne!(weighted, first);
    assert_ne!(weighted, last);
    assert_ne!(first, last);
}

/// Test MergeStrategy Clone and Copy.
#[test]
fn test_merge_strategy_clone_copy() {
    let strategy = MergeStrategy::Average;
    let cloned = strategy;
    let copied = strategy;

    assert_eq!(strategy, cloned);
    assert_eq!(strategy, copied);
}

/// Test MergeStrategy Debug.
#[test]
fn test_merge_strategy_debug() {
    let strategy = MergeStrategy::WeightedAverage;
    let debug_str = format!("{:?}", strategy);
    assert!(debug_str.contains("WeightedAverage"));
}

// ============================================================================
// UpdateMode Tests
// ============================================================================

/// Test UpdateMode default.
#[test]
fn test_update_mode_default() {
    assert_eq!(UpdateMode::default(), UpdateMode::Incremental);
}

/// Test UpdateMode variants are distinct.
#[test]
fn test_update_mode_variants() {
    let full = UpdateMode::Full;
    let incremental = UpdateMode::Incremental;

    assert_ne!(full, incremental);
}

/// Test UpdateMode Clone and Copy.
#[test]
fn test_update_mode_clone_copy() {
    let mode = UpdateMode::Full;
    let cloned = mode;
    let copied = mode;

    assert_eq!(mode, cloned);
    assert_eq!(mode, copied);
}

/// Test UpdateMode Debug.
#[test]
fn test_update_mode_debug() {
    let mode = UpdateMode::Incremental;
    let debug_str = format!("{:?}", mode);
    assert!(debug_str.contains("Incremental"));
}

// ============================================================================
// Cross-Enum Tests
// ============================================================================

/// Test that all enums can be used together.
#[test]
fn test_enum_combination() {
    let boundary = BoundaryPolicy::Reflect;
    let merge = MergeStrategy::TakeFirst;
    let update = UpdateMode::Full;

    // Should compile and work together
    let config = (boundary, merge, update);
    assert_eq!(config.0, BoundaryPolicy::Reflect);
    assert_eq!(config.1, MergeStrategy::TakeFirst);
    assert_eq!(config.2, UpdateMode::Full);
}

/// Test that all defaults are reasonable.
#[test]
fn test_all_defaults() {
    // Verify documented defaults
    assert_eq!(BoundaryPolicy::default(), BoundaryPolicy::Extend);
    assert_eq!(MergeStrategy::default(), MergeStrategy::WeightedAverage);
    assert_eq!(UpdateMode::default(), UpdateMode::Incremental);
}
