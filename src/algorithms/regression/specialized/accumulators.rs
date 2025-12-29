//! Specialized Accumulators
//!
//! ## Purpose
//!
//! This module provides optimized scalar and SIMD accumulation functions for building Normal Equations matrices in 1D/2D linear and quadratic regression.

// External dependencies
use num_traits::Float;
use wide::f64x2;

// ============================================================================
// Specialized Accumulation Functions
// ============================================================================

/// Optimized accumulation for 1D Linear Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_linear_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query: T,
    xtwx: &mut [T; 4],
    xtwy: &mut [T; 2],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_wy = T::zero();
    let mut s_wdxy = T::zero();

    for i in 0..n {
        let w = weights[i];
        if w <= T::epsilon() {
            continue;
        }
        let idx = indices[i];
        let dx = x[idx] - query;
        let y_val = y[idx];
        let wdx = w * dx;
        s_w = s_w + w;
        s_dx = s_dx + wdx;
        s_dx2 = s_dx2 + wdx * dx;
        s_wy = s_wy + w * y_val;
        s_wdxy = s_wdxy + wdx * y_val;
    }
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dx;
    xtwx[3] = s_dx2;
    xtwy[0] = s_wy;
    xtwy[1] = s_wdxy;
}

/// Optimized accumulation for 2D Linear Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_linear_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query_x: T,
    query_y: T,
    xtwx: &mut [T; 9],
    xtwy: &mut [T; 3],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dy = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_dy2 = T::zero();
    let mut s_dxdy = T::zero();
    let mut s_wy = T::zero();
    let mut s_wdxy = T::zero();
    let mut s_wdyy = T::zero();

    for i in 0..n {
        let w = weights[i];
        if w <= T::epsilon() {
            continue;
        }
        let idx = indices[i];
        let dx = x[idx * 2] - query_x;
        let dy = x[idx * 2 + 1] - query_y;
        let y_val = y[idx];
        let wdx = w * dx;
        let wdy = w * dy;
        s_w = s_w + w;
        s_dx = s_dx + wdx;
        s_dy = s_dy + wdy;
        s_dx2 = s_dx2 + wdx * dx;
        s_dy2 = s_dy2 + wdy * dy;
        s_dxdy = s_dxdy + wdx * dy;
        s_wy = s_wy + w * y_val;
        s_wdxy = s_wdxy + wdx * y_val;
        s_wdyy = s_wdyy + wdy * y_val;
    }
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dy;
    xtwx[3] = s_dx;
    xtwx[4] = s_dx2;
    xtwx[5] = s_dxdy;
    xtwx[6] = s_dy;
    xtwx[7] = s_dxdy;
    xtwx[8] = s_dy2;
    xtwy[0] = s_wy;
    xtwy[1] = s_wdxy;
    xtwy[2] = s_wdyy;
}

/// Optimized accumulation for 2D Quadratic Case (Scalar).
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_quadratic_scalar<T: Float>(
    x: &[T],
    y: &[T],
    indices: &[usize],
    weights: &[T],
    query_x: T,
    query_y: T,
    xtwx: &mut [T; 36],
    xtwy: &mut [T; 6],
) {
    let n = indices.len();
    let mut s_w = T::zero();
    let mut s_dx = T::zero();
    let mut s_dy = T::zero();
    let mut s_dx2 = T::zero();
    let mut s_dxdy = T::zero();
    let mut s_dy2 = T::zero();
    let mut s_dx3 = T::zero();
    let mut s_dx2dy = T::zero();
    let mut s_dxdy2 = T::zero();
    let mut s_dy3 = T::zero();
    let mut s_dx4 = T::zero();
    let mut s_dx3dy = T::zero();
    let mut s_dx2dy2 = T::zero();
    let mut s_dxdy3 = T::zero();
    let mut s_dy4 = T::zero();
    let mut s_wy = T::zero();
    let mut s_wyx = T::zero();
    let mut s_wyy = T::zero();
    let mut s_wyx2 = T::zero();
    let mut s_wyxy = T::zero();
    let mut s_wyy2 = T::zero();

    for i in 0..n {
        let w = weights[i];
        if w <= T::epsilon() {
            continue;
        }
        let idx = indices[i];
        let dx = x[idx * 2] - query_x;
        let dy = x[idx * 2 + 1] - query_y;
        let y_val = y[idx];
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let dxdy = dx * dy;
        let wdx = w * dx;
        let wdy = w * dy;
        let wdx2 = w * dx2;
        let wdy2 = w * dy2;
        let wdxdy = w * dxdy;

        s_w = s_w + w;
        s_dx = s_dx + wdx;
        s_dy = s_dy + wdy;
        s_dx2 = s_dx2 + wdx2;
        s_dxdy = s_dxdy + wdxdy;
        s_dy2 = s_dy2 + wdy2;
        s_dx3 = s_dx3 + wdx2 * dx;
        s_dx2dy = s_dx2dy + wdx2 * dy;
        s_dxdy2 = s_dxdy2 + wdy2 * dx;
        s_dy3 = s_dy3 + wdy2 * dy;
        s_dx4 = s_dx4 + wdx2 * dx2;
        s_dx3dy = s_dx3dy + wdx2 * dxdy;
        s_dx2dy2 = s_dx2dy2 + dx2 * wdy2;
        s_dxdy3 = s_dxdy3 + dy2 * wdxdy;
        s_dy4 = s_dy4 + wdy2 * dy2;
        s_wy = s_wy + w * y_val;
        s_wyx = s_wyx + wdx * y_val;
        s_wyy = s_wyy + wdy * y_val;
        s_wyx2 = s_wyx2 + wdx2 * y_val;
        s_wyxy = s_wyxy + wdxdy * y_val;
        s_wyy2 = s_wyy2 + wdy2 * y_val;
    }

    // Terms: 1, x, y, x^2, xy, y^2
    // Row 0
    xtwx[0] = s_w;
    xtwx[1] = s_dx;
    xtwx[2] = s_dy;
    xtwx[3] = s_dx2;
    xtwx[4] = s_dxdy;
    xtwx[5] = s_dy2;
    // Row 1
    xtwx[6] = s_dx;
    xtwx[7] = s_dx2;
    xtwx[8] = s_dxdy;
    xtwx[9] = s_dx3;
    xtwx[10] = s_dx2dy;
    xtwx[11] = s_dxdy2;
    // Row 2
    xtwx[12] = s_dy;
    xtwx[13] = s_dxdy;
    xtwx[14] = s_dy2;
    xtwx[15] = s_dx2dy;
    xtwx[16] = s_dxdy2;
    xtwx[17] = s_dy3;
    // Row 3
    xtwx[18] = s_dx2;
    xtwx[19] = s_dx3;
    xtwx[20] = s_dx2dy;
    xtwx[21] = s_dx4;
    xtwx[22] = s_dx3dy;
    xtwx[23] = s_dx2dy2;
    // Row 4
    xtwx[24] = s_dxdy;
    xtwx[25] = s_dx2dy;
    xtwx[26] = s_dxdy2;
    xtwx[27] = s_dx3dy;
    xtwx[28] = s_dx2dy2;
    xtwx[29] = s_dxdy3;
    // Row 5
    xtwx[30] = s_dy2;
    xtwx[31] = s_dxdy2;
    xtwx[32] = s_dy3;
    xtwx[33] = s_dx2dy2;
    xtwx[34] = s_dxdy3;
    xtwx[35] = s_dy4;

    xtwy[0] = s_wy;
    xtwy[1] = s_wyx;
    xtwy[2] = s_wyy;
    xtwy[3] = s_wyx2;
    xtwy[4] = s_wyxy;
    xtwy[5] = s_wyy2;
}

/// Optimized accumulation for 2D Quadratic Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_quadratic_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query_x: f64,
    query_y: f64,
    xtwx: &mut [f64; 36],
    xtwy: &mut [f64; 6],
) {
    let n = indices.len();
    let mut i = 0;

    // SIMD accumulators
    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dy = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_dxdy = f64x2::splat(0.0);
    let mut s_dy2 = f64x2::splat(0.0);
    let mut s_dx3 = f64x2::splat(0.0);
    let mut s_dx2dy = f64x2::splat(0.0);
    let mut s_dxdy2 = f64x2::splat(0.0);
    let mut s_dy3 = f64x2::splat(0.0);
    let mut s_dx4 = f64x2::splat(0.0);
    let mut s_dx3dy = f64x2::splat(0.0);
    let mut s_dx2dy2 = f64x2::splat(0.0);
    let mut s_dxdy3 = f64x2::splat(0.0);
    let mut s_dy4 = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wyx = f64x2::splat(0.0);
    let mut s_wyy = f64x2::splat(0.0);
    let mut s_wyx2 = f64x2::splat(0.0);
    let mut s_wyxy = f64x2::splat(0.0);
    let mut s_wyy2 = f64x2::splat(0.0);

    let qx = f64x2::splat(query_x);
    let qy = f64x2::splat(query_y);

    // Process 2 elements at a time (f64x2)
    while i + 2 <= n {
        let idx0 = indices[i];
        let idx1 = indices[i + 1];

        // Gather weights
        let w = f64x2::new([weights[i], weights[i + 1]]);

        // Gather x coordinates (requires generic gather or manual construction)
        // Since we don't have direct gather for f64x2 from &[f64] with indices in 'wide' easily,
        // we construct manually.
        let px = f64x2::new([x[idx0 * 2], x[idx1 * 2]]);
        let py = f64x2::new([x[idx0 * 2 + 1], x[idx1 * 2 + 1]]);
        let y_vals = f64x2::new([y[idx0], y[idx1]]);

        let dx = px - qx;
        let dy = py - qy;

        // Calculations
        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let dxdy = dx * dy;

        let wdx = w * dx;
        let wdy = w * dy;
        let wdx2 = w * dx2;
        let wdy2 = w * dy2;
        let wdxdy = w * dxdy;

        s_w += w;
        s_dx += wdx;
        s_dy += wdy;
        s_dx2 += wdx2;
        s_dxdy += wdxdy;
        s_dy2 += wdy2;

        // Optimization: dx4 = dx2 * dx2

        s_dx3 += wdx2 * dx;
        s_dx2dy += wdx2 * dy;
        s_dxdy2 += wdy2 * dx;
        s_dy3 += wdy2 * dy;

        s_dx4 += wdx2 * dx2;
        s_dx3dy += wdx2 * dxdy; // w * dx^2 * dx * dy = w * dx^3 * dy
        s_dx2dy2 += wdy2 * dy2;
        s_dxdy3 += wdy2 * dxdy;
        s_dy4 += wdy2 * dy2;

        s_wy += w * y_vals;
        s_wyx += wdx * y_vals;
        s_wyy += wdy * y_vals;
        s_wyx2 += wdx2 * y_vals;
        s_wyxy += wdxdy * y_vals;
        s_wyy2 += wdy2 * y_vals;

        i += 2;
    }

    // Reduce SIMD accumulators
    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dy = s_dy.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_dxdy = s_dxdy.reduce_add();
    let mut a_dy2 = s_dy2.reduce_add();
    let mut a_dx3 = s_dx3.reduce_add();
    let mut a_dx2dy = s_dx2dy.reduce_add();
    let mut a_dxdy2 = s_dxdy2.reduce_add();
    let mut a_dy3 = s_dy3.reduce_add();
    let mut a_dx4 = s_dx4.reduce_add();
    let mut a_dx3dy = s_dx3dy.reduce_add();
    let mut a_dx2dy2 = s_dx2dy2.reduce_add();
    let mut a_dxdy3 = s_dxdy3.reduce_add();
    let mut a_dy4 = s_dy4.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wyx = s_wyx.reduce_add();
    let mut a_wyy = s_wyy.reduce_add();
    let mut a_wyx2 = s_wyx2.reduce_add();
    let mut a_wyxy = s_wyxy.reduce_add();
    let mut a_wyy2 = s_wyy2.reduce_add();

    // Handle tail
    for k in i..n {
        let w = weights[k];
        if w <= f64::EPSILON {
            continue;
        }

        let idx = indices[k];
        let dx = x[idx * 2] - query_x;
        let dy = x[idx * 2 + 1] - query_y;
        let y_val = y[idx];

        let dx2 = dx * dx;
        let dy2 = dy * dy;
        let dxdy = dx * dy;

        let wdx = w * dx;
        let wdy = w * dy;
        let wdx2 = w * dx2;
        let wdy2 = w * dy2;
        let wdxdy = w * dxdy;

        a_w += w;
        a_dx += wdx;
        a_dy += wdy;
        a_dx2 += wdx2;
        a_dxdy += wdxdy;
        a_dy2 += wdy2;

        a_dx3 += wdx2 * dx;
        a_dx2dy += wdx2 * dy;
        a_dxdy2 += wdy2 * dx;
        a_dy3 += wdy2 * dy;

        a_dx4 += wdx2 * dx2;
        a_dx3dy += wdx2 * dxdy;
        a_dx2dy2 += wdy2 * dx2; // Corrected: dx^2 * dy^2
        a_dxdy3 += wdy2 * dxdy;
        a_dy4 += wdy2 * dy2;

        a_wy += w * y_val;
        a_wyx += wdx * y_val;
        a_wyy += wdy * y_val;
        a_wyx2 += wdx2 * y_val;
        a_wyxy += wdxdy * y_val;
        a_wyy2 += wdy2 * y_val;
    }

    // Write to output buffers
    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dy;
    xtwx[3] = a_dx2;
    xtwx[4] = a_dxdy;
    xtwx[5] = a_dy2;

    xtwx[6] = a_dx;
    xtwx[7] = a_dx2;
    xtwx[8] = a_dxdy;
    xtwx[9] = a_dx3;
    xtwx[10] = a_dx2dy;
    xtwx[11] = a_dxdy2;

    xtwx[12] = a_dy;
    xtwx[13] = a_dxdy;
    xtwx[14] = a_dy2;
    xtwx[15] = a_dx2dy;
    xtwx[16] = a_dxdy2;
    xtwx[17] = a_dy3;

    xtwx[18] = a_dx2;
    xtwx[19] = a_dx3;
    xtwx[20] = a_dx2dy;
    xtwx[21] = a_dx4;
    xtwx[22] = a_dx3dy;
    xtwx[23] = a_dx2dy2;

    xtwx[24] = a_dxdy;
    xtwx[25] = a_dx2dy;
    xtwx[26] = a_dxdy2;
    xtwx[27] = a_dx3dy;
    xtwx[28] = a_dx2dy2;
    xtwx[29] = a_dxdy3;

    xtwx[30] = a_dy2;
    xtwx[31] = a_dxdy2;
    xtwx[32] = a_dy3;
    xtwx[33] = a_dx2dy2;
    xtwx[34] = a_dxdy3;
    xtwx[35] = a_dy4;

    xtwy[0] = a_wy;
    xtwy[1] = a_wyx;
    xtwy[2] = a_wyy;
    xtwy[3] = a_wyx2;
    xtwy[4] = a_wyxy;
    xtwy[5] = a_wyy2;
}

/// Optimized accumulation for 1D Linear Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_1d_linear_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query: f64,
    xtwx: &mut [f64; 4],
    xtwy: &mut [f64; 2],
) {
    let n = indices.len();
    let mut i = 0;

    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wdxy = f64x2::splat(0.0);

    let q = f64x2::splat(query);

    while i + 2 <= n {
        let idx0 = indices[i];
        let idx1 = indices[i + 1];

        let w = f64x2::new([weights[i], weights[i + 1]]);
        let x_val = f64x2::new([x[idx0], x[idx1]]);
        let y_val = f64x2::new([y[idx0], y[idx1]]);

        let dx = x_val - q;
        let wdx = w * dx;

        s_w += w;
        s_dx += wdx;
        s_dx2 += wdx * dx;
        s_wy += w * y_val;
        s_wdxy += wdx * y_val;

        i += 2;
    }

    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wdxy = s_wdxy.reduce_add();

    // Tail
    for k in i..n {
        let w = weights[k];
        if w <= f64::EPSILON {
            continue;
        }

        let idx = indices[k];
        let dx = x[idx] - query;
        let y_val = y[idx];
        let wdx = w * dx;

        a_w += w;
        a_dx += wdx;
        a_dx2 += wdx * dx;
        a_wy += w * y_val;
        a_wdxy += wdx * y_val;
    }

    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dx;
    xtwx[3] = a_dx2;
    xtwy[0] = a_wy;
    xtwy[1] = a_wdxy;
}

/// Optimized accumulation for 2D Linear Case using SIMD.
#[allow(clippy::too_many_arguments)]
pub fn accumulate_2d_linear_simd(
    x: &[f64],
    y: &[f64],
    indices: &[usize],
    weights: &[f64],
    query_x: f64,
    query_y: f64,
    xtwx: &mut [f64; 9],
    xtwy: &mut [f64; 3],
) {
    let n = indices.len();
    let mut i = 0;

    let mut s_w = f64x2::splat(0.0);
    let mut s_dx = f64x2::splat(0.0);
    let mut s_dy = f64x2::splat(0.0);
    let mut s_dx2 = f64x2::splat(0.0);
    let mut s_dy2 = f64x2::splat(0.0);
    let mut s_dxdy = f64x2::splat(0.0);
    let mut s_wy = f64x2::splat(0.0);
    let mut s_wdxy = f64x2::splat(0.0);
    let mut s_wdyy = f64x2::splat(0.0);

    let qx = f64x2::splat(query_x);
    let qy = f64x2::splat(query_y);

    while i + 2 <= n {
        let idx0 = indices[i];
        let idx1 = indices[i + 1];

        let w = f64x2::new([weights[i], weights[i + 1]]);
        let px = f64x2::new([x[idx0 * 2], x[idx1 * 2]]);
        let py = f64x2::new([x[idx0 * 2 + 1], x[idx1 * 2 + 1]]);
        let y_vals = f64x2::new([y[idx0], y[idx1]]);

        let dx = px - qx;
        let dy = py - qy;

        let wdx = w * dx;
        let wdy = w * dy;

        s_w += w;
        s_dx += wdx;
        s_dy += wdy;
        s_dx2 += wdx * dx;
        s_dy2 += wdy * dy;
        s_dxdy += wdx * dy;
        s_wy += w * y_vals;
        s_wdxy += wdx * y_vals;
        s_wdyy += wdy * y_vals;

        i += 2;
    }

    let mut a_w = s_w.reduce_add();
    let mut a_dx = s_dx.reduce_add();
    let mut a_dy = s_dy.reduce_add();
    let mut a_dx2 = s_dx2.reduce_add();
    let mut a_dy2 = s_dy2.reduce_add();
    let mut a_dxdy = s_dxdy.reduce_add();
    let mut a_wy = s_wy.reduce_add();
    let mut a_wdxy = s_wdxy.reduce_add();
    let mut a_wdyy = s_wdyy.reduce_add();

    for k in i..n {
        let w = weights[k];
        if w <= f64::EPSILON {
            continue;
        }

        let idx = indices[k];
        let dx = x[idx * 2] - query_x;
        let dy = x[idx * 2 + 1] - query_y;
        let y_val = y[idx];

        let wdx = w * dx;
        let wdy = w * dy;

        a_w += w;
        a_dx += wdx;
        a_dy += wdy;
        a_dx2 += wdx * dx;
        a_dy2 += wdy * dy;
        a_dxdy += wdx * dy;
        a_wy += w * y_val;
        a_wdxy += wdx * y_val;
        a_wdyy += wdy * y_val;
    }

    xtwx[0] = a_w;
    xtwx[1] = a_dx;
    xtwx[2] = a_dy;
    xtwx[3] = a_dx;
    xtwx[4] = a_dx2;
    xtwx[5] = a_dxdy;
    xtwx[6] = a_dy;
    xtwx[7] = a_dxdy;
    xtwx[8] = a_dy2;

    xtwy[0] = a_wy;
    xtwy[1] = a_wdxy;
    xtwy[2] = a_wdyy;
}
