//! Combined Visualization Examples for LOESS
//!
//! This script runs multiple scenarios to generate CSV data for visualization.
//! It covers:
//! 1. Degree Comparison (Linear vs Quadratic)
//! 2. Fraction Comparison (Effect of bandwidth)
//! 3. Interval Comparison (Confidence vs Prediction)
//! 4. Robustness Comparison (With vs Without robustness iterations)
//! 5. LOESS Concept (Local weighting visualization)
//! 6. Multivariate LOESS (2D surface smoothing)

use loess_rs::prelude::*;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running All Visualization Examples...");
    println!("=====================================");
    println!();

    // Ensure output directory exists
    let output_dir = "../output/visual/";
    std::fs::create_dir_all(output_dir)?;
    println!("Output directory: {}", output_dir);
    println!();

    run_degree_comparison()?;
    println!();

    run_fraction_comparison()?;
    println!();

    run_intervals_comparison()?;
    println!();

    run_robustness_comparison()?;
    println!();

    run_loess_concept()?;
    println!();

    run_multivariate_loess()?;
    println!();

    println!("All examples completed successfully.");
    Ok(())
}

/// 1. Degree Comparison
fn run_degree_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Generate data with a sharp Gaussian peak + asymmetric features
    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 4.0 - 2.0; // t in [-2, 2]
        // Sharp Gaussian peak + cubic term for asymmetry
        let signal = (-4.0 * t * t).exp() + 0.3 * t * t * t / 8.0;
        // Add small noise
        let noise = 0.03 * ((i as f64 * 7.0).sin() + (i as f64 * 13.0).cos()) / 2.0;

        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    // LOWESS: Linear degree - will flatten the peak
    let lowess_res = Loess::new()
        .degree(Linear)
        .fraction(0.25)
        .iterations(3)
        .surface_mode(Direct)
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // LOESS: Quadratic degree - captures the curvature of the peak
    let loess_res = Loess::new()
        .degree(Quadratic)
        .fraction(0.25)
        .iterations(3)
        .surface_mode(Direct)
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Write to CSV
    // Write to CSV
    let path = "../output/visual/degree_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_lowess,y_loess")?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], lowess_res.y[i], loess_res.y[i]
        )?;
    }

    let peak_idx = n / 2;
    let lowess_peak_error = (lowess_res.y[peak_idx] - y_true[peak_idx]).abs();
    let loess_peak_error = (loess_res.y[peak_idx] - y_true[peak_idx]).abs();

    println!("1. Degree Comparison: LOWESS vs LOESS");
    println!("-------------------------------------");
    println!("Peak errors (at x=0 where curvature is highest):");
    println!("  LOWESS (Linear):    {:.6}", lowess_peak_error);
    println!("  LOESS (Quadratic):  {:.6}", loess_peak_error);
    println!(
        "  Improvement ratio:  {:.2}x",
        lowess_peak_error / loess_peak_error
    );
    println!("Results exported to {}", path);

    Ok(())
}

/// 2. Fraction Comparison
fn run_fraction_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Generate data with multiple features
    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = 0.5 * t + 0.3 * t * t - 0.02 * t * t * t
            + 2.0 * (t * 1.5).sin()
            + 0.5 * (t * 5.0).sin();
        let noise = 0.3 * ((i as f64 * 7.0).sin() + (i as f64 * 13.0).cos()) / 2.0;

        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("2. Fraction Comparison");
    println!("----------------------");

    let fractions = [0.2, 0.5, 0.9];
    let mut results = Vec::new();

    for &frac in &fractions {
        let result = Loess::new()
            .degree(Quadratic)
            .fraction(frac)
            .iterations(2)
            .surface_mode(Direct)
            .boundary_policy(Reflect)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();

        let mut mse = 0.0;
        for i in 0..n {
            let error = result.y[i] - y_true[i];
            mse += error * error;
        }
        let rmse = (mse / n as f64).sqrt();

        println!("Fraction {:.1}: RMSE = {:.6}", frac, rmse);
        results.push(result);
    }

    let path = "../output/visual/fraction_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_frac_0.2,y_frac_0.5,y_frac_0.9")?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], results[0].y[i], results[1].y[i], results[2].y[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 3. Intervals Comparison
fn run_intervals_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 8.0 - 1.0;
        let signal = 3.0 + 2.0 * t - 0.3 * t * t + 1.5 * (t * 0.8).sin();
        let noise = 0.5 * ((i as f64 * 7.0).sin() + (i as f64 * 13.0).cos());

        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("3. Intervals Comparison");
    println!("-----------------------");

    // Confidence
    let result_conf = Loess::new()
        .degree(Quadratic)
        .fraction(0.3)
        .iterations(2)
        .confidence_intervals(0.95)
        .surface_mode(Direct)
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Prediction
    let result_pred = Loess::new()
        .degree(Quadratic)
        .fraction(0.3)
        .iterations(2)
        .prediction_intervals(0.95)
        .surface_mode(Direct)
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let conf_lower = result_conf.confidence_lower.as_ref().unwrap();
    let conf_upper = result_conf.confidence_upper.as_ref().unwrap();
    let pred_lower = result_pred.prediction_lower.as_ref().unwrap();
    let pred_upper = result_pred.prediction_upper.as_ref().unwrap();

    let avg_conf_width: f64 = conf_upper
        .iter()
        .zip(conf_lower.iter())
        .map(|(u, l)| u - l)
        .sum::<f64>()
        / n as f64;
    let avg_pred_width: f64 = pred_upper
        .iter()
        .zip(pred_lower.iter())
        .map(|(u, l)| u - l)
        .sum::<f64>()
        / n as f64;

    println!("Avg Confidence Width: {:.3}", avg_conf_width);
    println!("Avg Prediction Width: {:.3}", avg_pred_width);
    println!(
        "Ratio (Pred/Conf):    {:.2}x",
        avg_pred_width / avg_conf_width
    );

    let path = "../output/visual/intervals_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_smooth,conf_lower,conf_upper,pred_lower,pred_upper"
    )?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            result_conf.y[i],
            conf_lower[i],
            conf_upper[i],
            pred_lower[i],
            pred_upper[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 4. Robustness Comparison
fn run_robustness_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = 3.0 * (t * 0.8).sin();
        let mut value = signal + 0.5 * ((i as f64 * 17.0).sin() + (i as f64 * 3.0).cos());

        // Add outliers
        if t <= 4.0 {
            let pseudo_rand = ((i as f64 * 1337.0).sin() * 43758.5453).fract().abs();
            if pseudo_rand > 0.85 {
                value += 10.0 + pseudo_rand * 10.0;
            }
        }

        x.push(t);
        y_true.push(signal);
        y.push(value);
    }

    println!("4. Robustness Comparison");
    println!("------------------------");

    // Non-Robust (0 iterations)
    let result_non_robust = Loess::new()
        .degree(Quadratic)
        .fraction(0.25)
        .iterations(0)
        .surface_mode(Direct)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Robust (6 iterations)
    let result_robust = Loess::new()
        .degree(Quadratic)
        .fraction(0.25)
        .iterations(6)
        .surface_mode(Direct)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let mut mse_nr = 0.0;
    let mut mse_r = 0.0;
    for i in 0..n {
        let err_nr = result_non_robust.y[i] - y_true[i];
        let err_r = result_robust.y[i] - y_true[i];
        mse_nr += err_nr * err_nr;
        mse_r += err_r * err_r;
    }
    let rmse_nr = (mse_nr / n as f64).sqrt();
    let rmse_r = (mse_r / n as f64).sqrt();

    println!("RMSE (Non-Robust): {:.4}", rmse_nr);
    println!("RMSE (Robust):     {:.4}", rmse_r);
    println!("Improvement:       {:.2}x", rmse_nr / rmse_r);

    let path = "../output/visual/robustness_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_non_robust,y_robust")?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], result_non_robust.y[i], result_robust.y[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 5. LOESS Concept
fn run_loess_concept() -> Result<(), Box<dyn std::error::Error>> {
    let n = 80;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 2.0 * std::f64::consts::PI;
        let signal = t.sin();
        let noise = 0.3 * ((i as f64 * 7.0).sin() * (i as f64 * 3.0).cos());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    let fraction = 0.35;
    let result = Loess::new()
        .degree(Quadratic)
        .fraction(fraction)
        .iterations(0)
        .surface_mode(Direct)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    // Focus point visualization
    let focus_idx = 35;
    let x0 = x[focus_idx];

    // Manual neighbor finding and weighting for visualization
    let k = (fraction * n as f64).ceil() as usize;
    let mut distances: Vec<(usize, f64)> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| (i, (xi - x0).abs()))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let max_dist = distances[k - 1].1;

    let mut weights = vec![0.0; n];
    for i in 0..n {
        let dist = (x[i] - x0).abs();
        if dist < max_dist {
            let u = dist / max_dist;
            weights[i] = (1.0 - u * u * u).powi(3);
        }
    }

    // Manual Weighted Least Squares output (simplified)
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wx2 = 0.0;
    let mut sum_wx3 = 0.0;
    let mut sum_wx4 = 0.0;
    let mut sum_wy = 0.0;
    let mut sum_wxy = 0.0;
    let mut sum_wx2y = 0.0;

    for i in 0..n {
        if weights[i] > 0.0 {
            let w = weights[i];
            let dx = x[i] - x0;
            let dx2 = dx * dx;
            sum_w += w;
            sum_wx += w * dx;
            sum_wx2 += w * dx2;
            sum_wx3 += w * dx2 * dx;
            sum_wx4 += w * dx2 * dx2;
            sum_wy += w * y[i];
            sum_wxy += w * dx * y[i];
            sum_wx2y += w * dx2 * y[i];
        }
    }

    // Cramer's rule
    let da = sum_wy * (sum_wx2 * sum_wx4 - sum_wx3 * sum_wx3)
        - sum_wx * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
        + sum_wx2 * (sum_wxy * sum_wx3 - sum_wx2 * sum_wx2y);
    let det = sum_w * (sum_wx2 * sum_wx4 - sum_wx3 * sum_wx3)
        - sum_wx * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
        + sum_wx2 * (sum_wx * sum_wx3 - sum_wx2 * sum_wx2);
    let db = sum_w * (sum_wxy * sum_wx4 - sum_wx3 * sum_wx2y)
        - sum_wy * (sum_wx * sum_wx4 - sum_wx3 * sum_wx2)
        + sum_wx2 * (sum_wx * sum_wx2y - sum_wxy * sum_wx2);
    let dc = sum_w * (sum_wx2 * sum_wx2y - sum_wxy * sum_wx3)
        - sum_wx * (sum_wx * sum_wx2y - sum_wxy * sum_wx2)
        + sum_wy * (sum_wx * sum_wx3 - sum_wx2 * sum_wx2);

    let a = da / det;
    let b = db / det;
    let c = dc / det;

    println!("5. LOESS Concept");
    println!("----------------");
    println!("Focus point: x = {:.2} (Index {})", x0, focus_idx);
    println!("Local Fit: a={:.3}, b={:.3}, c={:.3}", a, b, c);

    let path = "../output/visual/loess_concept.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_noisy,y_smooth,weight,y_local_fit_x0,is_focus")?;

    for i in 0..n {
        let is_focus = if i == focus_idx { 1 } else { 0 };
        let dx = x[i] - x0;
        let local_val = if weights[i] > 0.0 {
            a + b * dx + c * dx * dx
        } else {
            f64::NAN
        };
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y[i], result.y[i], weights[i], local_val, is_focus
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}

/// 6. Multivariate LOESS
fn run_multivariate_loess() -> Result<(), Box<dyn std::error::Error>> {
    let n_x = 30;
    let n_y = 30;
    let n = n_x * n_y;

    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    let mut z_true = Vec::with_capacity(n);

    for i in 0..n_x {
        for j in 0..n_y {
            let x_val = (i as f64 / (n_x - 1) as f64) * 4.0 - 2.0;
            let y_val = (j as f64 / (n_y - 1) as f64) * 4.0 - 2.0;
            // A clean Gaussian hill
            let z_signal = (-(x_val * x_val + y_val * y_val) * 0.5).exp();
            let noise = 0.05 * ((i * 7 + j * 13) as f64).sin();

            x.push(x_val);
            x.push(y_val);
            z_true.push(z_signal);
            y.push(z_signal + noise);
        }
    }

    println!("6. Multivariate LOESS");
    println!("---------------------");
    println!("Smoothing 2D surface ({} points)", n);

    let result = Loess::new()
        .dimensions(2)
        .degree(Quadratic)
        .fraction(0.3)
        .iterations(3)
        .surface_mode(Interpolation)
        .interpolation_vertices(100)
        .boundary_policy(Reflect)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let mut mse = 0.0;
    for i in 0..n {
        let error = (result.y[i] - z_true[i]).abs();
        mse += error * error;
    }
    mse /= n as f64;

    println!("RMSE: {:.6}", mse.sqrt());

    let path = "../output/visual/multivariate_loess.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y,z_true,z_noisy,z_smooth")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i * 2],
            x[i * 2 + 1],
            z_true[i],
            y[i],
            result.y[i]
        )?;
    }
    println!("Results exported to {}", path);

    Ok(())
}
