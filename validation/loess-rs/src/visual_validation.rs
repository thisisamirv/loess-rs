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

    run_kernel_comparison()?;
    println!();

    run_robust_method_comparison()?;
    println!();

    run_boundary_policy_comparison()?;
    println!();

    run_higher_degree_comparison()?;
    println!();

    run_gap_handling()?;
    println!();

    run_cv_comparison()?;
    println!();

    run_surface_mode_comparison()?;
    println!();

    run_scaling_method_comparison()?;
    println!();

    run_zero_weight_fallback_comparison()?;
    println!();

    run_degree_interpolation_comparison()?;
    println!();

    run_streaming_comparison()?;
    println!();

    run_online_comparison()?;
    println!();

    run_auto_converge_comparison()?;
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

/// 7. Kernel Comparison
fn run_kernel_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 6.0;
        let signal = (t).sin() + 0.5 * (t * 3.0).cos();
        let noise = 0.2 * ((i as f64 * 11.0).sin());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("7. Kernel Comparison");
    println!("--------------------");

    let kernels = [
        Tricube,
        Gaussian,
        Uniform,
        Cosine,
        Epanechnikov,
        Biweight,
        Triangle,
    ];
    let mut results = Vec::new();

    for &kernel in &kernels {
        let result = Loess::new()
            .weight_function(kernel)
            .fraction(0.3)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Kernel processed: {}", kernel.name());
    }

    let path = "../output/visual/kernel_comparison.csv";
    let mut file = File::create(path)?;
    write!(file, "x,y_true,y_noisy")?;
    for kernel in &kernels {
        write!(file, ",y_{}", kernel.name().to_lowercase())?;
    }
    writeln!(file)?;

    for i in 0..n {
        write!(file, "{},{},{}", x[i], y_true[i], y[i])?;
        for result in &results {
            write!(file, ",{}", result.y[i])?;
        }
        writeln!(file)?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 8. Robustness Method Comparison
fn run_robust_method_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = (t * 0.5).sin() * 5.0;
        let mut value = signal + 0.3 * (i as f64 * 5.0).cos();

        // Add extreme outliers
        if i % 20 == 0 {
            value += 15.0;
        } else if i % 20 == 10 {
            value -= 15.0;
        }

        x.push(t);
        y_true.push(signal);
        y.push(value);
    }

    println!("8. Robustness Method Comparison");
    println!("-------------------------------");

    let methods = [Bisquare, Huber, Talwar];
    let mut results = Vec::new();

    for &method in &methods {
        let result = Loess::new()
            .robustness_method(method)
            .iterations(5)
            .fraction(0.2)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Method processed: {:?}", method);
    }

    let path = "../output/visual/robust_method_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_bisquare,y_huber,y_talwar")?;
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

/// 9. Boundary Policy Comparison
fn run_boundary_policy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Data with strong trend at boundaries
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let signal = (3.0 * t).exp(); // Exponential growth
        let noise = 0.1 * (i as f64 * 10.0).sin();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("9. Boundary Policy Comparison");
    println!("-----------------------------");

    let policies = [NoBoundary, Extend, Reflect];
    let mut results = Vec::new();

    for &policy in &policies {
        let result = Loess::new()
            .boundary_policy(policy)
            .fraction(0.4) // Larger fraction highlights boundary bias
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Policy processed: {:?}", policy);
    }

    let path = "../output/visual/boundary_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_none,y_extend,y_reflect")?;
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

/// 10. Higher Degree Comparison
fn run_higher_degree_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        // Function with complex curvature: quartic-like
        let signal = 0.1 * (t - 5.0).powi(4) - 2.0 * (t - 5.0).powi(2) + 10.0;
        let noise = 0.5 * (i as f64 * 3.0).sin();
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("10. Higher Degree Comparison");
    println!("----------------------------");

    let degrees = [Quadratic, Cubic, Quartic];
    let mut results = Vec::new();

    for &degree in &degrees {
        let result = Loess::new()
            .degree(degree)
            .fraction(0.2)
            .adapter(Batch)
            .build()
            .unwrap()
            .fit(&x, &y)
            .unwrap();
        results.push(result);
        println!("  Degree processed: {:?}", degree);
    }

    let path = "../output/visual/higher_degree_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_quadratic,y_cubic,y_quartic")?;
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

/// 11. Gap Handling
fn run_gap_handling() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        // Create a large gap in the middle
        if t > 4.0 && t < 7.0 {
            continue;
        }
        let signal = (t).sin();
        let noise = 0.1 * (i as f64 * 7.0).cos();
        x.push(t);
        y.push(signal + noise);
    }

    println!("11. Gap Handling");
    println!("----------------");

    let result = Loess::new()
        .degree(Quadratic)
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/gap_handling.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_noisy,y_smooth")?;
    for i in 0..x.len() {
        writeln!(file, "{},{},{}", x[i], y[i], result.y[i])?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 12. Cross-Validation Comparison
fn run_cv_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    // Highly noisy signal on a small scale to make bandwidth selection critical
    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 5.0;
        let signal = (t * 2.5).sin();
        let noise = 0.6 * ((i as f64 * 17.0).sin() * (i as f64 * 13.0).cos());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("12. Cross-Validation Comparison");
    println!("-------------------------------");

    let candidate_fractions = [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8,
    ];

    // 1. LOOCV
    let loocv_result = Loess::new()
        .cross_validate(LOOCV(&candidate_fractions))
        .degree(Quadratic)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  LOOCV Best Fraction: {}", loocv_result.fraction_used);

    // 2. K-Fold (5 folds)
    let kfold_result = Loess::new()
        .cross_validate(KFold(5, &candidate_fractions).seed(42))
        .degree(Quadratic)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  5-Fold Best Fraction: {}", kfold_result.fraction_used);

    // 3. No CV (Fixed bad fraction - too small)
    let fixed_result = Loess::new()
        .fraction(0.1) // Overfitting deliberately
        .degree(Quadratic)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();
    println!("  No CV Fraction (Fixed): 0.1");

    // Export scores comparison
    let scores_path = "../output/visual/cv_scores.csv";
    let mut score_file = File::create(scores_path)?;
    writeln!(score_file, "fraction,loocv_rmse,kfold_rmse")?;
    let loocv_scores = loocv_result.cv_scores.as_ref().unwrap();
    let kfold_scores = kfold_result.cv_scores.as_ref().unwrap();
    for i in 0..candidate_fractions.len() {
        writeln!(
            score_file,
            "{},{},{}",
            candidate_fractions[i], loocv_scores[i], kfold_scores[i]
        )?;
    }
    println!("CV Scores exported to {}", scores_path);

    // Export fits comparison
    let fits_path = "../output/visual/cv_fits.csv";
    let mut fit_file = File::create(fits_path)?;
    writeln!(fit_file, "x,y_true,y_noisy,y_loocv,y_kfold,y_fixed")?;
    for i in 0..n {
        writeln!(
            fit_file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], loocv_result.y[i], kfold_result.y[i], fixed_result.y[i]
        )?;
    }
    println!("CV Fits exported to {}", fits_path);

    Ok(())
}

/// 13. Surface Mode Comparison
fn run_surface_mode_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        let signal = (t * 1.5).sin() + 0.5 * (t * 4.0).cos();
        let noise = 0.2 * ((i as f64 * 11.0).sin());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("13. Surface Mode Comparison");
    println!("---------------------------");

    let result_direct = Loess::new()
        .surface_mode(Direct)
        .fraction(0.2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_interp = Loess::new()
        .surface_mode(Interpolation)
        .cell(0.1)
        .fraction(0.2)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/surface_mode_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_direct,y_interpolation")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], result_direct.y[i], result_interp.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 14. Scaling Method Comparison
fn run_scaling_method_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 5.0;
        let signal = (t * 0.8).sin() * 2.0;
        let mut value = signal + 0.2 * (i as f64 * 5.0).cos();

        // Add asymmetric outliers
        if i % 15 == 0 {
            value += 8.0; // Large positive outliers
        }

        x.push(t);
        y_true.push(signal);
        y.push(value);
    }

    println!("14. Scaling Method Comparison");
    println!("-----------------------------");

    let result_mad = Loess::new()
        .scaling_method(MAD)
        .iterations(4)
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_mar = Loess::new()
        .scaling_method(MAR)
        .iterations(4)
        .fraction(0.3)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/scaling_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_mad,y_mar")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], result_mad.y[i], result_mar.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 15. Zero Weight Fallback Comparison
fn run_zero_weight_fallback_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let mut x = Vec::new();
    let mut y = Vec::new();

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 10.0;
        x.push(t);
        // Create an extreme outlier that will be zero-weighted by robustness
        if i == 50 {
            y.push(100.0);
        } else {
            y.push((t * 0.5).cos());
        }
    }

    println!("15. Zero Weight Fallback Comparison");
    println!("-----------------------------------");

    let result_mean = Loess::new()
        .zero_weight_fallback(UseLocalMean)
        .fraction(0.1)
        .iterations(5)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let result_orig = Loess::new()
        .zero_weight_fallback(ReturnOriginal)
        .fraction(0.1)
        .iterations(5)
        .adapter(Batch)
        .build()
        .unwrap()
        .fit(&x, &y)
        .unwrap();

    let path = "../output/visual/zero_weight_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_noisy,y_mean,y_original")?;
    for i in 0..x.len() {
        writeln!(
            file,
            "{},{},{},{}",
            x[i], y[i], result_mean.y[i], result_orig.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 16. Degree Interpolation Comparison
fn run_degree_interpolation_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64 / (n - 1) as f64) * 8.0 - 4.0;
        let signal = (-(t * t)).exp() + 0.1 * t; // Sharp peak
        let noise = 0.05 * ((i as f64 * 13.0).sin());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("16. Degree Interpolation Comparison");
    println!("-----------------------------------");

    let lin_direct = Loess::new()
        .degree(Linear)
        .surface_mode(Direct)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    let lin_interp = Loess::new()
        .degree(Linear)
        .surface_mode(Interpolation)
        .cell(0.2)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    let quad_direct = Loess::new()
        .degree(Quadratic)
        .surface_mode(Direct)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    let quad_interp = Loess::new()
        .degree(Quadratic)
        .surface_mode(Interpolation)
        .cell(0.2)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    let cubic_direct = Loess::new()
        .degree(Cubic)
        .surface_mode(Direct)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    let cubic_interp = Loess::new()
        .degree(Cubic)
        .surface_mode(Interpolation)
        .cell(0.2)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    let quartic_direct = Loess::new()
        .degree(Quartic)
        .surface_mode(Direct)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;
    let quartic_interp = Loess::new()
        .degree(Quartic)
        .surface_mode(Interpolation)
        .cell(0.2)
        .fraction(0.2)
        .adapter(Batch)
        .build()?
        .fit(&x, &y)?;

    let path = "../output/visual/degree_interpolation_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_lin_direct,y_lin_interp,y_quad_direct,y_quad_interp,y_cubic_direct,y_cubic_interp,y_quartic_direct,y_quartic_interp"
    )?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            lin_direct.y[i],
            lin_interp.y[i],
            quad_direct.y[i],
            quad_interp.y[i],
            cubic_direct.y[i],
            cubic_interp.y[i],
            quartic_direct.y[i],
            quartic_interp.y[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 17. Streaming Adapter Comparison
fn run_streaming_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1000;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / 100.0;
        let signal = (t).sin() + 0.5 * (t * 2.5).cos();
        let noise = 0.1 * ((i as f64 * 7.0).sin());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("17. Streaming Adapter Comparison");
    println!("-------------------------------");

    let chunk_size = 200;
    let overlap = 80;

    let mut streaming_weighted = Loess::new()
        .adapter(Streaming)
        .chunk_size(chunk_size)
        .overlap(overlap)
        .merge_strategy(WeightedAverage)
        .fraction(0.2)
        .build()?;

    let mut streaming_average = Loess::new()
        .adapter(Streaming)
        .chunk_size(chunk_size)
        .overlap(overlap)
        .merge_strategy(Average)
        .fraction(0.2)
        .build()?;

    let mut streaming_first = Loess::new()
        .adapter(Streaming)
        .chunk_size(chunk_size)
        .overlap(overlap)
        .merge_strategy(TakeFirst)
        .fraction(0.2)
        .build()?;

    let mut y_weighted = Vec::new();
    let mut y_average = Vec::new();
    let mut y_first = Vec::new();

    for i in (0..n).step_by(chunk_size) {
        let end = (i + chunk_size).min(n);
        let chunk_x = &x[i..end];
        let chunk_y = &y[i..end];

        y_weighted.extend(streaming_weighted.process_chunk(chunk_x, chunk_y)?.y);
        y_average.extend(streaming_average.process_chunk(chunk_x, chunk_y)?.y);
        y_first.extend(streaming_first.process_chunk(chunk_x, chunk_y)?.y);

        if end == n {
            break;
        }
    }

    y_weighted.extend(streaming_weighted.finalize()?.y);
    y_average.extend(streaming_average.finalize()?.y);
    y_first.extend(streaming_first.finalize()?.y);

    let path = "../output/visual/streaming_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_weighted,y_average,y_first")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            x[i], y_true[i], y[i], y_weighted[i], y_average[i], y_first[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 18. Online Adapter Comparison
fn run_online_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 500;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64;
        // A signal that shifts its mean over time
        let signal = (t * 0.05).sin() + if t > 250.0 { 2.0 } else { 0.0 };
        let noise = 0.2 * ((i as f64 * 3.0).cos());
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("18. Online Adapter Comparison");
    println!("----------------------------");

    let mut online_small = Loess::new()
        .adapter(Online)
        .window_capacity(50)
        .update_mode(Incremental)
        .build()?;

    let mut online_large = Loess::new()
        .adapter(Online)
        .window_capacity(200)
        .update_mode(Full)
        .iterations(2)
        .build()?;

    let mut y_small = Vec::new();
    let mut y_large = Vec::new();

    for i in 0..n {
        let xi = [x[i]];
        let yi = y[i];

        y_small.push(
            online_small
                .add_point(&xi, yi)?
                .map(|o| o.smoothed)
                .unwrap_or(y[0]),
        );
        y_large.push(
            online_large
                .add_point(&xi, yi)?
                .map(|o| o.smoothed)
                .unwrap_or(y[0]),
        );
    }

    let path = "../output/visual/online_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(file, "x,y_true,y_noisy,y_small_window,y_large_window")?;
    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{}",
            x[i], y_true[i], y[i], y_small[i], y_large[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}

/// 19. Auto-Convergence Comparison
fn run_auto_converge_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let n = 200;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut y_true = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / 10.0;
        let signal = (t).sin();
        let noise = if i % 20 == 0 {
            2.0
        } else {
            0.1 * (i as f64).cos()
        };
        x.push(t);
        y_true.push(signal);
        y.push(signal + noise);
    }

    println!("19. Auto-Convergence Comparison (All Adapters)");
    println!("--------------------------------------------");

    let iterations = 4;
    let tolerance = 0.001;

    // 1. Batch
    let batch_off = Loess::new()
        .fraction(0.3)
        .degree(Quadratic)
        .iterations(iterations)
        .adapter(Batch)
        .build()?;
    let batch_on = Loess::new()
        .fraction(0.3)
        .degree(Quadratic)
        .iterations(iterations)
        .auto_converge(tolerance)
        .adapter(Batch)
        .build()?;

    let res_batch_off = batch_off.fit(&x, &y)?;
    let res_batch_on = batch_on.fit(&x, &y)?;

    // 2. Streaming
    let mut stream_off = Loess::new()
        .fraction(0.3)
        .degree(Quadratic)
        .iterations(iterations)
        .chunk_size(100)
        .overlap(20)
        .adapter(Streaming)
        .build()?;
    let mut stream_on = Loess::new()
        .fraction(0.3)
        .degree(Quadratic)
        .iterations(iterations)
        .auto_converge(tolerance)
        .chunk_size(100)
        .overlap(20)
        .adapter(Streaming)
        .build()?;

    let mut y_stream_off = Vec::new();
    let mut y_stream_on = Vec::new();
    let mut iter_stream_off = Vec::new();
    let mut iter_stream_on = Vec::new();

    for i in (0..n).step_by(100) {
        let end = (i + 100).min(n);
        let out_off = stream_off.process_chunk(&x[i..end], &y[i..end])?;
        let out_on = stream_on.process_chunk(&x[i..end], &y[i..end])?;

        let n_off = out_off.y.len();
        let n_on = out_on.y.len();

        y_stream_off.extend(out_off.y);
        y_stream_on.extend(out_on.y);

        iter_stream_off.extend(vec![out_off.iterations_used.unwrap_or(0); n_off]);
        iter_stream_on.extend(vec![out_on.iterations_used.unwrap_or(0); n_on]);
    }
    let fin_off = stream_off.finalize()?;
    let fin_on = stream_on.finalize()?;

    let n_fin_off = fin_off.y.len();
    let n_fin_on = fin_on.y.len();

    y_stream_off.extend(fin_off.y);
    y_stream_on.extend(fin_on.y);

    iter_stream_off.extend(vec![fin_off.iterations_used.unwrap_or(0); n_fin_off]);
    iter_stream_on.extend(vec![fin_on.iterations_used.unwrap_or(0); n_fin_on]);

    // 3. Online
    let mut online_off = Loess::new()
        .fraction(0.3)
        .degree(Quadratic)
        .iterations(iterations)
        .window_capacity(50)
        .update_mode(Full)
        .adapter(Online)
        .build()?;
    let mut online_on = Loess::new()
        .fraction(0.3)
        .degree(Quadratic)
        .iterations(iterations)
        .auto_converge(tolerance)
        .window_capacity(50)
        .update_mode(Full)
        .adapter(Online)
        .build()?;

    let mut y_online_off = Vec::new();
    let mut y_online_on = Vec::new();
    let mut iter_online_off = Vec::new();
    let mut iter_online_on = Vec::new();

    for i in 0..n {
        let xi = [x[i]];
        let out_off = online_off.add_point(&xi, y[i])?;
        let out_on = online_on.add_point(&xi, y[i])?;

        match (out_off, out_on) {
            (Some(off), Some(on)) => {
                y_online_off.push(off.smoothed);
                y_online_on.push(on.smoothed);
                iter_online_off.push(off.iterations_used.unwrap_or(0));
                iter_online_on.push(on.iterations_used.unwrap_or(0));
            }
            _ => {
                // Not enough points yet
                y_online_off.push(y_true[i]);
                y_online_on.push(y_true[i]);
                iter_online_off.push(0);
                iter_online_on.push(0);
            }
        }
    }

    let path = "../output/visual/auto_converge_comparison.csv";
    let mut file = File::create(path)?;
    writeln!(
        file,
        "x,y_true,y_noisy,y_batch_off,y_batch_on,y_stream_off,y_stream_on,y_online_off,y_online_on,iter_batch_off,iter_batch_on,iter_stream_off,iter_stream_on,iter_online_off,iter_online_on"
    )?;

    for i in 0..n {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            x[i],
            y_true[i],
            y[i],
            res_batch_off.y[i],
            res_batch_on.y[i],
            y_stream_off[i],
            y_stream_on[i],
            y_online_off[i],
            y_online_on[i],
            res_batch_off.iterations_used.unwrap_or(0),
            res_batch_on.iterations_used.unwrap_or(0),
            iter_stream_off[i],
            iter_stream_on[i],
            iter_online_off[i],
            iter_online_on[i]
        )?;
    }
    println!("Results exported to {}", path);
    Ok(())
}
