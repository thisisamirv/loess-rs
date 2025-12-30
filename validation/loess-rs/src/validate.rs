use loess_rs::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize)]
struct ValidationData {
    name: String,
    notes: String,
    input: InputData,
    params: Params,
    #[serde(skip_deserializing)]
    result: ResultData,
}

#[derive(Debug, Deserialize, Serialize)]
struct InputData {
    x: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Params {
    fraction: f64,
    degree: usize,
    iterations: usize,
    #[serde(flatten)]
    extra: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ResultData {
    fitted: Vec<f64>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = Path::new("../output/r");
    let output_dir = Path::new("../output/loess_rs");

    if !input_dir.exists() {
        eprintln!(
            "Input directory {:?} does not exist. Run validate.py first.",
            input_dir
        );
        return Ok(());
    }

    fs::create_dir_all(output_dir)?;

    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            println!("Processing {:?}", path.file_name().unwrap());
            process_file(&path, output_dir)?;
        }
    }

    Ok(())
}

fn process_file(input_path: &Path, output_dir: &Path) -> Result<(), Box<dyn Error>> {
    let file = fs::File::open(input_path)?;
    let mut data: ValidationData = serde_json::from_reader(file)?;

    let degree = match data.params.degree {
        0 => Constant,
        1 => Linear,
        2 => Quadratic,
        3 => Cubic,
        _ => panic!("Unsupported degree: {}", data.params.degree),
    };

    let surface_mode = if let Some(extra) = &data.params.extra {
        let is_direct = extra.get("surface").and_then(|v| v.as_str()) == Some("direct")
            || extra
                .get("extra")
                .and_then(|v| v.get("surface"))
                .and_then(|v| v.as_str())
                == Some("direct");

        if is_direct {
            Direct
        } else {
            Interpolation
        }
    } else {
        Interpolation
    };

    let processor = Loess::new()
        .fraction(data.params.fraction)
        .degree(degree)
        .iterations(data.params.iterations)
        .surface_mode(surface_mode)
        .adapter(Batch)
        .build()?;

    let result = processor.fit(&data.input.x, &data.input.y)?;
    let fitted = result.y;

    data.result.fitted = fitted;

    let output_path = output_dir.join(input_path.file_name().unwrap());
    let output_json = serde_json::to_string_pretty(&data)?;
    fs::write(output_path, output_json)?;

    Ok(())
}
