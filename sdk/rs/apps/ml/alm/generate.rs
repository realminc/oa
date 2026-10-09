use std::{error::Error, path::PathBuf, time::Instant};

use oa::Cli;
use oa::sdk::{
	data::HumanMl3dDataset,
	ml::alm::{Alm, AlmGenerationOptions},
};

// ─── config ──────────────────────────────────────────────────────────────────

struct AlmGenerateConfig {
	model: String,
	dataset: String,
	split: String,
	output: String,
	count: usize,
	temperature: f32,
	max_len: usize,
	seed: u64,
	conditioning_clip: usize,
	caption_index: usize,
	prompt: String,
}

impl Default for AlmGenerateConfig {
	fn default() -> Self {
		Self {
			model: String::new(),
			dataset: "data/humanMl3d/Cmp".into(),
			split: "train".into(),
			output: "var/alm".into(),
			count: 3,
			temperature: 1.0,
			max_len: 64,
			seed: 42,
			conditioning_clip: 0,
			caption_index: 0,
			prompt: String::new(),
		}
	}
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
	let mut cli = Cli::new("genalm", "generate ALM motion sequences");
	cli.add_option(
		"--model",
		|c: &mut AlmGenerateConfig| &mut c.model,
		"ALM bundle path (required)",
	);
	cli.add_option(
		"--dataset",
		|c: &mut AlmGenerateConfig| &mut c.dataset,
		"dataset directory",
	);
	cli.add_option(
		"--split",
		|c: &mut AlmGenerateConfig| &mut c.split,
		"dataset split name",
	);
	cli.add_option(
		"--out-dir",
		|c: &mut AlmGenerateConfig| &mut c.output,
		"output directory",
	);
	cli.add_option(
		"--gen-count",
		|c: &mut AlmGenerateConfig| &mut c.count,
		"number of sequences to generate",
	);
	cli.add_option(
		"--gen-temp",
		|c: &mut AlmGenerateConfig| &mut c.temperature,
		"sampling temperature",
	);
	cli.add_option(
		"--gen-len",
		|c: &mut AlmGenerateConfig| &mut c.max_len,
		"max generation length (tokens)",
	);
	cli.add_option(
		"--seed",
		|c: &mut AlmGenerateConfig| &mut c.seed,
		"random seed",
	);
	cli.add_option(
		"--conditioning-clip",
		|c: &mut AlmGenerateConfig| &mut c.conditioning_clip,
		"conditioning clip index",
	);
	cli.add_option(
		"--caption-index",
		|c: &mut AlmGenerateConfig| &mut c.caption_index,
		"caption index within clip",
	);
	cli.add_option(
		"--prompt",
		|c: &mut AlmGenerateConfig| &mut c.prompt,
		"text prompt (requires native CLIP)",
	);
	if !cli.parse() {
		return Ok(());
	}
	let cfg = cli.into_config();

	if cfg.model.is_empty() {
		return Err(argument_error("--model is required"));
	}

	let model_path = PathBuf::from(&cfg.model);
	let dataset_path = PathBuf::from(&cfg.dataset);
	let output_path = PathBuf::from(&cfg.output);
	let prompt_opt: Option<&str> = if cfg.prompt.is_empty() {
		None
	} else {
		Some(&cfg.prompt)
	};

	let engine = oa::Engine::new()?;
	let model = Alm::load_bundle(&engine, &model_path)?;
	let conditioned = model.config().prior.text_feature_dim > 0;
	let load_count = if conditioned && prompt_opt.is_none() {
		cfg.conditioning_clip + 1
	} else {
		1
	};
	let dataset = HumanMl3dDataset::open_cmp(&dataset_path, &cfg.split, load_count)?;
	std::fs::create_dir_all(&output_path)?;
	let cached_feature = if conditioned && prompt_opt.is_none() {
		Some(caption_feature(&engine, &model, &dataset, &cfg)?)
	} else {
		None
	};
	if !conditioned && prompt_opt.is_some() {
		return Err(argument_error(
			"an unconditional ALM cannot accept --prompt",
		));
	}
	if prompt_opt.is_some() && !model.has_native_text_encoder() {
		return Err(argument_error(
			"--prompt requires an ALM bundle with native CLIP and merge assets",
		));
	}
	println!(
		"\ngenalm — {} clips · max {} tokens · temperature {:.2}",
		cfg.count, cfg.max_len, cfg.temperature
	);
	let run_start = Instant::now();
	for index in 0..cfg.count {
		let generation = AlmGenerationOptions {
			temperature: cfg.temperature,
			top_k: 0,
			top_p: 0.9,
			max_length: cfg.max_len,
			seed: cfg.seed.wrapping_add(index as u64),
			use_cache: true,
		};
		let start = Instant::now();
		let motion = if let Some(prompt) = prompt_opt {
			model.generate_motion_prompt(prompt, generation)?
		} else if let Some(feature) = cached_feature.as_ref() {
			model.generate_motion_conditioned(feature, generation)?
		} else {
			model.generate_motion(1, generation)?
		};
		let elapsed = start.elapsed();
		let Some(motion) = motion else {
			println!("  [{index}] EOM emitted before a motion token");
			continue;
		};
		let [batch, frames, feature_dim] = motion.shape() else {
			return Err(argument_error("generated ALM motion is not rank three"));
		};
		if *batch != 1 || *feature_dim != dataset.feature_dim() {
			return Err(argument_error(
				"generated ALM motion geometry does not match the dataset",
			));
		}
		let mut values = motion.read_f32()?;
		dataset.denormalize(&mut values)?;
		let stem = format!("alm_gen_{index}_t{:.1}", cfg.temperature);
		let path = output_path.join(format!("{stem}.npy"));
		write_npy_f32(&path, &[*frames, *feature_dim], &values)?;
		let metadata = format!(
			"format=oa_alm_generation_v1\nbundle={}\ndataset={}\nsplit={}\nseed={}\ntemperature={}\nmax_motion_tokens={}\nframes={}\nfeature_dim={}\nprompt={}\n",
			model_path.display(),
			dataset_path.display(),
			cfg.split,
			generation.seed,
			generation.temperature,
			generation.max_length,
			frames,
			feature_dim,
			prompt_opt.unwrap_or("")
		);
		std::fs::write(output_path.join(format!("{stem}.meta.txt")), metadata)?;
		println!(
			"  [{index}] {frames} frames × {feature_dim} · {:.1} ms · saved {}",
			elapsed.as_secs_f64() * 1000.0,
			path.display()
		);
	}
	println!(
		"generation complete in {:.2}s",
		run_start.elapsed().as_secs_f64()
	);
	Ok(())
}

fn caption_feature(
	engine: &oa::Engine,
	model: &Alm,
	dataset: &HumanMl3dDataset,
	cfg: &AlmGenerateConfig,
) -> Result<oa::Matrix, Box<dyn Error>> {
	if dataset.text_feature_format() != Some("oa_clip_text_v1")
		|| dataset.text_feature_model() != model.text_encoder_identity()
		|| dataset.text_feature_dim() != model.config().prior.text_feature_dim
	{
		return Err(argument_error(
			"dataset cached text identity does not match the ALM bundle",
		));
	}
	let features = dataset
		.clip_text_features(cfg.conditioning_clip)
		.ok_or_else(|| argument_error("conditioning clip is outside the loaded split"))?;
	let captions = dataset
		.clip_captions(cfg.conditioning_clip)
		.ok_or_else(|| argument_error("conditioning clip has no captions"))?;
	let width = dataset.text_feature_dim();
	let expected = captions
		.len()
		.checked_mul(width)
		.ok_or_else(|| argument_error("cached caption-feature size overflows usize"))?;
	if cfg.caption_index >= captions.len() || features.len() != expected {
		return Err(argument_error(
			"caption index or cached text-feature geometry is invalid",
		));
	}
	let start = cfg.caption_index * width;
	Ok(oa::Matrix::from_f32(
		engine,
		[1, width],
		&features[start..start + width],
	)?)
}

fn write_npy_f32(path: &std::path::Path, shape: &[usize], values: &[f32]) -> std::io::Result<()> {
	if shape.iter().product::<usize>() != values.len() {
		return Err(std::io::Error::new(
			std::io::ErrorKind::InvalidInput,
			"NPY shape does not match values",
		));
	}
	let dimensions = shape
		.iter()
		.map(usize::to_string)
		.collect::<Vec<_>>()
		.join(", ");
	let comma = if shape.len() == 1 { "," } else { "" };
	let mut header =
		format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions}{comma}), }}");
	let padding = (16 - ((10 + header.len() + 1) % 16)) % 16;
	header.extend(std::iter::repeat_n(' ', padding));
	header.push('\n');
	let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
	bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
	bytes.extend_from_slice(header.as_bytes());
	for value in values {
		bytes.extend_from_slice(&value.to_le_bytes());
	}
	std::fs::write(path, bytes)
}

fn argument_error(message: &str) -> Box<dyn Error> {
	std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}
