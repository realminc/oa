use std::{env, error::Error, path::PathBuf, time::Instant};

use oa::sdk::{
	data::HumanMl3dDataset,
	ml::alm::{Alm, AlmGenerationOptions},
};

struct Options {
	model: PathBuf,
	dataset: PathBuf,
	split: String,
	output: PathBuf,
	count: usize,
	temperature: f32,
	max_len: usize,
	seed: u64,
	conditioning_clip: usize,
	caption_index: usize,
	prompt: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
	let options = parse_options()?;
	let engine = oa::Engine::new()?;
	let model = Alm::load_bundle(&engine, &options.model)?;
	let conditioned = model.config().prior.text_feature_dim > 0;
	let load_count = if conditioned && options.prompt.is_none() {
		options.conditioning_clip + 1
	} else {
		1
	};
	let dataset = HumanMl3dDataset::open_cmp(&options.dataset, &options.split, load_count)?;
	std::fs::create_dir_all(&options.output)?;
	let cached_feature = if conditioned && options.prompt.is_none() {
		Some(caption_feature(&engine, &model, &dataset, &options)?)
	} else {
		None
	};
	if !conditioned && options.prompt.is_some() {
		return Err(argument_error(
			"an unconditional ALM cannot accept --prompt",
		));
	}
	if options.prompt.is_some() && !model.has_native_text_encoder() {
		return Err(argument_error(
			"--prompt requires an ALM bundle with native CLIP and merge assets",
		));
	}
	println!(
		"\ngenalm — {} clips · max {} tokens · temperature {:.2}",
		options.count, options.max_len, options.temperature
	);
	let run_start = Instant::now();
	for index in 0..options.count {
		let generation = AlmGenerationOptions {
			temperature: options.temperature,
			top_k: 0,
			top_p: 0.9,
			max_length: options.max_len,
			seed: options.seed.wrapping_add(index as u64),
			use_cache: true,
		};
		let start = Instant::now();
		let motion = if let Some(prompt) = options.prompt.as_deref() {
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
		let stem = format!("alm_gen_{index}_t{:.1}", options.temperature);
		let path = options.output.join(format!("{stem}.npy"));
		write_npy_f32(&path, &[*frames, *feature_dim], &values)?;
		let metadata = format!(
			"format=oa_alm_generation_v1\nbundle={}\ndataset={}\nsplit={}\nseed={}\ntemperature={}\nmax_motion_tokens={}\nframes={}\nfeature_dim={}\nprompt={}\n",
			options.model.display(),
			options.dataset.display(),
			options.split,
			generation.seed,
			generation.temperature,
			generation.max_length,
			frames,
			feature_dim,
			options.prompt.as_deref().unwrap_or("")
		);
		std::fs::write(options.output.join(format!("{stem}.meta.txt")), metadata)?;
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
	options: &Options,
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
		.clip_text_features(options.conditioning_clip)
		.ok_or_else(|| argument_error("conditioning clip is outside the loaded split"))?;
	let captions = dataset
		.clip_captions(options.conditioning_clip)
		.ok_or_else(|| argument_error("conditioning clip has no captions"))?;
	let width = dataset.text_feature_dim();
	let expected = captions
		.len()
		.checked_mul(width)
		.ok_or_else(|| argument_error("cached caption-feature size overflows usize"))?;
	if options.caption_index >= captions.len() || features.len() != expected {
		return Err(argument_error(
			"caption index or cached text-feature geometry is invalid",
		));
	}
	let start = options.caption_index * width;
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

fn parse_options() -> Result<Options, Box<dyn Error>> {
	let mut options = Options {
		model: PathBuf::new(),
		dataset: "data/humanMl3d/Cmp".into(),
		split: "train".into(),
		output: "var/alm".into(),
		count: 3,
		temperature: 1.0,
		max_len: 64,
		seed: 42,
		conditioning_clip: 0,
		caption_index: 0,
		prompt: None,
	};
	let mut arguments = env::args().skip(1);
	while let Some(argument) = arguments.next() {
		let mut value = || {
			arguments
				.next()
				.ok_or_else(|| argument_error(&format!("{argument} requires a value")))
		};
		match argument.as_str() {
			"--model" => options.model = value()?.into(),
			"--dataset" => options.dataset = value()?.into(),
			"--split" => options.split = value()?,
			"--out-dir" => options.output = value()?.into(),
			"--gen-count" => options.count = parse(&value()?, &argument)?,
			"--gen-temp" => options.temperature = parse(&value()?, &argument)?,
			"--gen-len" => options.max_len = parse(&value()?, &argument)?,
			"--seed" => options.seed = parse(&value()?, &argument)?,
			"--conditioning-clip" => options.conditioning_clip = parse(&value()?, &argument)?,
			"--caption-index" => options.caption_index = parse(&value()?, &argument)?,
			"--prompt" => options.prompt = Some(value()?),
			"--help" | "-h" => {
				println!(
					"genalm --model MODEL.oam [--dataset DIR] [--split NAME] [--out-dir DIR] [--gen-count N] [--gen-temp F] [--gen-len N] [--seed N] [--conditioning-clip N] [--caption-index N] [--prompt TEXT]"
				);
				std::process::exit(0);
			}
			_ => return Err(argument_error(&format!("unknown option {argument}"))),
		}
	}
	if options.model.as_os_str().is_empty() {
		return Err(argument_error("--model is required"));
	}
	Ok(options)
}

fn parse<T: std::str::FromStr>(text: &str, option: &str) -> Result<T, Box<dyn Error>> {
	text.parse()
		.map_err(|_| argument_error(&format!("{option} has an invalid value")))
}

fn argument_error(message: &str) -> Box<dyn Error> {
	std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}
