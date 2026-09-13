use std::{env, error::Error, path::PathBuf, rc::Rc};

use oa::sdk::{
	data::HumanMl3dDataset,
	ml::alm::{
		AlmFfnType, AlmPrior, AlmPriorConfig, AlmTokenizer, AlmTokenizerConfig,
		training::{AlmTrainingConfig, PriorTrainingConfig, TokenizerTrainingConfig, train_alm},
	},
};

struct Options {
	dataset: PathBuf,
	split: String,
	output: PathBuf,
	max_clips: usize,
	seed: u64,
	tokenizer_epochs: u64,
	prior_epochs: u64,
	batch_size: usize,
	sequence_len: usize,
	prior_sequence_len: usize,
	num_codes: usize,
	tokenizer_width: usize,
	code_dim: usize,
	downsample_stages: usize,
	depth: usize,
	model_width: usize,
	num_heads: usize,
	num_layers: usize,
	hidden_width: usize,
	max_sequence_len: usize,
	ffn_type: AlmFfnType,
	text_conditioning: bool,
}

impl Default for Options {
	fn default() -> Self {
		Self {
			dataset: "data/humanMl3d/Cmp".into(),
			split: "train".into(),
			output: "var/model/dev/Alm/Alm.oam".into(),
			max_clips: 0,
			seed: 42,
			tokenizer_epochs: 50,
			prior_epochs: 50,
			batch_size: 32,
			sequence_len: 64,
			prior_sequence_len: 64,
			num_codes: 512,
			tokenizer_width: 384,
			code_dim: 256,
			downsample_stages: 2,
			depth: 3,
			model_width: 384,
			num_heads: 6,
			num_layers: 6,
			hidden_width: 1536,
			max_sequence_len: 260,
			ffn_type: AlmFfnType::Dense,
			text_conditioning: true,
		}
	}
}

fn main() -> Result<(), Box<dyn Error>> {
	let options = parse_options()?;
	let dataset = HumanMl3dDataset::open_cmp(&options.dataset, &options.split, options.max_clips)?;
	let engine = oa::Engine::new()?;
	let tokenizer = Rc::new(AlmTokenizer::with_seed(
		&engine,
		AlmTokenizerConfig {
			input_dim: dataset.feature_dim(),
			width: options.tokenizer_width,
			code_dim: options.code_dim,
			num_codes: options.num_codes,
			downsample_stages: options.downsample_stages,
			depth: options.depth,
			commitment_beta: 0.25,
			ema_decay: 0.99,
			ema_epsilon: 1.0e-5,
			dead_threshold: 2.0,
		},
		options.seed,
	)?);
	let text_feature_dim = if options.text_conditioning {
		dataset.text_feature_dim()
	} else {
		0
	};
	if options.text_conditioning && text_feature_dim == 0 {
		return Err(argument_error(
			"--text-conditioning requires the dataset's oa_clip_text_v1 cache",
		));
	}
	let mut prior_config = AlmPriorConfig {
		model_width: options.model_width,
		num_heads: options.num_heads,
		num_layers: options.num_layers,
		hidden_width: options.hidden_width,
		text_feature_dim,
		ffn_type: options.ffn_type,
		batch_size: options.batch_size,
		sequence_length: options
			.prior_sequence_len
			.checked_add(1 + usize::from(text_feature_dim > 0))
			.ok_or_else(|| argument_error("prior sequence length overflows usize"))?,
		max_sequence_length: options.max_sequence_len,
		..AlmPriorConfig::default()
	};
	prior_config.sync_vocab(options.num_codes)?;
	let prior = Rc::new(AlmPrior::with_seed(
		&engine,
		prior_config,
		options.seed.wrapping_add(1),
	)?);

	println!("\ntrainalm — OARS tokenizer + Transformer prior");
	println!(
		"  data: {} · {} clips · {} frames · feature dim {}",
		options.dataset.display(),
		dataset.len(),
		dataset.total_frames(),
		dataset.feature_dim()
	);
	println!(
		"  tokenizer: {} epochs · batch {} · {} frames · {} codes",
		options.tokenizer_epochs, options.batch_size, options.sequence_len, options.num_codes
	);
	println!(
		"  prior: {} epochs · batch {} · {} token pairs · text {}",
		options.prior_epochs,
		options.batch_size,
		options
			.prior_sequence_len
			.checked_add(1)
			.ok_or_else(|| argument_error("prior sequence length overflows usize"))?,
		if text_feature_dim == 0 {
			"none"
		} else {
			"CLIP"
		}
	);

	let report = train_alm(
		&engine,
		&dataset,
		tokenizer,
		prior,
		AlmTrainingConfig {
			tokenizer: TokenizerTrainingConfig {
				epochs: options.tokenizer_epochs,
				batch_size: options.batch_size,
				sequence_len: options.sequence_len,
				..TokenizerTrainingConfig::default()
			},
			prior: PriorTrainingConfig {
				epochs: options.prior_epochs,
				batch_size: options.batch_size,
				window_len: options
					.prior_sequence_len
					.checked_add(1)
					.ok_or_else(|| argument_error("prior window length overflows usize"))?,
				..PriorTrainingConfig::default()
			},
			text_seed: options.seed,
		},
	)?;
	if let Some(parent) = options.output.parent() {
		std::fs::create_dir_all(parent)?;
	}
	report.model.save_bundle(&options.output)?;
	println!(
		"\nALM complete: {} tokens · tokenizer {:.6} → {:.6} · prior {:.6} → {:.6}",
		report.corpus_tokens,
		report.tokenizer.initial_reconstruction_loss,
		report.tokenizer.final_reconstruction_loss,
		report.prior.initial_loss,
		report.prior.final_loss
	);
	println!("saved {}", options.output.display());
	Ok(())
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
	let mut options = Options::default();
	let mut arguments = env::args().skip(1);
	while let Some(argument) = arguments.next() {
		let mut value = || {
			arguments
				.next()
				.ok_or_else(|| argument_error(&format!("{argument} requires a value")))
		};
		match argument.as_str() {
			"--dataset" => options.dataset = value()?.into(),
			"--split" => options.split = value()?,
			"--output" => options.output = value()?.into(),
			"--max-clips" => options.max_clips = parse(&value()?, &argument)?,
			"--seed" => options.seed = parse(&value()?, &argument)?,
			"--tok-epochs" => options.tokenizer_epochs = parse(&value()?, &argument)?,
			"--lm-epochs" => options.prior_epochs = parse(&value()?, &argument)?,
			"--batch" => options.batch_size = parse(&value()?, &argument)?,
			"--seq-len" => options.sequence_len = parse(&value()?, &argument)?,
			"--lm-seq-len" => options.prior_sequence_len = parse(&value()?, &argument)?,
			"--codes" => options.num_codes = parse(&value()?, &argument)?,
			"--width" => options.tokenizer_width = parse(&value()?, &argument)?,
			"--code-dim" => options.code_dim = parse(&value()?, &argument)?,
			"--down-t" => options.downsample_stages = parse(&value()?, &argument)?,
			"--depth" => options.depth = parse(&value()?, &argument)?,
			"--dmodel" => options.model_width = parse(&value()?, &argument)?,
			"--lm-heads" => options.num_heads = parse(&value()?, &argument)?,
			"--lm-layers" => options.num_layers = parse(&value()?, &argument)?,
			"--lm-ffn" => options.hidden_width = parse(&value()?, &argument)?,
			"--lm-max-seq-len" => options.max_sequence_len = parse(&value()?, &argument)?,
			"--lm-ffn-type" => {
				options.ffn_type = match value()?.as_str() {
					"dense" => AlmFfnType::Dense,
					"moe" => AlmFfnType::Moe,
					"hybrid" => AlmFfnType::Hybrid,
					_ => {
						return Err(argument_error(
							"--lm-ffn-type must be dense, moe, or hybrid",
						));
					}
				}
			}
			"--unconditional" => options.text_conditioning = false,
			"--help" | "-h" => {
				print_help();
				std::process::exit(0);
			}
			_ => return Err(argument_error(&format!("unknown option {argument}"))),
		}
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

fn print_help() {
	println!(
		"trainalm --dataset DIR --output MODEL.oam [--split train] [--max-clips N]\n\
		 --tok-epochs N --lm-epochs N --batch N --seq-len N --lm-seq-len N\n\
		 --codes N --width N --code-dim N --down-t N --depth N\n\
		 --dmodel N --lm-heads N --lm-layers N --lm-ffn N\n\
		 --lm-max-seq-len N --lm-ffn-type dense|moe|hybrid [--unconditional] [--seed N]"
	);
}
