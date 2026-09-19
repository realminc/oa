use std::{error::Error, path::PathBuf, rc::Rc};

use oa::Cli;
use oa::sdk::{
	data::HumanMl3dDataset,
	ml::alm::{
		AlmFfnType, AlmPrior, AlmPriorConfig, AlmTokenizer, AlmTokenizerConfig, ClipText,
		training::{
			AlmTrainingConfig, AlmValidation, PriorTrainingConfig, PriorValidationConfig,
			StageCheckpointConfig, TokenizerTrainingConfig, TokenizerValidationConfig, train_alm,
			train_alm_with_native_text, train_alm_with_validation,
		},
	},
};

// ─── config ──────────────────────────────────────────────────────────────────
//
// Paths are stored as String (ParseCliValue) and converted to PathBuf when
// passed to the SDK. String enum fields (ffn_type, text_conditioning) are
// validated and converted after parse.

struct AlmTrainConfig {
	dataset: String,
	split: String,
	validation_split: String,
	validation_batches: usize,
	output: String,
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
	commitment_beta: f32,
	ema_decay: f32,
	ema_epsilon: f32,
	dead_threshold: f32,
	model_width: usize,
	num_heads: usize,
	num_layers: usize,
	hidden_width: usize,
	max_sequence_len: usize,
	/// "dense" | "moe" | "hybrid"
	ffn_type: String,
	moe_num_experts: usize,
	moe_experts_per_token: usize,
	moe_every: usize,
	moe_balance_rate: f32,
	moe_aux_loss_alpha: f32,
	moe_router_z_loss_beta: f32,
	/// "clip" | "none"
	text_conditioning: String,
	clip_text_model: String,
	clip_merges: String,
	tokenizer_learning_rate: f32,
	tokenizer_minimum_learning_rate: f32,
	tokenizer_warmup_steps: u64,
	tokenizer_weight_decay: f32,
	prior_learning_rate: f32,
	prior_minimum_learning_rate: f32,
	prior_warmup_steps: u64,
	prior_weight_decay: f32,
	checkpoint_directory: String,
	checkpoint_save_every: u64,
	checkpoint_keep: usize,
	no_checkpoint: bool,
	resume: bool,
	restore_best: bool,
}

impl Default for AlmTrainConfig {
	fn default() -> Self {
		Self {
			dataset: "data/humanMl3d/Cmp".into(),
			split: "train".into(),
			validation_split: "val".into(),
			validation_batches: 0,
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
			commitment_beta: 0.25,
			ema_decay: 0.99,
			ema_epsilon: 1.0e-5,
			dead_threshold: 2.0,
			model_width: 384,
			num_heads: 6,
			num_layers: 6,
			hidden_width: 1536,
			max_sequence_len: 260,
			ffn_type: "dense".into(),
			moe_num_experts: 4,
			moe_experts_per_token: 2,
			moe_every: 2,
			moe_balance_rate: 1.0e-3,
			moe_aux_loss_alpha: 0.01,
			moe_router_z_loss_beta: 1.0e-3,
			text_conditioning: "clip".into(),
			clip_text_model: "var/model/ref/ClipText/ClipText.oam".into(),
			clip_merges: "var/model/ref/ClipText/merges.txt".into(),
			tokenizer_learning_rate: 2.0e-4,
			tokenizer_minimum_learning_rate: 2.0e-5,
			tokenizer_warmup_steps: 500,
			tokenizer_weight_decay: 0.0,
			prior_learning_rate: 1.0e-4,
			prior_minimum_learning_rate: 1.0e-5,
			prior_warmup_steps: 300,
			prior_weight_decay: 0.01,
			checkpoint_directory: "var/model/dev".into(),
			checkpoint_save_every: 0,
			checkpoint_keep: 5,
			no_checkpoint: false,
			resume: false,
			restore_best: false,
		}
	}
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
	let mut cli = Cli::new("trainalm", "OARS tokenizer + Transformer prior");
	cli.add_option(
		"--dataset",
		|c: &mut AlmTrainConfig| &mut c.dataset,
		"dataset directory",
	);
	cli.add_option(
		"--split",
		|c: &mut AlmTrainConfig| &mut c.split,
		"training split name",
	);
	cli.add_option(
		"--val-split",
		|c: &mut AlmTrainConfig| &mut c.validation_split,
		"validation split name",
	);
	cli.add_option(
		"--val-batches",
		|c: &mut AlmTrainConfig| &mut c.validation_batches,
		"max validation batches (0=all)",
	);
	cli.add_option(
		"--output",
		|c: &mut AlmTrainConfig| &mut c.output,
		"output model path",
	);
	cli.add_option(
		"--max-clips",
		|c: &mut AlmTrainConfig| &mut c.max_clips,
		"max clips to load (0=all)",
	);
	cli.add_option(
		"--seed",
		|c: &mut AlmTrainConfig| &mut c.seed,
		"random seed",
	);
	cli.add_option(
		"--tok-epochs",
		|c: &mut AlmTrainConfig| &mut c.tokenizer_epochs,
		"tokenizer training epochs",
	);
	cli.add_option(
		"--lm-epochs",
		|c: &mut AlmTrainConfig| &mut c.prior_epochs,
		"prior training epochs",
	);
	cli.add_option(
		"--batch",
		|c: &mut AlmTrainConfig| &mut c.batch_size,
		"batch size",
	);
	cli.add_option(
		"--seq-len",
		|c: &mut AlmTrainConfig| &mut c.sequence_len,
		"tokenizer sequence length",
	);
	cli.add_option(
		"--lm-seq-len",
		|c: &mut AlmTrainConfig| &mut c.prior_sequence_len,
		"prior sequence length",
	);
	cli.add_option(
		"--codes",
		|c: &mut AlmTrainConfig| &mut c.num_codes,
		"number of VQ codes",
	);
	cli.add_option(
		"--width",
		|c: &mut AlmTrainConfig| &mut c.tokenizer_width,
		"tokenizer width",
	);
	cli.add_option(
		"--code-dim",
		|c: &mut AlmTrainConfig| &mut c.code_dim,
		"code dimension",
	);
	cli.add_option(
		"--down-t",
		|c: &mut AlmTrainConfig| &mut c.downsample_stages,
		"downsample stages",
	);
	cli.add_option(
		"--depth",
		|c: &mut AlmTrainConfig| &mut c.depth,
		"tokenizer depth",
	);
	cli.add_option(
		"--commit-beta",
		|c: &mut AlmTrainConfig| &mut c.commitment_beta,
		"VQ commitment beta",
	);
	cli.add_option(
		"--ema-decay",
		|c: &mut AlmTrainConfig| &mut c.ema_decay,
		"EMA decay",
	);
	cli.add_option(
		"--ema-eps",
		|c: &mut AlmTrainConfig| &mut c.ema_epsilon,
		"EMA epsilon",
	);
	cli.add_option(
		"--dead-thresh",
		|c: &mut AlmTrainConfig| &mut c.dead_threshold,
		"dead code threshold",
	);
	cli.add_option(
		"--dmodel",
		|c: &mut AlmTrainConfig| &mut c.model_width,
		"prior model width",
	);
	cli.add_option(
		"--lm-heads",
		|c: &mut AlmTrainConfig| &mut c.num_heads,
		"prior attention heads",
	);
	cli.add_option(
		"--lm-layers",
		|c: &mut AlmTrainConfig| &mut c.num_layers,
		"prior layers",
	);
	cli.add_option(
		"--lm-ffn",
		|c: &mut AlmTrainConfig| &mut c.hidden_width,
		"prior FFN hidden width",
	);
	cli.add_option(
		"--lm-max-seq-len",
		|c: &mut AlmTrainConfig| &mut c.max_sequence_len,
		"prior max sequence length",
	);
	cli.add_option(
		"--lm-ffn-type",
		|c: &mut AlmTrainConfig| &mut c.ffn_type,
		"FFN type: dense|moe|hybrid",
	);
	cli.add_option(
		"--lm-moe-experts",
		|c: &mut AlmTrainConfig| &mut c.moe_num_experts,
		"MoE expert count",
	);
	cli.add_option(
		"--lm-moe-top-k",
		|c: &mut AlmTrainConfig| &mut c.moe_experts_per_token,
		"MoE top-k",
	);
	cli.add_option(
		"--lm-moe-every",
		|c: &mut AlmTrainConfig| &mut c.moe_every,
		"MoE every N layers",
	);
	cli.add_option(
		"--lm-moe-balance-rate",
		|c: &mut AlmTrainConfig| &mut c.moe_balance_rate,
		"MoE balance rate",
	);
	cli.add_option(
		"--lm-moe-aux-alpha",
		|c: &mut AlmTrainConfig| &mut c.moe_aux_loss_alpha,
		"MoE aux loss alpha",
	);
	cli.add_option(
		"--lm-moe-z-beta",
		|c: &mut AlmTrainConfig| &mut c.moe_router_z_loss_beta,
		"MoE router z-loss beta",
	);
	cli.add_option(
		"--tok-lr",
		|c: &mut AlmTrainConfig| &mut c.tokenizer_learning_rate,
		"tokenizer learning rate",
	);
	cli.add_option(
		"--tok-min-lr",
		|c: &mut AlmTrainConfig| &mut c.tokenizer_minimum_learning_rate,
		"tokenizer min lr",
	);
	cli.add_option(
		"--tok-warmup",
		|c: &mut AlmTrainConfig| &mut c.tokenizer_warmup_steps,
		"tokenizer warmup steps",
	);
	cli.add_option(
		"--tok-wd",
		|c: &mut AlmTrainConfig| &mut c.tokenizer_weight_decay,
		"tokenizer weight decay",
	);
	cli.add_option(
		"--lm-lr",
		|c: &mut AlmTrainConfig| &mut c.prior_learning_rate,
		"prior learning rate",
	);
	cli.add_option(
		"--lm-min-lr",
		|c: &mut AlmTrainConfig| &mut c.prior_minimum_learning_rate,
		"prior min lr",
	);
	cli.add_option(
		"--lm-warmup",
		|c: &mut AlmTrainConfig| &mut c.prior_warmup_steps,
		"prior warmup steps",
	);
	cli.add_option(
		"--lm-wd",
		|c: &mut AlmTrainConfig| &mut c.prior_weight_decay,
		"prior weight decay",
	);
	cli.add_option(
		"--text-conditioning",
		|c: &mut AlmTrainConfig| &mut c.text_conditioning,
		"text conditioning: clip|none",
	);
	cli.add_option(
		"--clip-text-model",
		|c: &mut AlmTrainConfig| &mut c.clip_text_model,
		"CLIP text model path",
	);
	cli.add_option(
		"--clip-merges",
		|c: &mut AlmTrainConfig| &mut c.clip_merges,
		"CLIP BPE merges path",
	);
	cli.add_option(
		"--checkpoint-dir",
		|c: &mut AlmTrainConfig| &mut c.checkpoint_directory,
		"checkpoint directory",
	);
	cli.add_option(
		"--model-dir",
		|c: &mut AlmTrainConfig| &mut c.checkpoint_directory,
		"checkpoint directory (alias)",
	);
	cli.add_option(
		"--checkpoint-save-every",
		|c: &mut AlmTrainConfig| &mut c.checkpoint_save_every,
		"save every N steps",
	);
	cli.add_option(
		"--ckpt-save-every",
		|c: &mut AlmTrainConfig| &mut c.checkpoint_save_every,
		"save every N steps (alias)",
	);
	cli.add_option(
		"--checkpoint-keep",
		|c: &mut AlmTrainConfig| &mut c.checkpoint_keep,
		"max checkpoints to keep",
	);
	cli.add_option(
		"--ckpt-max-keep",
		|c: &mut AlmTrainConfig| &mut c.checkpoint_keep,
		"max checkpoints to keep (alias)",
	);
	cli.add_flag(
		"--resume",
		|c: &mut AlmTrainConfig, v| c.resume = v,
		"resume from latest checkpoint",
	);
	cli.add_flag(
		"--no-checkpoint",
		|c: &mut AlmTrainConfig, v| c.no_checkpoint = v,
		"disable checkpointing",
	);
	cli.add_flag(
		"--restore-best",
		|c: &mut AlmTrainConfig, v| c.restore_best = v,
		"restore best checkpoint after training",
	);
	cli.add_flag(
		"--ckpt-restore-best",
		|c: &mut AlmTrainConfig, v| c.restore_best = v,
		"restore best (alias)",
	);
	cli.add_flag(
		"--no-restore-best",
		|c: &mut AlmTrainConfig, v| c.restore_best = !v,
		"do not restore best",
	);
	cli.add_flag(
		"--unconditional",
		|c: &mut AlmTrainConfig, v| {
			if v {
				c.text_conditioning = "none".into();
			}
		},
		"disable text conditioning",
	);
	cli.set_epilog(
		"trainalm --dataset DIR --output MODEL.oam [--split train] [--val-split val]\n\
		 [--val-batches N] [--max-clips N]\n\
		 --tok-epochs N --lm-epochs N --batch N --seq-len N --lm-seq-len N\n\
		 --codes N --width N --code-dim N --down-t N --depth N\n\
		 [--commit-beta F] [--ema-decay F] [--ema-eps F] [--dead-thresh F]\n\
		 --dmodel N --lm-heads N --lm-layers N --lm-ffn N\n\
		 --lm-max-seq-len N --lm-ffn-type dense|moe|hybrid\n\
		 [--lm-moe-experts N] [--lm-moe-top-k N] [--lm-moe-every N]\n\
		 [--lm-moe-balance-rate F] [--lm-moe-aux-alpha F] [--lm-moe-z-beta F]\n\
		 [--tok-lr F] [--tok-min-lr F] [--tok-warmup N] [--tok-wd F]\n\
		 [--lm-lr F] [--lm-min-lr F] [--lm-warmup N] [--lm-wd F] [--seed N]\n\
		 [--text-conditioning clip|none] [--clip-text-model MODEL.oam]\n\
		 [--clip-merges merges.txt] [--unconditional]\n\
		 [--checkpoint-dir DIR] [--checkpoint-save-every N] [--checkpoint-keep N]\n\
		 [--resume] [--restore-best|--no-restore-best] [--no-checkpoint]",
	);
	if !cli.parse() {
		return Ok(());
	}
	let cfg = cli.into_config();

	// Validate and convert string enum fields.
	let ffn_type = match cfg.ffn_type.as_str() {
		"dense" => AlmFfnType::Dense,
		"moe" => AlmFfnType::Moe,
		"hybrid" => AlmFfnType::Hybrid,
		other => {
			return Err(argument_error(&format!(
				"--lm-ffn-type must be dense, moe, or hybrid; got {other:?}"
			)));
		}
	};
	let use_text = match cfg.text_conditioning.as_str() {
		"clip" => true,
		"none" => false,
		other => {
			return Err(argument_error(&format!(
				"--text-conditioning must be clip or none; got {other:?}"
			)));
		}
	};
	if cfg.resume && cfg.no_checkpoint {
		return Err(argument_error("--resume conflicts with --no-checkpoint"));
	}
	let checkpoint_enabled = !cfg.no_checkpoint;

	// Convert String paths to PathBuf.
	let dataset_path = PathBuf::from(&cfg.dataset);
	let output_path = PathBuf::from(&cfg.output);
	let ckpt_dir = PathBuf::from(&cfg.checkpoint_directory);
	let clip_model = PathBuf::from(&cfg.clip_text_model);
	let clip_merges = PathBuf::from(&cfg.clip_merges);

	let dataset = HumanMl3dDataset::open_cmp(&dataset_path, &cfg.split, cfg.max_clips)?;
	let validation =
		match HumanMl3dDataset::open_cmp(&dataset_path, &cfg.validation_split, cfg.max_clips) {
			Ok(dataset) => Some(dataset),
			Err(error) => {
				eprintln!(
					"validation split '{}' unavailable; validation disabled: {error}",
					cfg.validation_split
				);
				None
			}
		};
	let engine = oa::Engine::new()?;
	let native_text = if use_text {
		let model = Rc::new(ClipText::load_model(&engine, &clip_model)?);
		let merges = std::fs::read(&clip_merges)?;
		Some((model, merges))
	} else {
		None
	};
	let tokenizer = Rc::new(AlmTokenizer::with_seed(
		&engine,
		AlmTokenizerConfig {
			input_dim: dataset.feature_dim(),
			width: cfg.tokenizer_width,
			code_dim: cfg.code_dim,
			num_codes: cfg.num_codes,
			downsample_stages: cfg.downsample_stages,
			depth: cfg.depth,
			commitment_beta: cfg.commitment_beta,
			ema_decay: cfg.ema_decay,
			ema_epsilon: cfg.ema_epsilon,
			dead_threshold: cfg.dead_threshold,
		},
		cfg.seed,
	)?);
	let text_feature_dim = native_text
		.as_ref()
		.map_or(0, |(model, _)| model.config().projection_dim);
	let mut prior_config = AlmPriorConfig {
		model_width: cfg.model_width,
		num_heads: cfg.num_heads,
		num_layers: cfg.num_layers,
		hidden_width: cfg.hidden_width,
		text_feature_dim,
		ffn_type,
		moe_num_experts: cfg.moe_num_experts,
		moe_experts_per_token: cfg.moe_experts_per_token,
		moe_every: cfg.moe_every,
		moe_balance_rate: cfg.moe_balance_rate,
		moe_aux_loss_alpha: cfg.moe_aux_loss_alpha,
		moe_router_z_loss_beta: cfg.moe_router_z_loss_beta,
		batch_size: cfg.batch_size,
		sequence_length: cfg
			.prior_sequence_len
			.checked_add(1 + usize::from(text_feature_dim > 0))
			.ok_or_else(|| argument_error("prior sequence length overflows usize"))?,
		max_sequence_length: cfg.max_sequence_len,
		learning_rate: cfg.prior_learning_rate,
		num_epochs: usize::try_from(cfg.prior_epochs)
			.map_err(|_| argument_error("prior epoch count exceeds usize"))?,
		..AlmPriorConfig::default()
	};
	prior_config.sync_vocab(cfg.num_codes)?;
	let prior = Rc::new(AlmPrior::with_seed(
		&engine,
		prior_config,
		cfg.seed.wrapping_add(1),
	)?);

	println!("\ntrainalm — OARS tokenizer + Transformer prior");
	println!(
		"  data: {} · {} clips · {} frames · feature dim {}",
		dataset_path.display(),
		dataset.len(),
		dataset.total_frames(),
		dataset.feature_dim()
	);
	if let Some(validation) = validation.as_ref() {
		println!(
			"  validation: {} · {} clips · {} frames · max batches {}",
			cfg.validation_split,
			validation.len(),
			validation.total_frames(),
			if cfg.validation_batches == 0 {
				"all".to_owned()
			} else {
				cfg.validation_batches.to_string()
			}
		);
	}
	println!(
		"  tokenizer: {} epochs · batch {} · {} frames · {} codes",
		cfg.tokenizer_epochs, cfg.batch_size, cfg.sequence_len, cfg.num_codes
	);
	println!(
		"    AdamW: lr {:.3e} → {:.3e} · warmup {} · weight decay {:.3e}",
		cfg.tokenizer_learning_rate,
		cfg.tokenizer_minimum_learning_rate,
		cfg.tokenizer_warmup_steps,
		cfg.tokenizer_weight_decay
	);
	println!(
		"  prior: {} epochs · batch {} · {} token pairs · text {}",
		cfg.prior_epochs,
		cfg.batch_size,
		cfg
			.prior_sequence_len
			.checked_add(1)
			.ok_or_else(|| argument_error("prior sequence length overflows usize"))?,
		if text_feature_dim == 0 {
			"none"
		} else {
			"native CLIP"
		}
	);
	println!(
		"    AdamW: lr {:.3e} → {:.3e} · warmup {} · weight decay {:.3e}",
		cfg.prior_learning_rate,
		cfg.prior_minimum_learning_rate,
		cfg.prior_warmup_steps,
		cfg.prior_weight_decay
	);
	if native_text.is_some() {
		println!(
			"  CLIP: {} · merges {}",
			clip_model.display(),
			clip_merges.display()
		);
	}

	let stage_checkpoint = |model_name: &str| {
		checkpoint_enabled.then(|| StageCheckpointConfig {
			directory: ckpt_dir.clone(),
			model_name: model_name.to_owned(),
			context: String::new(),
			max_keep: cfg.checkpoint_keep,
			save_every: cfg.checkpoint_save_every,
			resume: cfg.resume,
			restore_best: cfg.restore_best,
			verbose: true,
		})
	};
	let prior_window_len = cfg
		.prior_sequence_len
		.checked_add(1)
		.ok_or_else(|| argument_error("prior window length overflows usize"))?;
	let training_config = AlmTrainingConfig {
		tokenizer: TokenizerTrainingConfig {
			epochs: cfg.tokenizer_epochs,
			batch_size: cfg.batch_size,
			sequence_len: cfg.sequence_len,
			learning_rate: cfg.tokenizer_learning_rate,
			minimum_learning_rate: cfg.tokenizer_minimum_learning_rate,
			warmup_steps: cfg.tokenizer_warmup_steps,
			weight_decay: cfg.tokenizer_weight_decay,
			checkpoint: stage_checkpoint("AlmTokenizer"),
			..TokenizerTrainingConfig::default()
		},
		prior: PriorTrainingConfig {
			epochs: cfg.prior_epochs,
			batch_size: cfg.batch_size,
			window_len: prior_window_len,
			learning_rate: cfg.prior_learning_rate,
			minimum_learning_rate: cfg.prior_minimum_learning_rate,
			warmup_steps: cfg.prior_warmup_steps,
			weight_decay: cfg.prior_weight_decay,
			checkpoint: stage_checkpoint("AlmPrior"),
			..PriorTrainingConfig::default()
		},
		text_seed: cfg.seed,
	};
	let validation_policy = validation.as_ref().map(|validation| AlmValidation {
		dataset: validation,
		tokenizer: TokenizerValidationConfig {
			sequence_len: cfg.sequence_len,
			batch_size: cfg.batch_size,
			max_batches: cfg.validation_batches,
		},
		prior: PriorValidationConfig {
			window_len: prior_window_len,
			batch_size: cfg.batch_size,
			max_batches: cfg.validation_batches,
		},
	});
	let report = if let Some((clip_text, clip_merges_bytes)) = native_text {
		train_alm_with_native_text(
			&engine,
			&dataset,
			validation_policy,
			tokenizer,
			prior,
			clip_text,
			&clip_merges_bytes,
			training_config,
		)?
	} else {
		match validation_policy {
			Some(validation) => train_alm_with_validation(
				&engine,
				&dataset,
				validation,
				tokenizer,
				prior,
				training_config,
			)?,
			None => train_alm(&engine, &dataset, tokenizer, prior, training_config)?,
		}
	};
	if let Some(parent) = output_path.parent() {
		std::fs::create_dir_all(parent)?;
	}
	report.model.save_bundle(&output_path)?;
	println!(
		"\nALM complete: {} tokens · tokenizer {:.6} → {:.6} · prior {:.6} → {:.6}",
		report.corpus_tokens,
		report.tokenizer.initial_reconstruction_loss,
		report.tokenizer.final_reconstruction_loss,
		report.prior.initial_loss,
		report.prior.final_loss
	);
	if report.tokenizer.resumed_from_step > 0 || report.prior.resumed_from_step > 0 {
		println!(
			"  resumed: tokenizer step {} · prior step {}",
			report.tokenizer.resumed_from_step, report.prior.resumed_from_step
		);
	}
	if let Some(validation) = report.tokenizer.validation {
		println!(
			"  tokenizer validation: loss {:.6} · velocity {:.6} · MPJPE {:.3} cm · contact {:.2}% · foot skate {:.3} cm/frame · codes {}/{} · perplexity {:.2}",
			validation.reconstruction_loss,
			validation.velocity_loss,
			validation.mpjpe_cm,
			validation.contact_accuracy * 100.0,
			validation.foot_skate_cm_per_frame,
			validation.live_codes,
			cfg.num_codes,
			validation.codebook_perplexity,
		);
	}
	if let Some(validation) = report.prior.validation {
		println!(
			"  prior validation: loss {:.6} · perplexity {:.3} · token accuracy {:.2}% · EOS accuracy {:.2}%",
			validation.loss,
			validation.perplexity,
			validation.token_accuracy * 100.0,
			validation.eos_accuracy * 100.0,
		);
	}
	println!("saved {}", output_path.display());
	Ok(())
}

fn argument_error(message: &str) -> Box<dyn Error> {
	std::io::Error::new(std::io::ErrorKind::InvalidInput, message).into()
}
