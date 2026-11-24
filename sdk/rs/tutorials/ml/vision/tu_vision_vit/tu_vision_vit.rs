use oa::ml::{ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary};
use oa::{Cli, LogMetrics};

// Synthetic 3-class image task: horizontal bar, vertical bar, or diagonal
// stripe on a 16x16 grayscale grid with light noise. Fixed geometry keeps
// the contract free of padding; patches embed through a shared Linear.
const IMAGE_SIDE: usize = 16;
const PATCH_SIDE: usize = 4;
const NUM_PATCHES: usize = (IMAGE_SIDE / PATCH_SIDE) * (IMAGE_SIDE / PATCH_SIDE); // 16
const TOKENS: usize = NUM_PATCHES + 1; // 16 patches + 1 class token
const NUM_CLASSES: usize = 3;
const MODEL_WIDTH: usize = 32;
const TRAINING_STEPS: usize = 200;
const BATCH_SIZE: usize = 32;
const EVAL_IMAGES: usize = 64;

#[derive(Default)]
struct VitConfig {
	steps: u64,
	batch: u64,
	lr: f32,
}

struct ShapeRng(u64);

impl ShapeRng {
	fn new(seed: u64) -> Self {
		Self(seed | 1)
	}

	fn next(&mut self) -> u64 {
		self.0 = self
			.0
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		self.0 >> 11
	}

	fn unit(&mut self) -> f32 {
		(self.next() % (1 << 24)) as f32 / ((1 << 24) as f32)
	}

	fn below(&mut self, bound: usize) -> usize {
		(self.next() % bound as u64) as usize
	}
}

// Emit `batch` images flattened as patch-major rows: batch * NUM_PATCHES
// rows of PATCH_SIDE*PATCH_SIDE pixels, plus one class label each.
fn next_batch(rng: &mut ShapeRng, batch: usize) -> (Vec<f32>, Vec<u32>) {
	let mut patches = vec![0.0_f32; batch * NUM_PATCHES * PATCH_SIDE * PATCH_SIDE];
	let mut labels = Vec::with_capacity(batch);
	for image in 0..batch {
		let class = rng.below(NUM_CLASSES) as u32;
		labels.push(class);
		let mut pixels = vec![0.0_f32; IMAGE_SIDE * IMAGE_SIDE];
		for pixel in pixels.iter_mut() {
			*pixel = 0.05 * rng.unit();
		}
		match class {
			0 => {
				let row = 2 + rng.below(IMAGE_SIDE - 4);
				for column in 0..IMAGE_SIDE {
					pixels[row * IMAGE_SIDE + column] = 0.85 + 0.1 * rng.unit();
				}
			}
			1 => {
				let column = 2 + rng.below(IMAGE_SIDE - 4);
				for row in 0..IMAGE_SIDE {
					pixels[row * IMAGE_SIDE + column] = 0.85 + 0.1 * rng.unit();
				}
			}
			_ => {
				let offset = 1 + rng.below(2);
				for diagonal in 0..IMAGE_SIDE {
					let column = diagonal + offset;
					if column < IMAGE_SIDE {
						pixels[diagonal * IMAGE_SIDE + column] = 0.85 + 0.1 * rng.unit();
					}
				}
			}
		}
		let base = image * NUM_PATCHES * PATCH_SIDE * PATCH_SIDE;
		for patch in 0..NUM_PATCHES {
			let patch_row = (patch / (IMAGE_SIDE / PATCH_SIDE)) * PATCH_SIDE;
			let patch_col = (patch % (IMAGE_SIDE / PATCH_SIDE)) * PATCH_SIDE;
			for y in 0..PATCH_SIDE {
				for x in 0..PATCH_SIDE {
					patches[base + patch * PATCH_SIDE * PATCH_SIDE + y * PATCH_SIDE + x] =
						pixels[(patch_row + y) * IMAGE_SIDE + patch_col + x];
				}
			}
		}
	}
	(patches, labels)
}

struct Vit {
	patch_projection: oa::ml::nn::Linear,
	class_embedding: oa::ml::nn::Embedding,
	position_embedding: oa::ml::nn::Embedding,
	blocks: Vec<oa::ml::nn::TransformerBlock>,
	final_norm: oa::ml::nn::LayerNorm,
	head: oa::ml::nn::Linear,
}

impl Vit {
	fn new(engine: &oa::Engine, seed: u64) -> oa::Result<Self> {
		let patch_projection =
			oa::ml::nn::Linear::new_with_seed_on(engine, PATCH_SIDE * PATCH_SIDE, MODEL_WIDTH, seed)?;
		let class_embedding =
			oa::ml::nn::Embedding::new_with_seed_on(engine, 1, MODEL_WIDTH, seed.wrapping_add(1))?;
		let position_embedding =
			oa::ml::nn::Embedding::new_with_seed_on(engine, TOKENS, MODEL_WIDTH, seed.wrapping_add(2))?;
		let mut blocks = Vec::new();
		for layer in 0..2_usize {
			let block = oa::ml::nn::TransformerBlock::new_with_seed_on(
				engine,
				MODEL_WIDTH,
				64,
				TOKENS,
				4,
				1.0e-5,
				seed.wrapping_add(3 + layer as u64),
			)?;
			block.set_attention_mode(oa::ml::nn::AttentionMode::Bidirectional);
			blocks.push(block);
		}
		let final_norm = oa::ml::nn::LayerNorm::new_on(engine, MODEL_WIDTH, 1.0e-5)?;
		let head =
			oa::ml::nn::Linear::new_with_seed_on(engine, MODEL_WIDTH, NUM_CLASSES, seed.wrapping_add(7))?;
		Ok(Self {
			patch_projection,
			class_embedding,
			position_embedding,
			blocks,
			final_norm,
			head,
		})
	}

	fn all_parameters(&self) -> oa::Result<Vec<oa::ml::Parameter>> {
		let mut parameters = self.patch_projection.all_parameters()?;
		parameters.extend(self.class_embedding.all_parameters()?);
		parameters.extend(self.position_embedding.all_parameters()?);
		for block in &self.blocks {
			parameters.extend(block.all_parameters()?);
		}
		parameters.extend(self.final_norm.all_parameters()?);
		parameters.extend(self.head.all_parameters()?);
		Ok(parameters)
	}

	fn num_parameters(&self) -> oa::Result<usize> {
		Ok(
			self
				.all_parameters()?
				.iter()
				.map(|parameter| parameter.data().shape().iter().product::<usize>())
				.sum(),
		)
	}

	// `patches` is [B*NUM_PATCHES, PATCH_SIDE*PATCH_SIDE]; logits are [B, C].
	fn forward(&self, patches: &oa::Matrix, batch: usize) -> oa::Result<oa::Matrix> {
		let cls_index = oa::Matrix::new_from_slice([1], &[0_u32])?;
		let position_pattern = (0..batch)
			.flat_map(|_| 0..TOKENS as u32)
			.collect::<Vec<u32>>();
		let position_indices = oa::Matrix::new_from_slice([batch * TOKENS], &position_pattern)?;
		let patch_tokens = self.patch_projection.forward(patches)?;
		let mut sequence_parts = Vec::with_capacity(batch * 2);
		for item in 0..batch {
			sequence_parts.push(self.class_embedding.forward(&cls_index)?);
			sequence_parts.push(oa::matrix::slice(
				&patch_tokens,
				0,
				(item * NUM_PATCHES) as i64,
				((item + 1) * NUM_PATCHES) as i64,
			)?);
		}
		let sequence = oa::matrix::concat(&sequence_parts, 0)?;
		let positions = self.position_embedding.forward(&position_indices)?;
		let mut hidden = oa::matrix::add(&sequence, &positions)?;
		for block in &self.blocks {
			hidden = block.forward(&hidden)?;
		}
		let hidden = self.final_norm.forward(&hidden)?;
		let mut class_rows = Vec::with_capacity(batch);
		for item in 0..batch {
			class_rows.push(oa::matrix::slice(
				&hidden,
				0,
				(item * TOKENS) as i64,
				(item * TOKENS + 1) as i64,
			)?);
		}
		let class_tokens = oa::matrix::concat(&class_rows, 0)?;
		self.head.forward(&class_tokens)
	}
}

fn accuracy(logits: &oa::Matrix, labels: &[u32], batch: usize) -> oa::Result<f32> {
	let values = logits.read_f32()?;
	let correct = (0..batch)
		.filter(|item| {
			let row = &values[item * NUM_CLASSES..(item + 1) * NUM_CLASSES];
			let prediction = row
				.iter()
				.enumerate()
				.max_by(|left, right| left.1.total_cmp(right.1))
				.map(|(index, _)| index as u32)
				.expect("nonempty class logits");
			prediction == labels[*item]
		})
		.count();
	Ok(correct as f32 / batch as f32)
}

fn main() -> oa::Result<()> {
	let mut cli = Cli::new("tu_vision_vit", "OA Tutorial — Vision Transformer");
	cli.add_option(
		"--steps",
		|c: &mut VitConfig| &mut c.steps,
		"training steps",
	);
	cli.add_option("--batch", |c: &mut VitConfig| &mut c.batch, "batch size");
	cli.add_option("--lr", |c: &mut VitConfig| &mut c.lr, "learning rate");
	if !cli.parse() {
		return Ok(());
	}
	let steps = if cli.config().steps > 0 {
		cli.config().steps as usize
	} else {
		TRAINING_STEPS
	};
	let batch = if cli.config().batch > 0 {
		cli.config().batch as usize
	} else {
		BATCH_SIZE
	};
	let lr = if cli.config().lr > 0.0 {
		cli.config().lr
	} else {
		0.005
	};

	let engine = oa::Engine::new()?;
	oa::Engine::set_default(engine.clone());
	let model = Vit::new(&engine, 0x7B17)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, lr)?;
	let mut rng = ShapeRng::new(0x5EED_7B17);

	println!("\n╔══════════════════════════════════════════════════════════════════╗");
	println!("║  OA Tutorial — Vision Transformer · patch + bidirectional (autograd)║");
	println!("╚══════════════════════════════════════════════════════════════════╝\n");
	println!("task: classify a 16x16 image as horizontal bar, vertical bar, or diagonal");
	println!("model: patch Linear + class token + learned positions + 2 bidirectional blocks");
	println!(
		"params: {}    Optimizer: AdamW(lr={lr})\n",
		model.num_parameters()?
	);
	println!("training: {steps} steps · batch={batch} · tokens={TOKENS} · D={MODEL_WIDTH}");

	let mut loss_metric = LossMetric::default();
	let mut progress = ProgressBar::default();
	let mut summary_callback = TrainingSummary::default();
	let mut training = ItTraining::new(
		&engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: steps as u64,
			batch_size: batch as u64,
			sequence_length: (NUM_PATCHES * PATCH_SIDE * PATCH_SIDE) as u64,
			sequence_unit: "pixel".into(),
			timer_name: "vit_training_step".into(),
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary_callback);

	let mut metrics = LogMetrics::new();
	let log_dir = oa::Path::var_rel("log/vision/vit");
	if let Err(e) = metrics.open(&log_dir) {
		eprintln!("[OA] metrics unavailable: {e}");
	}

	let mut initial_loss = 0.0_f32;
	for _ in 0..steps {
		assert!(training.step(
			|| {
				let (patches, labels) = next_batch(&mut rng, batch);
				Ok((
					oa::Matrix::new_from_f32([batch * NUM_PATCHES, PATCH_SIDE * PATCH_SIDE], &patches)?,
					oa::Matrix::new_from_slice([batch], &labels)?,
				))
			},
			|(patches, labels)| {
				let tape = oa::ml::GradientTape::new();
				let logits = model.forward(&patches, batch)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &labels)?;
				tape.backward(&loss)?;
				Ok(loss)
			},
		)?);
		let snap = training.snapshot();
		if snap.step_count() == 1 {
			initial_loss = snap.last_loss().unwrap_or_default();
		}
		if let Some(loss) = snap.last_loss() {
			metrics.log_scalar("loss", snap.step_count() as i64, loss as f64);
		}
	}
	let program_diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let training_result = training.finish()?;
	let final_training_loss = training_result.last_loss().unwrap_or_default();

	let (eval_patches, eval_labels) = next_batch(&mut rng, EVAL_IMAGES);
	let patches = oa::Matrix::new_from_f32(
		[EVAL_IMAGES * NUM_PATCHES, PATCH_SIDE * PATCH_SIDE],
		&eval_patches,
	)?;
	let labels = oa::Matrix::new_from_slice([EVAL_IMAGES], &eval_labels)?;
	let logits = model.forward(&patches, EVAL_IMAGES)?;
	let evaluation_loss = oa::ml::loss::cross_entropy(&logits, &labels)?.read_f32()?[0];
	let evaluation_accuracy = accuracy(&logits, &eval_labels, EVAL_IMAGES)?;

	println!("\nEvaluation:");
	println!(
		"  random-loss baseline ln({NUM_CLASSES}) = {:.4}",
		(NUM_CLASSES as f64).ln()
	);
	println!("  loss: evaluation {evaluation_loss:.6} · last training {final_training_loss:.6}");
	println!("  image accuracy: {:.1}%", evaluation_accuracy * 100.0);
	println!("  steps: {steps} · batch: {batch} · lr: {lr}");

	// Quantized-inference check: pack the trained patch projection to Q8 and
	// compare its forward against the dense module on the same eval batch.
	let quantized_projection = oa::ml::nn::QuantizedLinear::from_linear(
		&model.patch_projection,
		oa::ml::matrix::Quantization::Q8,
	)?;
	let dense_projection = model.patch_projection.forward(&patches)?.read_f32()?;
	let packed_projection = quantized_projection.forward(&patches)?.read_f32()?;
	let max_deviation = dense_projection
		.iter()
		.zip(&packed_projection)
		.map(|(dense, packed)| (dense - packed).abs())
		.fold(0.0_f32, f32::max);
	println!("  quantized patch projection (Q8): max |Δ| {max_deviation:.6}");
	assert!(
		max_deviation <= 0.02,
		"quantized projection deviates {max_deviation:.6} from the dense forward",
	);
	if let Some(diagnostics) = program_diagnostics {
		println!(
			"  program: {} nodes · {} recording · {} cache hits · {} uploads",
			diagnostics.node_count(),
			diagnostics.command_recording_count(),
			diagnostics.command_cache_hit_count(),
			diagnostics.input_upload_count(),
		);
	}
	if initial_loss > 0.0 {
		println!(
			"  loss improvement: {:.2}x ({initial_loss:.4} → {final_training_loss:.4})",
			initial_loss / final_training_loss.max(1.0e-9)
		);
	}

	let loss_baseline = (NUM_CLASSES as f32).ln() * 0.3;
	assert!(
		evaluation_loss < loss_baseline && evaluation_accuracy >= 0.95,
		"vit tutorial failed its convergence gate (loss {evaluation_loss:.6}, accuracy {:.3})",
		evaluation_accuracy,
	);
	metrics.flush()?;
	Ok(())
}
