use oa::ml::{ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary};
use oa::{Cli, LogMetrics};

// Synthetic 3-class image task shared with tu_vision_vit: horizontal bar,
// vertical bar, or diagonal stripe on a 16x16 grayscale grid with light
// noise. The residual network keeps the full [B, C, H, W] contract.
const IMAGE_SIDE: usize = 16;
const CHANNELS: usize = 8;
const NUM_CLASSES: usize = 3;
const TRAINING_STEPS: usize = 200;
const BATCH_SIZE: usize = 32;
const EVAL_IMAGES: usize = 64;

#[derive(Default)]
struct ResNetConfig {
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

fn next_batch(rng: &mut ShapeRng, batch: usize) -> (Vec<f32>, Vec<u32>) {
	let mut images = vec![0.0_f32; batch * IMAGE_SIDE * IMAGE_SIDE];
	let mut labels = Vec::with_capacity(batch);
	for image in 0..batch {
		let class = rng.below(NUM_CLASSES) as u32;
		labels.push(class);
		let base = image * IMAGE_SIDE * IMAGE_SIDE;
		for pixel in 0..IMAGE_SIDE * IMAGE_SIDE {
			images[base + pixel] = 0.05 * rng.unit();
		}
		match class {
			0 => {
				let row = 2 + rng.below(IMAGE_SIDE - 4);
				for column in 0..IMAGE_SIDE {
					images[base + row * IMAGE_SIDE + column] = 0.85 + 0.1 * rng.unit();
				}
			}
			1 => {
				let column = 2 + rng.below(IMAGE_SIDE - 4);
				for row in 0..IMAGE_SIDE {
					images[base + row * IMAGE_SIDE + column] = 0.85 + 0.1 * rng.unit();
				}
			}
			_ => {
				let offset = 1 + rng.below(2);
				for diagonal in 0..IMAGE_SIDE {
					let column = diagonal + offset;
					if column < IMAGE_SIDE {
						images[base + diagonal * IMAGE_SIDE + column] = 0.85 + 0.1 * rng.unit();
					}
				}
			}
		}
	}
	(images, labels)
}

// Standard residual stage: conv-bn-relu-conv-bn plus an identity skip,
// then a final relu. Channel counts stay constant so no projection skip
// is required.
struct ResidualBlock {
	conv_a: oa::ml::nn::Conv2d,
	norm_a: oa::ml::nn::BatchNorm2d,
	relu_a: oa::ml::nn::Relu,
	conv_b: oa::ml::nn::Conv2d,
	norm_b: oa::ml::nn::BatchNorm2d,
	relu_out: oa::ml::nn::Relu,
}

impl ResidualBlock {
	fn new(engine: &oa::Engine, seed: u64) -> oa::Result<Self> {
		Ok(Self {
			conv_a: oa::ml::nn::Conv2d::new_with_seed_on(engine, CHANNELS, CHANNELS, 3, 1, 1, 1, seed)?,
			norm_a: oa::ml::nn::BatchNorm2d::new_on(engine, CHANNELS)?,
			relu_a: oa::ml::nn::Relu::new(),
			conv_b: oa::ml::nn::Conv2d::new_with_seed_on(
				engine,
				CHANNELS,
				CHANNELS,
				3,
				1,
				1,
				1,
				seed.wrapping_add(1),
			)?,
			norm_b: oa::ml::nn::BatchNorm2d::new_on(engine, CHANNELS)?,
			relu_out: oa::ml::nn::Relu::new(),
		})
	}

	fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
		let hidden = self
			.relu_a
			.forward(&self.norm_a.forward(&self.conv_a.forward(input)?)?)?;
		let hidden = self.norm_b.forward(&self.conv_b.forward(&hidden)?)?;
		self.relu_out.forward(&oa::matrix::add(&hidden, input)?)
	}

	fn all_parameters(&self) -> oa::Result<Vec<oa::ml::Parameter>> {
		let mut parameters = self.conv_a.all_parameters()?;
		parameters.extend(self.norm_a.all_parameters()?);
		parameters.extend(self.conv_b.all_parameters()?);
		parameters.extend(self.norm_b.all_parameters()?);
		Ok(parameters)
	}
}

struct ResNet {
	stem_conv: oa::ml::nn::Conv2d,
	stem_norm: oa::ml::nn::BatchNorm2d,
	stem_relu: oa::ml::nn::Relu,
	blocks: Vec<ResidualBlock>,
	pool: oa::ml::nn::AdaptiveAvgPool2d,
	flatten: oa::ml::nn::Flatten,
	head: oa::ml::nn::Linear,
}

impl ResNet {
	fn new(engine: &oa::Engine, seed: u64) -> oa::Result<Self> {
		let stem_conv = oa::ml::nn::Conv2d::new_with_seed_on(engine, 1, CHANNELS, 3, 1, 1, 1, seed)?;
		let stem_norm = oa::ml::nn::BatchNorm2d::new_on(engine, CHANNELS)?;
		let stem_relu = oa::ml::nn::Relu::new();
		let mut blocks = Vec::new();
		for layer in 0..2_usize {
			blocks.push(ResidualBlock::new(
				engine,
				seed.wrapping_add(3 + layer as u64 * 2),
			)?);
		}
		let pool = oa::ml::nn::AdaptiveAvgPool2d::new(1)?;
		let flatten = oa::ml::nn::Flatten::new(1, -1);
		let head =
			oa::ml::nn::Linear::new_with_seed_on(engine, CHANNELS, NUM_CLASSES, seed.wrapping_add(9))?;
		Ok(Self {
			stem_conv,
			stem_norm,
			stem_relu,
			blocks,
			pool,
			flatten,
			head,
		})
	}

	fn all_parameters(&self) -> oa::Result<Vec<oa::ml::Parameter>> {
		let mut parameters = self.stem_conv.all_parameters()?;
		parameters.extend(self.stem_norm.all_parameters()?);
		for block in &self.blocks {
			parameters.extend(block.all_parameters()?);
		}
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

	// `input` is NCHW [B, 1, IMAGE_SIDE, IMAGE_SIDE]; logits are [B, C].
	fn forward(&self, input: &oa::Matrix) -> oa::Result<oa::Matrix> {
		let mut hidden = self
			.stem_relu
			.forward(&self.stem_norm.forward(&self.stem_conv.forward(input)?)?)?;
		for block in &self.blocks {
			hidden = block.forward(&hidden)?;
		}
		let pooled = self.pool.forward(&hidden)?;
		let features = self.flatten.forward(&pooled)?;
		self.head.forward(&features)
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
	let mut cli = Cli::new("tu_vision_resnet", "OA Tutorial — Residual ConvNet");
	cli.add_option(
		"--steps",
		|c: &mut ResNetConfig| &mut c.steps,
		"training steps",
	);
	cli.add_option("--batch", |c: &mut ResNetConfig| &mut c.batch, "batch size");
	cli.add_option("--lr", |c: &mut ResNetConfig| &mut c.lr, "learning rate");
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
	let model = ResNet::new(&engine, 0x7E57)?;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, lr)?;
	let mut rng = ShapeRng::new(0x5EED_7E57);

	println!("\n╔══════════════════════════════════════════════════════════════════╗");
	println!("║  OA Tutorial — Residual ConvNet · conv + BN + skip (autograd)    ║");
	println!("╚══════════════════════════════════════════════════════════════════╝\n");
	println!("task: classify a 16x16 image as horizontal bar, vertical bar, or diagonal");
	println!("model: conv stem + 2 residual blocks + global pool + linear head");
	println!(
		"params: {}    Optimizer: AdamW(lr={lr})\n",
		model.num_parameters()?
	);
	println!("training: {steps} steps · batch={batch} · input=[B,1,{IMAGE_SIDE},{IMAGE_SIDE}]");

	let mut loss_metric = LossMetric::default();
	let mut progress = ProgressBar::default();
	let mut summary_callback = TrainingSummary::default();
	let mut training = ItTraining::new(
		&engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: steps as u64,
			batch_size: batch as u64,
			sequence_length: (IMAGE_SIDE * IMAGE_SIDE) as u64,
			sequence_unit: "pixel".into(),
			timer_name: "resnet_training_step".into(),
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary_callback);

	let mut metrics = LogMetrics::new();
	let log_dir = oa::Path::var_rel("log/vision/resnet");
	if let Err(e) = metrics.open(&log_dir) {
		eprintln!("[OA] metrics unavailable: {e}");
	}

	let mut initial_loss = 0.0_f32;
	for _ in 0..steps {
		assert!(training.step(
			|| {
				let (images, labels) = next_batch(&mut rng, batch);
				Ok((
					oa::Matrix::new_from_f32([batch, 1, IMAGE_SIDE, IMAGE_SIDE], &images)?,
					oa::Matrix::new_from_slice([batch], &labels)?,
				))
			},
			|(images, labels)| {
				let tape = oa::ml::GradientTape::new();
				let logits = model.forward(&images)?;
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

	// Evaluation shares the training-mode batch-statistics BatchNorm path:
	// running statistics freeze at program-record time under captured replay,
	// so the honest eval metric normalizes over the 64-image eval batch.
	let (eval_images, eval_labels) = next_batch(&mut rng, EVAL_IMAGES);
	let images = oa::Matrix::new_from_f32([EVAL_IMAGES, 1, IMAGE_SIDE, IMAGE_SIDE], &eval_images)?;
	let labels = oa::Matrix::new_from_slice([EVAL_IMAGES], &eval_labels)?;
	let logits = model.forward(&images)?;
	let evaluation_loss = oa::ml::loss::cross_entropy(&logits, &labels)?.read_f32()?[0];
	let evaluation_accuracy = accuracy(&logits, &eval_labels, EVAL_IMAGES)?;

	println!("\nEvaluation (batch-statistics BatchNorm):");
	println!(
		"  random-loss baseline ln({NUM_CLASSES}) = {:.4}",
		(NUM_CLASSES as f64).ln()
	);
	println!("  loss: evaluation {evaluation_loss:.6} · last training {final_training_loss:.6}");
	println!("  image accuracy: {:.1}%", evaluation_accuracy * 100.0);
	println!("  steps: {steps} · batch: {batch} · lr: {lr}");
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
		"resnet tutorial failed its convergence gate (loss {evaluation_loss:.6}, accuracy {:.3})",
		evaluation_accuracy,
	);
	metrics.flush()?;
	Ok(())
}
