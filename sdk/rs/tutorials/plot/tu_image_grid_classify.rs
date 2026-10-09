// OA Tutorial — Image-grid Classification (Fashion-MNIST 5×5 grid)
//
// level-1 API: oa::ml::Module + oa::plot::Figure / Axes  (imshow / title / caption)
//
// Parallel to the TF Keras classification tutorial — same 5×5 prediction grid,
// OA Rust syntax, GPU all the way through the compositor.
//
//   Same renderer body, two sinks:
//     tu_image_grid_classify                     → show (interactive window)
//     tu_image_grid_classify --save grid.png      → save_to (batch headless)
//
// usage:
//   tu_image_grid_classify [data_dir] [--save path.png] [--steps N]
//
// Default data dir: ./data/fashionMnist
//
// Donor reference: tuImageGridClassify.cpp

use std::path::{Path, PathBuf};
use std::rc::Rc;

use oa::ml::{
	GradientTape, ItTraining, ItTrainingConfig, LossMetric, Module, ModuleRegistry, ProgressBar,
	TrainingSummary,
};
use oa::plot::{Axes, Figure, FigureConfig};
use oa::ui::Ui;
use oa::{Color, Engine, Matrix, Texture, ViewerLiveCapabilities, ViewerLiveSource};

// ── Fashion-MNIST class names ─────────────────────────────────────────────────

const CLASSES: [&str; 10] = [
	"T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot",
];
const NUM_CLASSES: usize = 10;

// ── Title colors (donor: kSuccess / kError / kMuted) ─────────────────────────

fn color_success() -> Color {
	Color::new(0.188, 0.820, 0.345, 1.0) // #30d158
}
fn color_error() -> Color {
	Color::new(1.000, 0.271, 0.227, 1.0) // #ff453a
}
fn color_muted() -> Color {
	Color::new(0.565, 0.565, 0.565, 1.0) // #909090
}

// ── Fashion-MNIST IDX file loader ─────────────────────────────────────────────

/// Raw decoded IDX dataset.
struct MnistData {
	/// Raw U8 pixel bytes, `count × 784`.
	images: Vec<u8>,
	/// Raw U8 label bytes, `count × 1`.
	labels: Vec<u8>,
	count: usize,
}

fn read_be32(bytes: &[u8], offset: &mut usize) -> Option<u32> {
	if bytes.len().checked_sub(*offset)? < 4 {
		return None;
	}
	let b = &bytes[*offset..*offset + 4];
	*offset += 4;
	Some(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn load_mnist_idx(dir: &Path, img_file: &str, lbl_file: &str) -> Option<MnistData> {
	let img_bytes = std::fs::read(dir.join(img_file)).ok()?;
	let lbl_bytes = std::fs::read(dir.join(lbl_file)).ok()?;

	let mut io = 0usize;
	let mut lo = 0usize;

	let img_magic = read_be32(&img_bytes, &mut io)?;
	if img_magic != 0x0000_0803 {
		return None;
	}
	let n = read_be32(&img_bytes, &mut io)? as usize;
	let rows = read_be32(&img_bytes, &mut io)? as usize;
	let cols = read_be32(&img_bytes, &mut io)? as usize;
	if rows != 28 || cols != 28 {
		return None;
	}

	let lbl_magic = read_be32(&lbl_bytes, &mut lo)?;
	if lbl_magic != 0x0000_0801 {
		return None;
	}
	let lbl_n = read_be32(&lbl_bytes, &mut lo)? as usize;
	if lbl_n != n {
		return None;
	}

	let img_expected = n * 784;
	if io + img_expected != img_bytes.len() || lo + n != lbl_bytes.len() {
		return None;
	}

	Some(MnistData {
		images: img_bytes[io..].to_vec(),
		labels: lbl_bytes[lo..].to_vec(),
		count: n,
	})
}

// ── MLP classifier: 784 → ReLU(128) → 10 ────────────────────────────────────

/// Two-layer MLP. Donor: `MnistClassifier` in tuImageGridClassify.cpp.
struct MnistClassifier {
	fc1: Rc<oa::ml::nn::Linear>,
	fc2: Rc<oa::ml::nn::Linear>,
	relu: oa::ml::nn::Relu,
	registry: ModuleRegistry,
}

impl MnistClassifier {
	fn new(engine: &Engine) -> oa::Result<Self> {
		let fc1 = Rc::new(oa::ml::nn::Linear::with_seed(engine, 784, 128, 1)?);
		let fc2 = Rc::new(oa::ml::nn::Linear::with_seed(engine, 128, NUM_CLASSES, 2)?);
		let relu = oa::ml::nn::Relu::new();
		let mut registry = ModuleRegistry::new();
		registry.register_module("fc1", fc1.clone() as Rc<dyn Module>)?;
		registry.register_module("fc2", fc2.clone() as Rc<dyn Module>)?;
		Ok(Self {
			fc1,
			fc2,
			relu,
			registry,
		})
	}
}

impl Module for MnistClassifier {
	fn forward(&self, input: &Matrix) -> oa::Result<Matrix> {
		let h = self.fc1.forward(input)?;
		let h = self.relu.forward(&h)?;
		self.fc2.forward(&h)
	}

	fn registry(&self) -> &ModuleRegistry {
		&self.registry
	}
}

// ── Grid cell: one tile + prediction info ────────────────────────────────────

struct GridCell {
	/// 28×28 RGBA8 texture.
	tile: Texture,
	actual: usize,
	predicted: usize,
	correct: bool,
}

// ── Training + inference pipeline ────────────────────────────────────────────

const GRID_ROWS: usize = 5;
const GRID_COLS: usize = 5;
const GRID_N: usize = GRID_ROWS * GRID_COLS; // 25

/// Train an MLP on Fashion-MNIST, then infer on the first 25 test images.
fn train_and_predict_grid(
	engine: &Engine,
	data_dir: &Path,
	train_steps: u64,
) -> anyhow::Result<Vec<GridCell>> {
	// Load raw test data (pixel bytes needed for textures).
	let test_raw = load_mnist_idx(data_dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
		.ok_or_else(|| {
			anyhow::anyhow!(
				"Fashion-MNIST test set not found at {} — \
			 download from https://github.com/zalandoresearch/fashion-mnist",
				data_dir.display()
			)
		})?;

	// Load training data.
	let train_raw = load_mnist_idx(
		data_dir,
		"train-images-idx3-ubyte",
		"train-labels-idx1-ubyte",
	)
	.ok_or_else(|| {
		anyhow::anyhow!(
			"Fashion-MNIST training set not found at {}",
			data_dir.display()
		)
	})?;

	println!(
		"Loaded {} train / {} test images",
		train_raw.count, test_raw.count
	);

	// Build model + optimizer.
	let model = MnistClassifier::new(engine)?;
	let params = model.all_parameters()?;
	let mut optimizer = oa::ml::AdamW::new(params, 0.001)?;

	const BATCH: usize = 64;
	let steps_per_epoch = (train_raw.count / BATCH) as u64;

	let mut loss_metric = LossMetric::default();
	let mut progress = ProgressBar::default();
	let mut summary = TrainingSummary::default();
	let mut training = ItTraining::new(
		engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: train_steps,
			steps_per_epoch,
			batch_size: BATCH as u64,
			timer_name: "image_grid_classify_step".into(),
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary);

	// Advance through the training set sequentially (deterministic, no shuffle).
	let n_train = train_raw.count;
	let mut step_offset = 0usize;

	while training.step(
		|| {
			let start = (step_offset * BATCH) % n_train.saturating_sub(1).max(BATCH);
			step_offset = step_offset.wrapping_add(1);

			let mut x_f32 = vec![0.0f32; BATCH * 784];
			let mut y_u32 = vec![0u32; BATCH];
			for bi in 0..BATCH {
				let si = (start + bi).min(n_train - 1);
				let src = &train_raw.images[si * 784..(si + 1) * 784];
				for (j, &pixel) in src.iter().enumerate() {
					x_f32[bi * 784 + j] = pixel as f32 / 255.0;
				}
				y_u32[bi] = train_raw.labels[si] as u32;
			}
			let x = Matrix::from_slice(engine, [BATCH, 784], &x_f32)?;
			let y = Matrix::from_slice(engine, [BATCH], &y_u32)?;
			Ok((x, y))
		},
		|(x, y)| {
			let tape = GradientTape::new();
			let logits = model.forward(&x)?;
			let loss = oa::ml::loss::cross_entropy(&logits, &y)?;
			tape.backward(&loss)?;
			Ok(loss)
		},
	)? {}

	training.finish()?;
	println!("Training done.");

	// ── Inference on the first GRID_N test images ─────────────────────────

	let mut x_f32 = vec![0.0f32; GRID_N * 784];
	for (i, pixel) in test_raw.images[..GRID_N * 784].iter().enumerate() {
		x_f32[i] = *pixel as f32 / 255.0;
	}
	let x_test = Matrix::from_slice(engine, [GRID_N, 784], &x_f32)?;
	let logits = model.forward(&x_test)?;
	let probs = oa::matrix::softmax(&logits, -1)?;
	let host_probs = probs.read_f32()?;

	// Build grid cells.
	let mut cells = Vec::with_capacity(GRID_N);
	for i in 0..GRID_N {
		// Argmax for predicted class.
		let row = &host_probs[i * NUM_CLASSES..(i + 1) * NUM_CLASSES];
		let predicted = row
			.iter()
			.enumerate()
			.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
			.map(|(idx, _)| idx)
			.unwrap_or(0);
		let actual = test_raw.labels[i] as usize;

		// Build an RGBA8 grayscale texture from raw pixel bytes.
		let src = &test_raw.images[i * 784..(i + 1) * 784];
		let mut rgba = vec![0u8; 28 * 28 * 4];
		for (p, &g) in src.iter().enumerate() {
			rgba[p * 4] = g;
			rgba[p * 4 + 1] = g;
			rgba[p * 4 + 2] = g;
			rgba[p * 4 + 3] = 255;
		}
		let tile = oa::render::texture_from_rgba8(engine, &rgba, 28, 28)?;

		cells.push(GridCell {
			tile,
			actual,
			predicted,
			correct: predicted == actual,
		});
	}

	let correct_count = cells.iter().filter(|c| c.correct).count();
	println!(
		"Prediction grid: {} / {} correct on the first {} test images",
		correct_count, GRID_N, GRID_N
	);

	Ok(cells)
}

// ── Figure population ─────────────────────────────────────────────────────────

fn populate_figure(fig: &mut Figure, cells: &[GridCell]) {
	for (i, cell) in cells.iter().enumerate().take(GRID_N) {
		let row = (i / GRID_COLS) as i32;
		let col = (i % GRID_COLS) as i32;
		let ax: &mut Axes = fig.ax(row, col);
		ax.imshow(cell.tile.clone());
		if cell.correct {
			ax.title_color(CLASSES[cell.predicted], color_success());
		} else {
			ax.title_color(CLASSES[cell.predicted], color_error());
			ax.caption_color(CLASSES[cell.actual], color_muted());
		}
		// No grid lines or legend for image cells.
		ax.grid(false);
		ax.legend(false);
	}
}

// ── ViewerLiveSource: drives the figure from inside a Viewer lifecycle ────────

/// Donor: `ImageGridClassifySource` in tuImageGridClassify.cpp.
pub struct ImageGridClassifySource {
	data_dir: PathBuf,
	train_steps: u64,
	fig: Figure,
	cells: Vec<GridCell>,
}

impl ImageGridClassifySource {
	pub fn new(data_dir: PathBuf, train_steps: u64) -> Self {
		let fig = Figure::new(FigureConfig {
			rows: GRID_ROWS as i32,
			cols: GRID_COLS as i32,
			width: 800,
			height: 800,
			h_spacing: 8,
			v_spacing: 8,
			padding: 8,
			..Default::default()
		});
		Self {
			data_dir,
			train_steps,
			fig,
			cells: Vec::new(),
		}
	}
}

impl ViewerLiveSource for ImageGridClassifySource {
	fn capabilities(&self) -> ViewerLiveCapabilities {
		ViewerLiveCapabilities::default()
	}

	fn open(&mut self, engine: &Engine) -> oa::Result<()> {
		self.cells = train_and_predict_grid(engine, &self.data_dir, self.train_steps)
			.map_err(|e| oa::Error::callback(e.to_string()))?;
		populate_figure(&mut self.fig, &self.cells);
		Ok(())
	}

	fn update(&mut self, _delta_ms: f32) -> oa::Result<()> {
		Ok(())
	}

	fn render(&mut self, ui: &mut Ui<'_>, width: u32, height: u32) -> oa::Result<()> {
		self.fig.render_frame(width, height, ui)
	}

	fn close(&mut self) -> oa::Result<()> {
		self.cells.clear();
		Ok(())
	}
}

// ── CLI + main ────────────────────────────────────────────────────────────────

fn usage(program: &str) {
	eprintln!(
		"usage: {program} [data_dir] [--save path.png] [--steps N]\n\
		 \n\
		 data_dir  path to the Fashion-MNIST IDX files (default: ./data/fashionMnist)\n\
		 --save    write the 5×5 grid as a PNG instead of opening a window\n\
		 --steps   number of training steps (default: 2000)"
	);
}

fn main() -> anyhow::Result<()> {
	let mut data_dir = PathBuf::from("data/fashionMnist");
	let mut save_path: Option<String> = None;
	let mut train_steps: u64 = 2000;
	let mut positional_done = false;

	let args: Vec<String> = std::env::args().collect();
	let mut i = 1;
	while i < args.len() {
		match args[i].as_str() {
			"--save" if i + 1 < args.len() => {
				i += 1;
				save_path = Some(args[i].clone());
			}
			"--steps" if i + 1 < args.len() => {
				i += 1;
				train_steps = args[i].parse().unwrap_or(2000);
			}
			"--help" => {
				usage(&args[0]);
				return Ok(());
			}
			other if !positional_done && !other.starts_with('-') => {
				data_dir = PathBuf::from(other);
				positional_done = true;
			}
			other => {
				eprintln!("unknown argument: {other}");
				usage(&args[0]);
				std::process::exit(1);
			}
		}
		i += 1;
	}

	println!();
	println!("╔══════════════════════════════════════════════════════════════════╗");
	println!("║  OA Tutorial — Fashion-MNIST Prediction grid (oa::plot)          ║");
	println!("║  Train → Predict → display  (5×5 grid, 25 test images)           ║");
	println!("║  Green title = correct  ·  Red title = wrong                     ║");
	println!(
		"║  Mode: {:<58}║",
		if save_path.is_some() {
			"save_to (headless PNG output)"
		} else {
			"show (interactive window)"
		}
	);
	println!("╚══════════════════════════════════════════════════════════════════╝");
	println!();

	// ── save mode: compute-only engine, no swapchain ─────────────────────────
	if let Some(ref path) = save_path {
		let engine = Engine::new()?;
		let cells = train_and_predict_grid(&engine, &data_dir, train_steps)?;
		let mut fig = Figure::new(FigureConfig {
			rows: GRID_ROWS as i32,
			cols: GRID_COLS as i32,
			width: 800,
			height: 800,
			h_spacing: 8,
			v_spacing: 8,
			padding: 4,
			..Default::default()
		});
		populate_figure(&mut fig, &cells);
		fig.save_to(&engine, path)?;
		println!("Saved: {path}");
		return Ok(());
	}

	// ── show mode: Viewer lifecycle with a live figure source ─────────────────
	let mut source = ImageGridClassifySource::new(data_dir, train_steps);
	let config = oa::ViewerConfig {
		mode: oa::ViewerMode::Live,
		width: 800,
		height: 800,
		show_help: false,
		show_stats: false,
		show_timeline: false,
		..Default::default()
	};
	let mut viewer = oa::Viewer::new(config);
	viewer.set_live_source(&mut source);

	let window = oa::SdlWindow::new("OA — Fashion-MNIST grid", 800, 800)
		.map_err(|e| anyhow::anyhow!("SDL window: {e}"))?;
	let extensions = window
		.required_instance_extensions()
		.map_err(|e| anyhow::anyhow!("SDL extensions: {e}"))?;
	let engine = Engine::builder().instance_extensions(extensions).build()?;
	viewer.run(&engine)?;
	Ok(())
}
