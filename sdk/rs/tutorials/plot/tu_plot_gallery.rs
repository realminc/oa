// OA Plot walkthrough: nine deterministic diagnostic plots, two themes, one
// GPU compositor, and the same retained Figure replay for PNG and oa::Viewer.
//
// usage:
//   tu_plot_gallery [--output DIR] [--headless]
//
// The two wireframe panels are deliberate 2D projections of scalar fields.
// They demonstrate explicit-X multi-series composition without claiming a
// general 3D scene, camera, depth, or surface-plot API.
//
// Donor reference: tuPlotGallery.cpp

use std::f32::consts::PI;

use oa::plot::{Axes, BarStyle, FigureConfig, HeatmapStyle, LineStyle, ScatterStyle, Theme};
use oa::{Color, Engine};

// ── Style helpers ─────────────────────────────────────────────────────────────

fn line(color: Color, label: &str, width: f32, antialias_samples: u32) -> LineStyle {
	LineStyle {
		color,
		label: label.to_owned(),
		width,
		antialias_samples,
	}
}

fn points(color: Color, label: &str, radius: f32) -> ScatterStyle {
	ScatterStyle {
		color,
		label: label.to_owned(),
		radius,
	}
}

fn bars(color: Color, label: &str, gap: f32) -> BarStyle {
	BarStyle {
		color,
		label: label.to_owned(),
		gap,
	}
}

fn gradient(a: Color, b: Color, t: f32) -> Color {
	let t = t.clamp(0.0, 1.0);
	Color::new(
		a.r + (b.r - a.r) * t,
		a.g + (b.g - a.g) * t,
		a.b + (b.b - a.b) * t,
		a.a + (b.a - a.a) * t,
	)
}

// ── Named colors matching the donor palette ───────────────────────────────────

fn accent() -> Color {
	Color::new(0.24, 0.65, 1.00, 1.0)
}
fn success() -> Color {
	Color::new(0.35, 0.82, 0.54, 1.0)
}
fn purple() -> Color {
	Color::new(0.75, 0.45, 1.00, 1.0)
}
fn cyan() -> Color {
	Color::new(0.30, 0.87, 0.89, 1.0)
}
fn warning() -> Color {
	Color::new(1.00, 0.80, 0.22, 1.0)
}
fn pink() -> Color {
	Color::new(1.00, 0.55, 0.75, 1.0)
}
fn gray() -> Color {
	Color::new(0.565, 0.565, 0.565, 1.0)
}

// ── Individual axes builders ──────────────────────────────────────────────────

// [oa-plot-intro-begin]
fn intro_figure() -> oa::plot::Figure {
	let mut fig = oa::plot::Figure::new(FigureConfig {
		title: "oa::plot intro".to_owned(),
		rows: 1,
		cols: 2,
		width: 1280,
		height: 560,
		h_spacing: 36,
		padding: 34,
		theme: Theme::Dark,
		..Default::default()
	});
	fig.title("One retained figure — Rust and C++ parity");

	let steps = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
	let train = [1.00f32, 0.78, 0.61, 0.48, 0.37, 0.29, 0.23, 0.19];
	let validation = [1.04f32, 0.83, 0.66, 0.53, 0.43, 0.35, 0.30, 0.27];
	let ideal = [0.0f32, 0.25, 0.50, 0.75, 1.0];
	let confidence = [0.10f32, 0.30, 0.50, 0.70, 0.90];
	let accuracy = [0.08f32, 0.34, 0.47, 0.74, 0.88];

	let curves = fig.ax(0, 0);
	curves.title("training curves");
	curves.x_label("optimizer step");
	curves.y_label("cross entropy");
	curves.limits(0.0, 7.0, 0.0, 1.1);
	curves.plot_xy(&steps, &train, line(accent(), "train", 1.6, 4));
	curves.plot_xy(&steps, &validation, line(success(), "validation", 1.6, 4));

	let quality = fig.ax(0, 1);
	quality.title("Calibration");
	quality.x_label("confidence");
	quality.y_label("observed accuracy");
	quality.limits(0.0, 1.0, 0.0, 1.0);
	quality.plot_xy(&ideal, &ideal, line(gray(), "ideal", 1.35, 4));
	quality.scatter(&confidence, &accuracy, points(cyan(), "model", 3.5));
	fig
}
// [oa-plot-intro-end]

fn training_plot(ax: &mut Axes) {
	let n = 72usize;
	let train: Vec<f32> = (0..n)
		.map(|i| {
			let step = i as f32;
			1.55 * (-step / 18.0).exp() + 0.035 * (step * 0.52).sin() + 0.11
		})
		.collect();
	let validation: Vec<f32> = (0..n)
		.map(|i| {
			let step = i as f32;
			1.45 * (-step / 21.0).exp() + 0.045 * (step * 0.39 + 0.8).sin() + 0.17
		})
		.collect();
	ax.title("training curves");
	ax.x_label("optimizer step");
	ax.y_label("cross entropy");
	ax.plot(&train, line(accent(), "train", 1.35, 4));
	ax.plot(&validation, line(success(), "validation", 1.35, 4));
}

fn roc_plot(ax: &mut Axes) {
	let n = 33usize;
	let rate: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
	let vision: Vec<f32> = rate
		.iter()
		.map(|&x| (1.0 - (-4.3 * x).exp()).clamp(0.0, 1.0))
		.collect();
	let fusion: Vec<f32> = rate
		.iter()
		.map(|&x| (1.0 - (-6.7 * x).exp()).clamp(0.0, 1.0))
		.collect();
	ax.title("ROC — threshold sweep");
	ax.x_label("false positive rate");
	ax.y_label("true positive rate");
	ax.limits(0.0, 1.0, 0.0, 1.0);
	ax.plot_xy(&rate, &vision, line(cyan(), "vision AUC 0.88", 1.35, 4));
	ax.plot_xy(&rate, &fusion, line(success(), "fusion AUC 0.93", 1.35, 4));
	ax.plot_xy(&rate, &rate, line(gray(), "chance", 1.35, 4));
}

fn precision_recall_plot(ax: &mut Axes) {
	let n = 33usize;
	let recall: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
	let vision: Vec<f32> = recall
		.iter()
		.map(|&x| (0.98 - 0.40 * x.powf(1.6) - 0.025 * (x * 5.0 * PI).sin()).clamp(0.0, 1.0))
		.collect();
	let fusion: Vec<f32> = recall
		.iter()
		.map(|&x| (0.995 - 0.27 * x.powf(2.1) - 0.015 * (x * 4.0 * PI).sin()).clamp(0.0, 1.0))
		.collect();
	ax.title("Precision-recall");
	ax.x_label("recall");
	ax.y_label("precision");
	ax.limits(0.0, 1.0, 0.0, 1.0);
	ax.plot_xy(&recall, &vision, line(purple(), "vision AP 0.84", 1.35, 4));
	ax.plot_xy(&recall, &fusion, line(pink(), "fusion AP 0.91", 1.35, 4));
}

fn score_histogram(ax: &mut Axes) {
	let scores: Vec<f32> = (0..192usize)
		.map(|i| {
			let phase = i as f32;
			let cluster = if i % 3 == 0 { 0.28 } else { 0.74 };
			(cluster + 0.12 * (phase * 1.73).sin() + 0.045 * (phase * 0.37).cos()).clamp(0.0, 1.0)
		})
		.collect();
	ax.title("confidence distribution");
	ax.x_label("model score");
	ax.y_label("samples");
	ax.histogram(&scores, 20, bars(accent(), "validation scores", 0.20));
}

fn calibration_scatter(ax: &mut Axes) {
	let n = 24usize;
	let confidence: Vec<f32> = (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect();
	let accuracy: Vec<f32> = confidence
		.iter()
		.map(|&x| (x + 0.055 * (8.0 * PI * x).sin() - 0.025).clamp(0.0, 1.0))
		.collect();
	let ideal: Vec<f32> = confidence.clone();
	ax.title("Calibration");
	ax.x_label("confidence");
	ax.y_label("observed accuracy");
	ax.limits(0.0, 1.0, 0.0, 1.0);
	ax.plot_xy(&confidence, &ideal, line(gray(), "ideal", 1.35, 4));
	ax.scatter(&confidence, &accuracy, points(cyan(), "model", 3.5));
}

fn throughput_bars(ax: &mut Axes) {
	let throughput = [0.42f32, 0.58, 0.71, 0.86, 0.79, 0.94];
	ax.title("normalized throughput");
	ax.x_label("kernel route");
	ax.y_label("relative peak");
	ax.limits(-0.5, 5.5, 0.0, 1.0);
	ax.bar(&throughput, bars(warning(), "device routes", 0.24));
}

fn confusion_heatmap(ax: &mut Axes) {
	let confusion = [
		42.0f32, 2.0, 0.0, 1.0, 0.0, 3.0, 37.0, 2.0, 0.0, 1.0, 0.0, 2.0, 40.0, 3.0, 0.0, 1.0, 0.0, 2.0,
		38.0, 2.0, 0.0, 1.0, 0.0, 2.0, 43.0,
	];
	ax.title("confusion matrix");
	ax.x_label("predicted class");
	ax.y_label("reference class");
	ax.heatmap(
		&confusion,
		5,
		5,
		HeatmapStyle {
			colormap: 1,
			auto_scale: true,
			show_grid: true,
			..Default::default()
		},
	);
	ax.grid(false);
}

fn wireframe(
	ax: &mut Axes,
	title: &str,
	height_fn: impl Fn(f32, f32) -> f32,
	near: Color,
	far: Color,
) {
	const LINES: i32 = 13;
	const SAMPLES: i32 = 29;
	// Row sweeps.
	for row in 0..LINES {
		let v = -1.0 + 2.0 * row as f32 / (LINES - 1) as f32;
		let mut px = vec![0.0f32; SAMPLES as usize];
		let mut py = vec![0.0f32; SAMPLES as usize];
		for col in 0..SAMPLES {
			let u = -1.0 + 2.0 * col as f32 / (SAMPLES - 1) as f32;
			let z = height_fn(u, v);
			px[col as usize] = 0.50 + 0.34 * (u - v);
			py[col as usize] = 0.58 - 0.16 * (u + v) - 0.23 * z;
		}
		let t = row as f32 / (LINES - 1) as f32;
		ax.plot_xy(&px, &py, line(gradient(far, near, t), "", 1.15, 8));
	}
	// Column sweeps.
	for col in 0..LINES {
		let u = -1.0 + 2.0 * col as f32 / (LINES - 1) as f32;
		let mut px = vec![0.0f32; SAMPLES as usize];
		let mut py = vec![0.0f32; SAMPLES as usize];
		for row in 0..SAMPLES {
			let v = -1.0 + 2.0 * row as f32 / (SAMPLES - 1) as f32;
			let z = height_fn(u, v);
			px[row as usize] = 0.50 + 0.34 * (u - v);
			py[row as usize] = 0.58 - 0.16 * (u + v) - 0.23 * z;
		}
		let t = col as f32 / (LINES - 1) as f32;
		let mut col_style = line(gradient(far, near, t), "", 1.15, 8);
		col_style.color.a = 0.78;
		ax.plot_xy(&px, &py, col_style);
	}
	ax.title(title);
	ax.limits(-0.20, 1.20, 0.02, 1.08);
	ax.grid(false);
	ax.legend(false);
}

fn loss_landscape(ax: &mut Axes) {
	wireframe(
		ax,
		"projected loss landscape",
		|x, y| {
			let bowl = 0.42 * (x * x + 0.75 * y * y);
			bowl + 0.18 * (2.7 * x).sin() * (3.1 * y).cos()
		},
		cyan(),
		purple(),
	);
}

fn gradient_basin(ax: &mut Axes) {
	wireframe(
		ax,
		"projected optimizer basin",
		|x, y| {
			let radius = (x * x + y * y).sqrt();
			0.34 * radius + 0.17 * (9.0 * radius).cos() * (-1.8 * radius).exp()
		},
		success(),
		warning(),
	);
}

fn gallery(theme: Theme) -> oa::plot::Figure {
	let mut fig = oa::plot::Figure::new(FigureConfig {
		title: "OA Plot gallery".to_owned(),
		rows: 3,
		cols: 3,
		width: 1600,
		height: 1050,
		h_spacing: 32,
		v_spacing: 36,
		padding: 34,
		theme,
		..Default::default()
	});
	fig.title("OA Plot — GPU-composed ML diagnostics");
	training_plot(fig.ax(0, 0));
	roc_plot(fig.ax(0, 1));
	precision_recall_plot(fig.ax(0, 2));
	score_histogram(fig.ax(1, 0));
	calibration_scatter(fig.ax(1, 1));
	throughput_bars(fig.ax(1, 2));
	confusion_heatmap(fig.ax(2, 0));
	loss_landscape(fig.ax(2, 1));
	gradient_basin(fig.ax(2, 2));
	fig
}

fn evaluation_figure() -> oa::plot::Figure {
	let mut fig = oa::plot::Figure::new(FigureConfig {
		title: "OA model evaluation".to_owned(),
		rows: 1,
		cols: 2,
		width: 1280,
		height: 560,
		h_spacing: 36,
		padding: 34,
		..Default::default()
	});
	fig.title("Model evaluation — ROC and precision-recall");
	roc_plot(fig.ax(0, 0));
	precision_recall_plot(fig.ax(0, 1));
	fig
}

fn diagnostics_figure() -> oa::plot::Figure {
	let mut fig = oa::plot::Figure::new(FigureConfig {
		title: "OA model diagnostics".to_owned(),
		rows: 2,
		cols: 2,
		width: 1200,
		height: 900,
		h_spacing: 32,
		v_spacing: 36,
		padding: 34,
		..Default::default()
	});
	fig.title("training and evaluation diagnostics");
	training_plot(fig.ax(0, 0));
	score_histogram(fig.ax(0, 1));
	calibration_scatter(fig.ax(1, 0));
	confusion_heatmap(fig.ax(1, 1));
	fig
}

fn landscapes_figure() -> oa::plot::Figure {
	let mut fig = oa::plot::Figure::new(FigureConfig {
		title: "OA projected landscapes".to_owned(),
		rows: 1,
		cols: 2,
		width: 1280,
		height: 560,
		h_spacing: 36,
		padding: 34,
		..Default::default()
	});
	fig.title("scalar-field projections — explicit-X wireframes");
	loss_landscape(fig.ax(0, 0));
	gradient_basin(fig.ax(0, 1));
	fig
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn usage(program: &str) {
	eprintln!("usage: {program} [--output DIR] [--headless]");
}

fn main() -> anyhow::Result<()> {
	let mut output = std::path::PathBuf::from("artifact/plot");
	let mut show = true;

	let args: Vec<String> = std::env::args().collect();
	let mut i = 1;
	while i < args.len() {
		match args[i].as_str() {
			"--headless" => {
				show = false;
			}
			"--output" if i + 1 < args.len() => {
				i += 1;
				output = std::path::PathBuf::from(&args[i]);
			}
			"--help" => {
				usage(&args[0]);
				return Ok(());
			}
			other => {
				eprintln!("unknown argument: {other}");
				usage(&args[0]);
				std::process::exit(1);
			}
		}
		i += 1;
	}

	std::fs::create_dir_all(&output).map_err(|e| {
		eprintln!("could not create output dir: {e}");
		e
	})?;

	let engine = if show {
		let window = oa::SdlWindow::new("OA Plot presentation probe", 1, 1)?;
		let extensions = window.required_instance_extensions()?;
		Engine::builder().instance_extensions(extensions).build()?
	} else {
		Engine::new()?
	};

	let mut dark = gallery(Theme::Dark);
	let mut light = gallery(Theme::Light);
	let mut intro = intro_figure();
	let mut evaluation = evaluation_figure();
	let mut diagnostics = diagnostics_figure();
	let mut landscapes = landscapes_figure();

	println!("OA Plot walkthrough artifacts:");

	let save = |fig: &mut oa::plot::Figure, name: &str| -> anyhow::Result<()> {
		let path = output.join(name);
		fig.save_to(&engine, path.to_str().unwrap_or(name))?;
		println!("  {}", path.display());
		Ok(())
	};

	save(&mut intro, "oa-plot-intro-dark.png")?;
	save(&mut dark, "oa-plot-gallery-dark.png")?;
	save(&mut light, "oa-plot-gallery-light.png")?;
	save(&mut evaluation, "oa-plot-evaluation-dark.png")?;
	save(&mut diagnostics, "oa-plot-diagnostics-dark.png")?;
	save(&mut landscapes, "oa-plot-landscapes-dark.png")?;

	if show {
		dark.show(&engine)?;
	}
	Ok(())
}
