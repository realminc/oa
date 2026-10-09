use std::{fs, path::Path};

use serde_json::Value;

use crate::{
	Engine, Error, Matrix, Result,
	core::vlm::{Quat, Vec3},
};

/// One HumanML3D caption and its optional partial-clip time range.
#[derive(Clone, Debug, PartialEq)]
pub struct HumanMl3dCaption {
	pub text: String,
	pub range_seconds: Option<(f32, f32)>,
}

/// Geometry and contact diagnostics for denormalized HumanML3D features.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HumanMl3dMotionMetrics {
	pub mpjpe_cm: f64,
	pub velocity_error_cm_per_frame: f64,
	pub contact_accuracy: f64,
	pub foot_skate_cm_per_frame: f64,
}

struct Clip {
	id: String,
	features: Vec<f32>,
	frames: usize,
	captions: Vec<HumanMl3dCaption>,
	text_features: Vec<f32>,
}

/// Checked HumanML3D-layout motion corpus retained in standardized host memory.
pub struct HumanMl3dDataset {
	feature_dim: usize,
	num_joints: usize,
	mean: Vec<f32>,
	standard_deviation: Vec<f32>,
	clips: Vec<Clip>,
	total_frames: usize,
	text_feature_dim: usize,
	text_feature_format: Option<String>,
	text_feature_model: Option<String>,
}

impl HumanMl3dDataset {
	/// Load HumanML3D, CMP, or KIT-ML data from the shared on-disk layout.
	///
	/// `max_clips == 0` loads the complete split. Motion files that cannot satisfy
	/// the rank-two feature contract are skipped exactly like the donor loader.
	///
	/// # Errors
	///
	/// Returns an error for unsupported feature width, missing/invalid Mean, Std,
	/// or split assets, an empty admitted split, arithmetic overflow, or I/O failure.
	pub fn open(
		root: impl AsRef<Path>,
		split: &str,
		max_clips: usize,
		feature_dim: usize,
	) -> Result<Self> {
		let num_joints = joints_for_feature_dim(feature_dim)
			.ok_or_else(|| Error::invalid_argument("HumanML3D feature width must be 263 or 251"))?;
		if split.is_empty() || split.contains('/') || split.contains('\\') {
			return Err(Error::invalid_argument(
				"HumanML3D split must be a file stem",
			));
		}
		let root = root.as_ref();
		let mean = load_npy_f32(&root.join("Mean.npy"))?;
		let standard_deviation = load_npy_f32(&root.join("Std.npy"))?;
		if mean.shape != [feature_dim] || standard_deviation.shape != [feature_dim] {
			return Err(Error::invalid_argument(
				"HumanML3D Mean/Std shape does not match feature width",
			));
		}
		let ids = read_nonempty_lines(&root.join(format!("{split}.txt")))?;
		if ids.is_empty() {
			return Err(Error::data_loss("HumanML3D split is empty"));
		}
		let manifest = load_text_manifest(&root.join("text_feats/manifest.json"))?;
		let mut clips = Vec::new();
		let mut total_frames = 0_usize;
		let mut text_feature_dim = 0_usize;
		for id in ids {
			if max_clips > 0 && clips.len() >= max_clips {
				break;
			}
			let Ok(mut motion) = load_npy_f32(&root.join("new_joint_vecs").join(format!("{id}.npy")))
			else {
				continue;
			};
			if motion.shape.len() != 2 || motion.shape[1] != feature_dim {
				continue;
			}
			let frames = motion.shape[0];
			total_frames = total_frames
				.checked_add(frames)
				.ok_or_else(|| Error::resource_exhausted("HumanML3D frame count overflows usize"))?;
			for row in motion.data.chunks_exact_mut(feature_dim) {
				for ((value, mean), deviation) in
					row.iter_mut().zip(&mean.data).zip(&standard_deviation.data)
				{
					*value = (*value - mean) / deviation;
				}
			}
			let captions = read_captions(&root.join("texts").join(format!("{id}.txt")))?;
			let text_features = if let Some(manifest) = &manifest {
				load_clip_text_features(root, &id, captions.len(), manifest.dimension).unwrap_or_default()
			} else {
				Vec::new()
			};
			if !text_features.is_empty() {
				text_feature_dim = manifest.as_ref().map_or(0, |value| value.dimension);
			}
			clips.push(Clip {
				id,
				features: motion.data,
				frames,
				captions,
				text_features,
			});
		}
		if clips.is_empty() {
			return Err(Error::data_loss("HumanML3D split contains no valid clips"));
		}
		Ok(Self {
			feature_dim,
			num_joints,
			mean: mean.data,
			standard_deviation: standard_deviation.data,
			clips,
			total_frames,
			text_feature_dim,
			text_feature_format: manifest.as_ref().map(|value| value.format.clone()),
			text_feature_model: manifest.map(|value| value.model),
		})
	}

	pub fn open_cmp(root: impl AsRef<Path>, split: &str, max_clips: usize) -> Result<Self> {
		Self::open(root, split, max_clips, 263)
	}

	pub fn open_kit_ml(root: impl AsRef<Path>, split: &str, max_clips: usize) -> Result<Self> {
		Self::open(root, split, max_clips, 251)
	}

	pub fn len(&self) -> usize {
		self.clips.len()
	}
	pub fn is_empty(&self) -> bool {
		self.clips.is_empty()
	}
	pub fn feature_dim(&self) -> usize {
		self.feature_dim
	}
	pub fn num_joints(&self) -> usize {
		self.num_joints
	}
	pub fn total_frames(&self) -> usize {
		self.total_frames
	}
	pub fn mean(&self) -> &[f32] {
		&self.mean
	}
	pub fn standard_deviation(&self) -> &[f32] {
		&self.standard_deviation
	}
	pub fn clip_id(&self, index: usize) -> Option<&str> {
		self.clips.get(index).map(|clip| clip.id.as_str())
	}
	pub fn clip_frames(&self, index: usize) -> Option<usize> {
		self.clips.get(index).map(|clip| clip.frames)
	}
	pub fn clip_data(&self, index: usize) -> Option<&[f32]> {
		self.clips.get(index).map(|clip| clip.features.as_slice())
	}
	pub fn clip_captions(&self, index: usize) -> Option<&[HumanMl3dCaption]> {
		self.clips.get(index).map(|clip| clip.captions.as_slice())
	}
	pub fn text_feature_dim(&self) -> usize {
		self.text_feature_dim
	}
	pub fn text_feature_format(&self) -> Option<&str> {
		self.text_feature_format.as_deref()
	}
	pub fn text_feature_model(&self) -> Option<&str> {
		self.text_feature_model.as_deref()
	}
	pub fn clip_text_features(&self, index: usize) -> Option<&[f32]> {
		self
			.clips
			.get(index)
			.map(|clip| clip.text_features.as_slice())
	}

	/// Upload one standardized clip as `[frames, feature_dim]`.
	pub fn get_item(&self, engine: &Engine, index: usize) -> Result<Matrix> {
		let clip = self
			.clips
			.get(index)
			.ok_or_else(|| Error::out_of_range("HumanML3D clip index is out of range"))?;
		Matrix::from_f32(engine, [clip.frames, self.feature_dim], &clip.features)
	}

	/// De-standardize `[frames, feature_dim]` values in place.
	pub fn denormalize(&self, features: &mut [f32]) -> Result<()> {
		if !features.len().is_multiple_of(self.feature_dim) {
			return Err(Error::invalid_argument(
				"HumanML3D features are not complete rows",
			));
		}
		for row in features.chunks_exact_mut(self.feature_dim) {
			for ((value, deviation), mean) in row.iter_mut().zip(&self.standard_deviation).zip(&self.mean)
			{
				*value = *value * deviation + mean;
			}
		}
		Ok(())
	}
}

/// Recover `[frames,joints,3]` world positions from denormalized features.
pub fn human_ml3d_recover_world_joints(
	features: &[f32],
	frames: usize,
	feature_dim: usize,
) -> Result<Vec<f32>> {
	let joints = joints_for_feature_dim(feature_dim)
		.ok_or_else(|| Error::invalid_argument("HumanML3D feature width must be 263 or 251"))?;
	let expected = frames
		.checked_mul(feature_dim)
		.ok_or_else(|| Error::resource_exhausted("HumanML3D feature count overflows"))?;
	if frames == 0 || features.len() != expected {
		return Err(Error::invalid_argument(
			"HumanML3D feature geometry is invalid",
		));
	}
	let mut yaw = vec![0.0_f32; frames];
	for frame in 1..frames {
		yaw[frame] = yaw[frame - 1] + features[(frame - 1) * feature_dim];
	}
	let mut roots = vec![Vec3::default(); frames];
	for frame in 0..frames {
		let delta = if frame == 0 {
			Vec3::default()
		} else {
			Vec3 {
				x: features[(frame - 1) * feature_dim + 1],
				y: 0.0,
				z: features[(frame - 1) * feature_dim + 2],
			}
		};
		let rotation = Quat {
			x: 0.0,
			y: -yaw[frame].sin(),
			z: 0.0,
			w: yaw[frame].cos(),
		};
		let rotated = rotation
			.try_rotate(delta)
			.ok_or_else(|| Error::data_loss("HumanML3D root rotation is invalid"))?;
		roots[frame] = if frame == 0 {
			rotated
		} else {
			roots[frame - 1] + rotated
		};
		roots[frame].y = features[frame * feature_dim + 3];
	}
	let output_count = frames
		.checked_mul(joints)
		.and_then(|value| value.checked_mul(3))
		.ok_or_else(|| Error::resource_exhausted("HumanML3D joint count overflows usize"))?;
	let mut output = Vec::with_capacity(output_count);
	for frame in 0..frames {
		let rotation = Quat {
			x: 0.0,
			y: -yaw[frame].sin(),
			z: 0.0,
			w: yaw[frame].cos(),
		};
		output.extend_from_slice(&[roots[frame].x, roots[frame].y, roots[frame].z]);
		for joint in 0..joints - 1 {
			let offset = frame * feature_dim + 4 + joint * 3;
			let local = Vec3 {
				x: features[offset],
				y: features[offset + 1],
				z: features[offset + 2],
			};
			let mut world = rotation
				.try_rotate(local)
				.ok_or_else(|| Error::data_loss("HumanML3D joint rotation is invalid"))?;
			world.x += roots[frame].x;
			world.z += roots[frame].z;
			output.extend_from_slice(&[world.x, world.y, world.z]);
		}
	}
	Ok(output)
}

pub fn human_ml3d_mpjpe_cm(predicted: &[f32], target: &[f32]) -> Result<f64> {
	if predicted.is_empty() || predicted.len() != target.len() || !predicted.len().is_multiple_of(3) {
		return Err(Error::invalid_argument(
			"world-joint arrays must be equal nonempty xyz triples",
		));
	}
	let (predicted_rows, []) = predicted.as_chunks::<3>() else {
		unreachable!("divisibility was validated")
	};
	let (target_rows, []) = target.as_chunks::<3>() else {
		unreachable!("divisibility was validated")
	};
	let sum = predicted_rows
		.iter()
		.zip(target_rows)
		.map(|(a, b)| {
			let dx = f64::from(a[0] - b[0]);
			let dy = f64::from(a[1] - b[1]);
			let dz = f64::from(a[2] - b[2]);
			(dx * dx + dy * dy + dz * dz).sqrt()
		})
		.sum::<f64>();
	Ok(100.0 * sum / (predicted.len() / 3) as f64)
}

pub fn human_ml3d_evaluate_motion(
	predicted: &[f32],
	target: &[f32],
	frames: usize,
	feature_dim: usize,
	contact_threshold: f32,
) -> Result<HumanMl3dMotionMetrics> {
	if !contact_threshold.is_finite() || predicted.len() != target.len() {
		return Err(Error::invalid_argument(
			"HumanML3D metric inputs are invalid",
		));
	}
	let joints = joints_for_feature_dim(feature_dim)
		.ok_or_else(|| Error::invalid_argument("HumanML3D feature width must be 263 or 251"))?;
	let predicted_world = human_ml3d_recover_world_joints(predicted, frames, feature_dim)?;
	let target_world = human_ml3d_recover_world_joints(target, frames, feature_dim)?;
	let mpjpe_cm = human_ml3d_mpjpe_cm(&predicted_world, &target_world)?;
	let mut velocity = 0.0;
	let mut velocity_count = 0_usize;
	for frame in 1..frames {
		for joint in 0..joints {
			let i = (frame * joints + joint) * 3;
			let p = i - joints * 3;
			let dx =
				f64::from((predicted_world[i] - predicted_world[p]) - (target_world[i] - target_world[p]));
			let dy = f64::from(
				(predicted_world[i + 1] - predicted_world[p + 1])
					- (target_world[i + 1] - target_world[p + 1]),
			);
			let dz = f64::from(
				(predicted_world[i + 2] - predicted_world[p + 2])
					- (target_world[i + 2] - target_world[p + 2]),
			);
			velocity += (dx * dx + dy * dy + dz * dz).sqrt();
			velocity_count += 1;
		}
	}
	let foot_joints = [7_usize, 10, 8, 11];
	let contact_offset = feature_dim - 4;
	let mut correct = 0;
	let mut contacts = 0;
	let mut skate = 0.0;
	let mut skate_count = 0;
	for frame in 0..frames {
		for (contact, joint) in foot_joints.into_iter().enumerate() {
			let k = frame * feature_dim + contact_offset + contact;
			let p = predicted[k] >= contact_threshold;
			let t = target[k] >= contact_threshold;
			correct += usize::from(p == t);
			contacts += 1;
			if frame > 0 && t && joint < joints {
				let i = (frame * joints + joint) * 3;
				let previous = i - joints * 3;
				let dx = f64::from(predicted_world[i] - predicted_world[previous]);
				let dz = f64::from(predicted_world[i + 2] - predicted_world[previous + 2]);
				skate += (dx * dx + dz * dz).sqrt();
				skate_count += 1;
			}
		}
	}
	Ok(HumanMl3dMotionMetrics {
		mpjpe_cm,
		velocity_error_cm_per_frame: if velocity_count == 0 {
			0.0
		} else {
			100.0 * velocity / velocity_count as f64
		},
		contact_accuracy: correct as f64 / contacts as f64,
		foot_skate_cm_per_frame: if skate_count == 0 {
			0.0
		} else {
			100.0 * skate / skate_count as f64
		},
	})
}

struct NpyF32 {
	shape: Vec<usize>,
	data: Vec<f32>,
}
fn load_npy_f32(path: &Path) -> Result<NpyF32> {
	let bytes = fs::read(path).map_err(|source| Error::io("read NumPy array", source))?;
	if bytes.get(..6) != Some(b"\x93NUMPY") {
		return Err(Error::data_loss("invalid NumPy magic"));
	}
	let major = *bytes
		.get(6)
		.ok_or_else(|| Error::data_loss("truncated NumPy version"))?;
	let (header_len, prefix): (usize, usize) = if major == 1 {
		(
			usize::from(u16::from_le_bytes(
				bytes
					.get(8..10)
					.and_then(|v| v.try_into().ok())
					.ok_or_else(|| Error::data_loss("truncated NumPy header"))?,
			)),
			10,
		)
	} else if major == 2 {
		(
			usize::try_from(u32::from_le_bytes(
				bytes
					.get(8..12)
					.and_then(|v| v.try_into().ok())
					.ok_or_else(|| Error::data_loss("truncated NumPy header"))?,
			))
			.map_err(|_| Error::data_loss("NumPy header exceeds usize"))?,
			12,
		)
	} else {
		return Err(Error::data_loss("unsupported NumPy version"));
	};
	let end = prefix
		.checked_add(header_len)
		.ok_or_else(|| Error::data_loss("NumPy header range overflows usize"))?;
	let header = std::str::from_utf8(
		bytes
			.get(prefix..end)
			.ok_or_else(|| Error::data_loss("truncated NumPy header"))?,
	)
	.map_err(|_| Error::data_loss("NumPy header is not ASCII"))?;
	if !header.contains("'<f4'") || header.contains("'fortran_order': True") {
		return Err(Error::data_loss(
			"NumPy array must be little-endian C-order F32",
		));
	}
	let shape_start = header
		.find("'shape':")
		.and_then(|p| header[p..].find('(').map(|q| p + q + 1))
		.ok_or_else(|| Error::data_loss("NumPy shape is missing"))?;
	let shape_end = header[shape_start..]
		.find(')')
		.map(|p| shape_start + p)
		.ok_or_else(|| Error::data_loss("NumPy shape is truncated"))?;
	let shape = header[shape_start..shape_end]
		.split(',')
		.filter_map(|s| {
			let s = s.trim();
			(!s.is_empty()).then(|| s.parse::<usize>())
		})
		.collect::<std::result::Result<Vec<_>, _>>()
		.map_err(|_| Error::data_loss("NumPy shape is invalid"))?;
	let count = shape.iter().try_fold(1_usize, |n, d| {
		n.checked_mul(*d)
			.ok_or_else(|| Error::data_loss("NumPy element count overflows"))
	})?;
	let payload = bytes
		.get(end..)
		.ok_or_else(|| Error::data_loss("NumPy payload is missing"))?;
	let byte_count = count
		.checked_mul(4)
		.ok_or_else(|| Error::data_loss("NumPy payload size overflows usize"))?;
	if payload.len() != byte_count {
		return Err(Error::data_loss("NumPy payload size mismatch"));
	}
	let (words, []) = payload.as_chunks::<4>() else {
		unreachable!("payload byte count was validated")
	};
	let data = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
	Ok(NpyF32 { shape, data })
}

fn read_nonempty_lines(path: &Path) -> Result<Vec<String>> {
	let text = fs::read_to_string(path).map_err(|source| Error::io("read HumanML3D text", source))?;
	Ok(
		text
			.lines()
			.map(str::trim_end)
			.filter(|line| !line.is_empty())
			.map(str::to_owned)
			.collect(),
	)
}
fn read_captions(path: &Path) -> Result<Vec<HumanMl3dCaption>> {
	let Ok(text) = fs::read_to_string(path) else {
		return Ok(Vec::new());
	};
	Ok(
		text
			.lines()
			.filter(|line| !line.is_empty())
			.map(|line| {
				let fields = line.split('#').collect::<Vec<_>>();
				let range_seconds = if fields.len() >= 4 {
					fields[2]
						.parse::<f32>()
						.ok()
						.zip(fields[3].parse::<f32>().ok())
						.filter(|(a, b)| a.is_finite() && b.is_finite() && (*a != 0.0 || *b != 0.0))
				} else {
					None
				};
				HumanMl3dCaption {
					text: fields[0].to_owned(),
					range_seconds,
				}
			})
			.collect(),
	)
}
struct TextManifest {
	format: String,
	model: String,
	dimension: usize,
}
fn load_text_manifest(path: &Path) -> Result<Option<TextManifest>> {
	let Ok(bytes) = fs::read(path) else {
		return Ok(None);
	};
	let value: Value = serde_json::from_slice(&bytes)
		.map_err(|_| Error::data_loss("invalid text-feature manifest"))?;
	let format = value.get("format").and_then(Value::as_str);
	let model = value.get("model").and_then(Value::as_str);
	let dimension = value
		.get("dim")
		.and_then(Value::as_u64)
		.and_then(|v| usize::try_from(v).ok());
	if format != Some("oa_clip_text_v1")
		|| model.is_none_or(str::is_empty)
		|| dimension.is_none_or(|v| v == 0)
		|| value.get("dtype").and_then(Value::as_str) != Some("float32")
		|| value.get("feature").and_then(Value::as_str)
			!= Some("CLIPTextModelWithProjection.text_embeds")
	{
		return Ok(None);
	}
	Ok(Some(TextManifest {
		format: format.unwrap_or_default().to_owned(),
		model: model.unwrap_or_default().to_owned(),
		dimension: dimension.unwrap_or_default(),
	}))
}
fn load_clip_text_features(
	root: &Path,
	id: &str,
	captions: usize,
	dimension: usize,
) -> Result<Vec<f32>> {
	let array = load_npy_f32(&root.join("text_feats").join(format!("{id}.npy")))?;
	let valid = array.shape == [captions, dimension] || (captions == 1 && array.shape == [dimension]);
	if !valid {
		return Err(Error::data_loss(
			"text feature shape does not match captions",
		));
	}
	Ok(array.data)
}
const fn joints_for_feature_dim(feature_dim: usize) -> Option<usize> {
	match feature_dim {
		263 => Some(22),
		251 => Some(21),
		_ => None,
	}
}
