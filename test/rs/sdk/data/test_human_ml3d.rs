use std::time::SystemTime;

use oa::sdk::data::{HumanMl3dDataset, human_ml3d_evaluate_motion};

#[test]
fn geometry_metrics_match_the_donor_identity_and_perturbation_oracle() -> oa::Result<()> {
	const FRAMES: usize = 3;
	const FEATURES: usize = 263;
	let mut target = vec![0.0_f32; FRAMES * FEATURES];
	for frame in 0..FRAMES {
		for contact in 0..4 {
			target[frame * FEATURES + FEATURES - 4 + contact] = 1.0;
		}
	}
	let identical = human_ml3d_evaluate_motion(&target, &target, FRAMES, FEATURES, 0.5)?;
	assert_eq!(identical.mpjpe_cm, 0.0);
	assert_eq!(identical.velocity_error_cm_per_frame, 0.0);
	assert_eq!(identical.foot_skate_cm_per_frame, 0.0);
	assert_eq!(identical.contact_accuracy, 1.0);

	let mut changed = target.clone();
	changed[FEATURES + 1] = 1.0;
	changed[FEATURES + FEATURES - 4] = 0.0;
	let perturbed = human_ml3d_evaluate_motion(&changed, &target, FRAMES, FEATURES, 0.5)?;
	assert!(perturbed.mpjpe_cm > 0.0);
	assert!(perturbed.velocity_error_cm_per_frame > 0.0);
	assert!(perturbed.contact_accuracy < 1.0);
	Ok(())
}

#[test]
fn synthetic_corpus_preserves_normalization_captions_and_text_contract() -> oa::Result<()> {
	let directory = std::env::temp_dir().join(format!(
		"oars-humanml3d-{}-{}",
		std::process::id(),
		SystemTime::now()
			.duration_since(SystemTime::UNIX_EPOCH)
			.expect("system clock predates Unix epoch")
			.as_nanos()
	));
	std::fs::create_dir_all(directory.join("new_joint_vecs")).expect("create motion directory");
	std::fs::create_dir_all(directory.join("texts")).expect("create caption directory");
	std::fs::create_dir_all(directory.join("text_feats")).expect("create text feature directory");
	write_npy(&directory.join("Mean.npy"), &[263], &vec![1.0; 263]);
	write_npy(&directory.join("Std.npy"), &[263], &vec![2.0; 263]);
	write_npy(
		&directory.join("new_joint_vecs/walk.npy"),
		&[2, 263],
		&vec![3.0; 526],
	);
	write_npy(
		&directory.join("text_feats/walk.npy"),
		&[2, 3],
		&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
	);
	std::fs::write(directory.join("train.txt"), "missing\nwalk\n").expect("write split file");
	std::fs::write(
		directory.join("texts/walk.txt"),
		"walk forward#tags#0#0\nturn left#tags#1.5#2.5\n",
	)
	.expect("write captions");
	std::fs::write(
		directory.join("text_feats/manifest.json"),
		r#"{"format":"oa_clip_text_v1","model":"clip-test","dim":3,"dtype":"float32","feature":"CLIPTextModelWithProjection.text_embeds"}"#,
	)
	.expect("write text manifest");

	let dataset = HumanMl3dDataset::open_cmp(&directory, "train", 0)?;
	assert_eq!(dataset.len(), 1);
	assert_eq!(dataset.feature_dim(), 263);
	assert_eq!(dataset.num_joints(), 22);
	assert_eq!(dataset.total_frames(), 2);
	assert_eq!(dataset.clip_id(0), Some("walk"));
	assert_eq!(dataset.clip_frames(0), Some(2));
	assert!(
		dataset
			.clip_data(0)
			.expect("clip zero")
			.iter()
			.all(|value| *value == 1.0)
	);
	let captions = dataset.clip_captions(0).expect("captions");
	assert_eq!(captions.len(), 2);
	assert_eq!(captions[0].range_seconds, None);
	assert_eq!(captions[1].range_seconds, Some((1.5, 2.5)));
	assert_eq!(dataset.text_feature_dim(), 3);
	assert_eq!(dataset.text_feature_format(), Some("oa_clip_text_v1"));
	assert_eq!(dataset.text_feature_model(), Some("clip-test"));
	assert_eq!(
		dataset.clip_text_features(0),
		Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..])
	);
	let mut denormalized = dataset.clip_data(0).expect("clip zero").to_vec();
	dataset.denormalize(&mut denormalized)?;
	assert!(denormalized.iter().all(|value| *value == 3.0));

	std::fs::remove_dir_all(directory).expect("remove synthetic HumanML3D corpus");
	Ok(())
}

fn write_npy(path: &std::path::Path, shape: &[usize], values: &[f32]) {
	assert_eq!(shape.iter().product::<usize>(), values.len());
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
	std::fs::write(path, bytes).expect("write NumPy fixture");
}
