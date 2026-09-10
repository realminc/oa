use oa::{DType, ErrorKind, Matrix, OpValueKind, vision::NmsConfig};

#[test]
fn box_iou_has_the_direct_vision_signature() {
	let _: fn(&Matrix, &Matrix) -> oa::Result<Matrix> = oa::vision::box_iou;
	let _: fn(&Matrix, &Matrix, &Matrix, NmsConfig) -> oa::Result<oa::vision::NmsResult> =
		oa::vision::nms;
	let _: fn(&Matrix, &Matrix, i32) -> oa::Result<Matrix> = oa::vision::confusion_matrix;
	let _: fn(&Matrix, &Matrix) -> oa::Result<Matrix> = oa::vision::binary_mask_counts;
	assert_eq!(NmsConfig::default().iou_threshold, 0.45);
	assert_eq!(NmsConfig::default().max_detections, 100);
}

test_vk!(box_iou_matches_independent_pairwise_oracle, engine, {
	let boxes_a = vec![0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0];
	let boxes_b = vec![0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0];
	let left = Matrix::from_f32(&engine, [2, 4], &boxes_a)?;
	let right = Matrix::from_f32(&engine, [2, 4], &boxes_b)?;
	let output = oa::vision::box_iou(&left, &right)?;
	assert_eq!(output.shape(), [2, 2]);
	assert_eq!(output.dtype(), DType::F32);
	assert_eq!(
		output
			.try_read_f32()
			.expect_err("recorded IoU output was prematurely readable")
			.kind(),
		ErrorKind::NotReady
	);
	let expected = pairwise_iou_reference(&boxes_a, &boxes_b);
	for (index, (actual, expected)) in output.read_f32()?.iter().zip(expected).enumerate() {
		assert!(
			(actual - expected).abs() <= 1.0e-6,
			"pair {index}: expected {expected}, found {actual}"
		);
	}
	assert_eq!(left.read_f32()?, boxes_a);
	assert_eq!(right.read_f32()?, boxes_b);
	Ok(())
});

test_vk!(
	box_iou_covers_odd_rows_and_malformed_values_without_aliasing,
	engine,
	{
		let boxes_a = vec![
			0.0,
			0.0,
			2.0,
			2.0,
			1.0,
			1.0,
			-1.0,
			2.0,
			f32::NAN,
			0.0,
			1.0,
			1.0,
		];
		let boxes_b = vec![
			0.0,
			0.0,
			2.0,
			2.0,
			5.0,
			5.0,
			1.0,
			1.0,
			1.0,
			1.0,
			0.0,
			0.0,
			0.0,
			0.0,
			f32::INFINITY,
			1.0,
			0.5,
			0.0,
			2.0,
			2.0,
		];
		let left = Matrix::from_f32(&engine, [3, 4], &boxes_a)?;
		let right = Matrix::from_f32(&engine, [5, 4], &boxes_b)?;
		let (plan, output) = engine.capture(|| oa::vision::box_iou(&left, &right))?;
		assert_eq!(output.shape(), [3, 5]);
		let graph = plan.semantic_graph();
		assert_eq!(graph.operations().len(), 1);
		assert_eq!(graph.operations()[0].name(), "oa::vision::box_iou");
		assert_eq!(
			graph.operations()[0].contract_hash(),
			oa::core::operation::vision::BOX_IOU.hash()
		);
		assert!(graph.operations()[0].attributes().is_empty());
		assert!(graph.operations()[0].aliases().is_empty());
		assert!(
			graph
				.values()
				.iter()
				.all(|value| value.kind() == OpValueKind::Matrix)
		);
		assert_eq!(plan.semantic_lowering().direct_op_count(), 1);
		let report: serde_json::Value =
			serde_json::from_str(&plan.debug_report_json("vision-box-iou"))
				.expect("execution report must be valid JSON");
		assert_eq!(report["nodes"][0]["kernel"], "vision.box_iou.f32");
		assert_eq!(
			report["nodes"][0]["physical_write"]["writes"][0]["binding"],
			2
		);
		engine.submit(&plan)?.wait()?;
		let expected = pairwise_iou_reference(&boxes_a, &boxes_b);
		for (actual, expected) in output.read_f32()?.iter().zip(expected) {
			assert!((actual - expected).abs() <= 1.0e-6);
		}
		Ok(())
	}
);

test_vk!(box_iou_rejects_invalid_structural_contracts, engine, {
	let other_engine = oa::Engine::new()?;
	let valid = Matrix::from_f32(&engine, [1, 4], &[0.0, 0.0, 1.0, 1.0])?;
	let rank_one = Matrix::from_f32(&engine, [4], &[0.0, 0.0, 1.0, 1.0])?;
	let wrong_columns = Matrix::from_f32(&engine, [1, 3], &[0.0, 0.0, 1.0])?;
	let empty = Matrix::from_f32(&engine, [0, 4], &[])?;
	let integer = Matrix::from_slice(&engine, [1, 4], &[0_i32, 0, 1, 1])?;
	let foreign = Matrix::from_f32(&other_engine, [1, 4], &[0.0, 0.0, 1.0, 1.0])?;
	for error in [
		oa::vision::box_iou(&rank_one, &valid)
			.err()
			.expect("rank-one input was accepted"),
		oa::vision::box_iou(&wrong_columns, &valid)
			.err()
			.expect("wrong width was accepted"),
		oa::vision::box_iou(&empty, &valid)
			.err()
			.expect("empty input was accepted"),
		oa::vision::box_iou(&integer, &valid)
			.err()
			.expect("I32 input was accepted"),
		oa::vision::box_iou(&valid, &foreign)
			.err()
			.expect("cross-engine input was accepted"),
	] {
		assert_eq!(error.kind(), ErrorKind::InvalidArgument);
	}
	Ok(())
});

test_vk!(nms_is_class_aware_score_ranked_and_deterministic, engine, {
	let boxes = Matrix::from_f32(
		&engine,
		[5, 4],
		&[
			0.50, 0.50, 0.40, 0.40, 0.51, 0.50, 0.40, 0.40, 0.50, 0.50, 0.40, 0.40, 0.10, 0.10,
			0.10, 0.10, 0.90, 0.90, 0.10, 0.10,
		],
	)?;
	let scores = Matrix::from_f32(&engine, [5], &[0.90, 0.80, 0.85, 0.70, 0.70])?;
	let classes = Matrix::from_slice(&engine, [5], &[0_i32, 0, 1, 0, 0])?;
	let config = NmsConfig {
		iou_threshold: 0.5,
		max_detections: 5,
		..NmsConfig::default()
	};
	let result = oa::vision::nms(&boxes, &scores, &classes, config)?;
	assert_eq!(result.indices.shape(), [5]);
	assert_eq!(result.indices.dtype(), DType::I32);
	assert_eq!(result.count.dtype(), DType::U32);
	assert_eq!(result.count.read::<u32>()?, vec![4]);
	assert_eq!(result.indices.read::<i32>()?, vec![0, 2, 3, 4, -1]);

	let class_agnostic = oa::vision::nms(
		&boxes,
		&scores,
		&classes,
		NmsConfig {
			class_agnostic: true,
			max_detections: 3,
			..config
		},
	)?;
	assert_eq!(class_agnostic.count.read::<u32>()?, vec![3]);
	assert_eq!(class_agnostic.indices.read::<i32>()?, vec![0, 3, 4]);
	Ok(())
});

test_vk!(
	confusion_and_binary_counts_match_host_oracles_and_atomic_contract,
	engine,
	{
		let predicted = Matrix::from_slice(&engine, [7], &[0_i32, 1, 2, 1, -1, 3, 2])?;
		let target = Matrix::from_slice(&engine, [7], &[0_i32, 2, 2, 1, 0, 0, 7])?;
		let (plan, confusion) =
			engine.capture(|| oa::vision::confusion_matrix(&predicted, &target, 3))?;
		assert_eq!(confusion.shape(), [3, 3]);
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("confusion"))
			.expect("execution report must be valid JSON");
		assert_eq!(
			report["nodes"][0]["kernel"],
			"vision.confusion_matrix_clear.u32"
		);
		assert_eq!(report["nodes"][1]["kernel"], "vision.confusion_matrix.i32");
		assert_eq!(
			report["nodes"][1]["physical_write"]["writes"][0]["partition"],
			"shared_atomic_contributors"
		);
		assert_eq!(
			report["nodes"][1]["physical_write"]["writes"][0]["collision"],
			"atomic_u32"
		);
		engine.submit(&plan)?.wait()?;
		assert_eq!(confusion.read::<u32>()?, vec![1, 0, 0, 0, 1, 0, 0, 1, 1]);
		engine.submit(&plan)?.wait()?;
		assert_eq!(confusion.read::<u32>()?, vec![1, 0, 0, 0, 1, 0, 0, 1, 1]);

		let mask_predicted = Matrix::from_slice(&engine, [7], &[1_u8, 1, 0, 0, 3, 0, 1])?;
		let mask_target = Matrix::from_slice(&engine, [7], &[1_u8, 0, 1, 0, 1, 0, 0])?;
		let (mask_plan, mask_counts) =
			engine.capture(|| oa::vision::binary_mask_counts(&mask_predicted, &mask_target))?;
		engine.submit(&mask_plan)?.wait()?;
		assert_eq!(mask_counts.read::<u32>()?, vec![2, 2, 1, 2]);
		engine.submit(&mask_plan)?.wait()?;
		assert_eq!(mask_counts.read::<u32>()?, vec![2, 2, 1, 2]);
		Ok(())
	}
);

test_vk!(
	detection_evaluation_matches_donor_dataset_oracle_and_split_graph,
	engine,
	{
		let predicted_boxes = Matrix::from_f32(
			&engine,
			[4, 4],
			&[
				0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.50, 0.50, 0.20, 0.20, 0.80, 0.80,
				0.20, 0.20,
			],
		)?;
		let predicted_scores = Matrix::from_f32(&engine, [4], &[0.90, 0.80, 0.70, 0.60])?;
		let predicted_classes = Matrix::from_slice(&engine, [4], &[0_i32, 0, 1, 0])?;
		let predicted_images = Matrix::from_slice(&engine, [4], &[0_i32, 0, 0, 1])?;
		let target_boxes = Matrix::from_f32(
			&engine,
			[3, 4],
			&[
				0.20, 0.20, 0.20, 0.20, 0.80, 0.80, 0.20, 0.20, 0.50, 0.50, 0.20, 0.20,
			],
		)?;
		let target_classes = Matrix::from_slice(&engine, [3], &[0_i32, 0, 1])?;
		let target_images = Matrix::from_slice(&engine, [3], &[0_i32, 1, 0])?;
		let thresholds = Matrix::from_f32(&engine, [2], &[0.50, 0.75])?;
		let (plan, result) = engine.capture(|| {
			oa::vision::evaluate(
				&predicted_boxes,
				&predicted_scores,
				&predicted_classes,
				&predicted_images,
				&target_boxes,
				&target_classes,
				&target_images,
				&thresholds,
				2,
				0.75,
			)
		})?;
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::vision::evaluate"
		);
		assert_eq!(plan.semantic_lowering().decomposed_op_count(), 1);
		let report: serde_json::Value = serde_json::from_str(&plan.debug_report_json("evaluate"))
			.expect("execution report must be valid JSON");
		let kernels = report["nodes"]
			.as_array()
			.expect("nodes must be an array")
			.iter()
			.map(|node| node["kernel"].as_str().expect("kernel must be text"))
			.collect::<Vec<_>>();
		assert_eq!(
			kernels,
			[
				"vision.detection_metric_curves.f32",
				"vision.detection_average_precision.f32",
				"vision.detection_mean_average_precision.f32",
			]
		);
		engine.submit(&plan)?.wait()?;
		assert_eq!(result.counts.shape(), [2, 2, 3]);
		assert_eq!(
			result.counts.read::<u32>()?,
			vec![1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1]
		);
		let per_class = result.per_class.read_f32()?;
		for threshold in 0..2 {
			let base = threshold * 8;
			assert!((per_class[base] - 0.5).abs() <= 1.0e-6);
			assert!((per_class[base + 1] - 0.5).abs() <= 1.0e-6);
			assert!((per_class[base + 2] - 0.5).abs() <= 1.0e-6);
			assert!((per_class[base + 3] - 84.333_336 / 101.0).abs() <= 1.0e-5);
			assert_eq!(&per_class[base + 4..base + 7], &[0.0, 0.0, 0.0]);
			assert!((per_class[base + 7] - 1.0).abs() <= 1.0e-6);
		}
		for value in result.mean_average_precision_by_threshold.read_f32()? {
			assert!((value - 0.917_491_7).abs() <= 1.0e-5);
		}
		assert!((result.mean_average_precision.read_f32()?[0] - 0.917_491_7).abs() <= 1.0e-5);
		engine.submit(&plan)?.wait()?;
		assert!((result.mean_average_precision.read_f32()?[0] - 0.917_491_7).abs() <= 1.0e-5);
		Ok(())
	}
);

test_vk!(
	segmentation_evaluation_matches_oracle_and_ignores_invalid_labels,
	engine,
	{
		let predicted = Matrix::from_slice(&engine, [2, 3], &[0_i32, 1, 2, 1, 8, 0])?;
		let target = Matrix::from_slice(&engine, [2, 3], &[0_i32, 2, 2, 1, 0, -1])?;
		let (plan, result) =
			engine.capture(|| oa::vision::evaluate_segmentation(&predicted, &target, 3))?;
		assert_eq!(plan.semantic_graph().operations().len(), 1);
		assert_eq!(
			plan.semantic_graph().operations()[0].name(),
			"oa::vision::evaluate_segmentation"
		);
		assert_eq!(plan.semantic_lowering().decomposed_op_count(), 1);
		engine.submit(&plan)?.wait()?;
		assert_eq!(
			result.confusion.read::<u32>()?,
			vec![1, 0, 0, 0, 1, 0, 0, 1, 1]
		);
		let expected = [
			1.0,
			1.0,
			1.0,
			1.0,
			0.5,
			1.0,
			2.0 / 3.0,
			0.5,
			1.0,
			0.5,
			2.0 / 3.0,
			0.5,
		];
		for (actual, expected) in result.per_class.read_f32()?.iter().zip(expected) {
			assert!((actual - expected).abs() <= 1.0e-6);
		}
		assert!((result.mean_iou.read_f32()?[0] - 2.0 / 3.0).abs() <= 1.0e-6);
		assert!((result.pixel_accuracy.read_f32()?[0] - 0.75).abs() <= 1.0e-6);
		engine.submit(&plan)?.wait()?;
		assert_eq!(
			result.confusion.read::<u32>()?,
			vec![1, 0, 0, 0, 1, 0, 0, 1, 1]
		);
		Ok(())
	}
);

test_vk!(
	complete_detection_surface_rejects_invalid_contracts,
	engine,
	{
		let other_engine = oa::Engine::new()?;
		let boxes = Matrix::from_f32(&engine, [1, 4], &[0.0, 0.0, 1.0, 1.0])?;
		let scores = Matrix::from_f32(&engine, [1], &[1.0])?;
		let classes = Matrix::from_slice(&engine, [1], &[0_i32])?;
		let images = Matrix::from_slice(&engine, [1], &[0_i32])?;
		let thresholds = Matrix::from_f32(&engine, [1], &[0.5])?;
		let foreign_scores = Matrix::from_f32(&other_engine, [1], &[1.0])?;
		let labels = Matrix::from_slice(&engine, [2], &[0_i32, 1])?;
		let short_labels = Matrix::from_slice(&engine, [1], &[0_i32])?;
		let masks = Matrix::from_slice(&engine, [2], &[0_u8, 1])?;
		let short_masks = Matrix::from_slice(&engine, [1], &[0_u8])?;

		for error in [
			oa::vision::nms(&boxes, &foreign_scores, &classes, NmsConfig::default())
				.err()
				.expect("cross-engine NMS was accepted"),
			oa::vision::nms(
				&boxes,
				&scores,
				&classes,
				NmsConfig {
					iou_threshold: 1.01,
					..NmsConfig::default()
				},
			)
			.err()
			.expect("invalid IoU threshold was accepted"),
			oa::vision::nms(
				&boxes,
				&scores,
				&classes,
				NmsConfig {
					max_detections: 0,
					..NmsConfig::default()
				},
			)
			.err()
			.expect("zero NMS maximum was accepted"),
			oa::vision::confusion_matrix(&labels, &short_labels, 2)
				.err()
				.expect("unequal confusion labels were accepted"),
			oa::vision::confusion_matrix(&labels, &labels, 0)
				.err()
				.expect("zero class count was accepted"),
			oa::vision::binary_mask_counts(&masks, &short_masks)
				.err()
				.expect("unequal masks were accepted"),
			oa::vision::evaluate(
				&boxes,
				&scores,
				&classes,
				&images,
				&boxes,
				&classes,
				&images,
				&thresholds,
				0,
				0.0,
			)
			.err()
			.expect("zero evaluation class count was accepted"),
			oa::vision::evaluate(
				&boxes,
				&scores,
				&classes,
				&images,
				&boxes,
				&classes,
				&images,
				&thresholds,
				1,
				f32::NAN,
			)
			.err()
			.expect("NaN evaluation score threshold was accepted"),
			oa::vision::evaluate_segmentation(&masks, &masks, 2)
				.err()
				.expect("U8 segmentation labels were accepted"),
			oa::vision::evaluate_segmentation(&labels, &short_labels, 2)
				.err()
				.expect("unequal segmentation labels were accepted"),
		] {
			assert_eq!(error.kind(), ErrorKind::InvalidArgument);
		}
		Ok(())
	}
);

fn pairwise_iou_reference(boxes_a: &[f32], boxes_b: &[f32]) -> Vec<f32> {
	let mut output = Vec::with_capacity(boxes_a.len() / 4 * boxes_b.len() / 4);
	for a in boxes_a.as_chunks::<4>().0 {
		for b in boxes_b.as_chunks::<4>().0 {
			output.push(box_iou_reference(a, b));
		}
	}
	output
}

fn box_iou_reference(a: &[f32], b: &[f32]) -> f32 {
	if a.iter().chain(b).any(|value| !value.is_finite()) {
		return 0.0;
	}
	let a_width = a[2].max(0.0);
	let a_height = a[3].max(0.0);
	let b_width = b[2].max(0.0);
	let b_height = b[3].max(0.0);
	let (a_x0, a_y0, a_x1, a_y1) = (
		a[0] - a_width * 0.5,
		a[1] - a_height * 0.5,
		a[0] + a_width * 0.5,
		a[1] + a_height * 0.5,
	);
	let (b_x0, b_y0, b_x1, b_y1) = (
		b[0] - b_width * 0.5,
		b[1] - b_height * 0.5,
		b[0] + b_width * 0.5,
		b[1] + b_height * 0.5,
	);
	let intersection_width = (a_x1.min(b_x1) - a_x0.max(b_x0)).max(0.0);
	let intersection_height = (a_y1.min(b_y1) - a_y0.max(b_y0)).max(0.0);
	let intersection = intersection_width * intersection_height;
	let union_area = a_width * a_height + b_width * b_height - intersection;
	if union_area.is_finite() && union_area > 0.0 {
		intersection / union_area
	} else {
		0.0
	}
}
