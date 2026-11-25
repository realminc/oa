test_vk!(
	metric_host_observations_match_the_device_scalars,
	runtime,
	{
		let loss = oa::Matrix::new_from_f32([1], &[0.375])?;
		assert_eq!(oa::ml::metric::scalar_loss(&loss)?, 0.375);

		let logits = oa::Matrix::new_from_f32(
			[4, 3],
			&[4.0, 4.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 3.0, 3.0, 1.0, 0.0],
		)?;
		let labels = oa::Matrix::new_from_slice([4], &[0_u32, 1, 0, 0])?;
		assert_eq!(oa::ml::metric::accuracy(&logits, &labels)?, 0.75);
		Ok(())
	}
);

test_vk!(metric_rejects_non_scalar_loss, runtime, {
	let loss = oa::Matrix::new_from_f32([2], &[0.25, 0.5])?;
	let error = oa::ml::metric::scalar_loss(&loss).expect_err("vector loss was accepted");
	assert_eq!(error.kind(), oa::ErrorKind::InvalidArgument);
	Ok(())
});

test_vk!(
	precision_recall_f1_and_macro_f1_match_hand_computed_counts,
	runtime,
	{
		// 4 samples, 3 classes. Predictions: argmax rows = [2,1,0,0].
		let logits = oa::Matrix::new_from_f32(
			[4, 3],
			&[1.0, 2.0, 4.0, 0.5, 3.0, 1.0, 2.0, 1.0, 0.5, 4.0, 0.0, 1.0],
		)?;
		let labels = oa::Matrix::new_from_slice([4], &[2_u32, 0, 0, 1])?;
		// class 0: TP=1 (row2), FP=1 (row3), FN=1 (row1) → P=1/2 R=1/2 F1=1/2
		let counts = oa::ml::metric::precision_recall_f1(&logits, &labels, 0)?;
		assert_eq!(counts.true_positives, 1);
		assert_eq!(counts.false_positives, 1);
		assert_eq!(counts.false_negatives, 1);
		assert!((counts.precision() - 0.5).abs() < 1.0e-6);
		assert!((counts.recall() - 0.5).abs() < 1.0e-6);
		assert!((counts.f1() - 0.5).abs() < 1.0e-6);
		// class 2: TP=1, FP=0, FN=0 → F1=1; class 1: TP=0,FN=... absent → 0
		let macro_f1 = oa::ml::metric::macro_f1(&logits, &labels, 3)?;
		assert!((macro_f1 - 0.5).abs() < 1.0e-6, "macro f1 {macro_f1}");
		assert!(oa::ml::metric::precision_recall_f1(&logits, &labels, 3).is_err());
		Ok(())
	}
);

test_vk!(roc_auc_matches_the_rank_sum_oracle_with_ties, runtime, {
	// Perfect separation → 1.0; reversed → 0.0; ties → 0.5 partial credit.
	let perfect = oa::Matrix::new_from_f32([4], &[0.1, 0.2, 0.8, 0.9])?;
	let labels = oa::Matrix::new_from_slice([4], &[0_u32, 0, 1, 1])?;
	assert_eq!(oa::ml::metric::roc_auc(&perfect, &labels)?, 1.0);
	let reversed = oa::Matrix::new_from_f32([4], &[0.9, 0.8, 0.2, 0.1])?;
	assert_eq!(oa::ml::metric::roc_auc(&reversed, &labels)?, 0.0);
	// One tie pair across classes: p(0.9) wins twice, p(0.5) ties n(0.5)
	// and wins over n(0.2) → 3.5/4 pairs.
	let tied = oa::Matrix::new_from_f32([4], &[0.5, 0.2, 0.9, 0.5])?;
	let tied_labels = oa::Matrix::new_from_slice([4], &[0_u32, 0, 1, 1])?;
	let auc = oa::ml::metric::roc_auc(&tied, &tied_labels)?;
	assert!((auc - 0.875).abs() < 1.0e-6, "tied auc {auc}");
	assert!(
		oa::ml::metric::roc_auc(&perfect, &oa::Matrix::new_from_slice([4], &[1_u32; 4])?).is_err()
	);
	Ok(())
});

#[test]
fn bleu_matches_nltk_style_corpus_oracle() -> oa::Result<()> {
	// Perfect single-sentence hypothesis → 1.0.
	let references = vec![vec![1_u32, 2, 3, 4]];
	let hypotheses = vec![vec![1_u32, 2, 3, 4]];
	assert_eq!(oa::ml::metric::bleu(&references, &hypotheses, 4)?, 1.0);
	// Short hypothesis: unigram/bigram/trigram/4-gram precision 1.0 but
	// brevity penalty exp(1 - 4/2) = e^-1.
	let short_hyp = vec![vec![1_u32, 2]];
	assert!((oa::ml::metric::bleu(&references, &short_hyp, 2)? - (-1.0_f32).exp()).abs() < 1.0e-6);
	// No overlap → 0.
	let no_match = vec![vec![9_u32, 9, 9, 9]];
	assert_eq!(oa::ml::metric::bleu(&references, &no_match, 4)?, 0.0);
	assert!(oa::ml::metric::bleu(&references, &hypotheses, 0).is_err());
	assert!(oa::ml::metric::bleu(&references, &[vec![1, 2], vec![3]], 2).is_err());
	Ok(())
}
