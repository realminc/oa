use std::{
	fs,
	time::{SystemTime, UNIX_EPOCH},
};

use oa::ml::BpeTokenizer;

#[test]
fn bpe_learns_full_width_nested_tokens_and_round_trips() {
	let corpus = b"forward strike forward strike backward guard forward strike";
	let mut tokenizer = BpeTokenizer::new(320);
	tokenizer.train(corpus, 64);

	assert!(tokenizer.num_merges() > 0);
	assert_eq!(tokenizer.vocab_size(), 256 + tokenizer.num_merges());
	let encoded = tokenizer.encode(corpus);
	assert!(encoded.len() < corpus.len());
	assert!(encoded.iter().any(|token| *token >= 256));
	assert_eq!(
		tokenizer.decode(&encoded).expect("decode should succeed"),
		corpus
	);
}

#[test]
fn bpe_training_and_prompt_alignment_are_deterministic() {
	let corpus = b"left right left right fast slow fast slow";
	let mut left = BpeTokenizer::new(300);
	let mut right = BpeTokenizer::new(300);
	left.train(corpus, 44);
	right.train(corpus, 44);
	assert_eq!(left.merges(), right.merges());
	assert_eq!(left.encode(corpus), right.encode(corpus));

	let prompt = left.encode_prompt(b"left right", 8);
	let encoded = left.encode(b"left right");
	let copied = encoded.len().min(8);
	assert_eq!(&prompt[8 - copied..], &encoded[encoded.len() - copied..]);
	assert!(prompt[..8 - copied].iter().all(|token| *token == 0));
}

#[test]
fn bpe_persists_exact_vocabulary_and_rejects_invalid_dependencies() -> oa::Result<()> {
	let corpus = "a fighter steps forward and swings the sword";
	let unique = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.expect("system time must follow Unix epoch")
		.as_nanos();
	let path = std::env::temp_dir().join(format!("oa_bpe_{}_{}.txt", std::process::id(), unique));
	let mut trained = BpeTokenizer::new(320);
	trained.train_text(corpus, 64);
	trained.save(&path)?;

	let mut loaded = BpeTokenizer::default();
	loaded.load(&path)?;
	assert_eq!(loaded.merges(), trained.merges());
	assert_eq!(loaded.encode_text(corpus), trained.encode_text(corpus));
	assert_eq!(loaded.decode_text(&loaded.encode_text(corpus))?, corpus);
	fs::remove_file(&path).expect("test vocabulary should remain removable");

	fs::write(&path, "oa_bpe_v1\n1\n256 0\n").expect("invalid fixture should be writable");
	let before = loaded.merges().to_vec();
	let error = loaded
		.load(&path)
		.expect_err("forward merge dependency was accepted");
	assert_eq!(error.kind(), oa::ErrorKind::DataLoss);
	assert_eq!(loaded.merges(), before);
	fs::remove_file(path).expect("invalid fixture should remain removable");
	Ok(())
}
