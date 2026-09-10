use crate::{Matrix, Result};

use super::super::{node::Node, tape::record_node};

pub(in crate::ml) fn record_bmm(left: &Matrix, right: &Matrix, output: &Matrix) -> Result<()> {
	record_node(Node::Bmm {
		left: left.clone(),
		right: right.clone(),
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_bmm_nt(left: &Matrix, right: &Matrix, output: &Matrix) -> Result<()> {
	record_node(Node::BmmNt {
		left: left.clone(),
		right: right.clone(),
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_bmm_tn(left: &Matrix, right: &Matrix, output: &Matrix) -> Result<()> {
	record_node(Node::BmmTn {
		left: left.clone(),
		right: right.clone(),
		output_id: output.value_id(),
	})
}

pub(in crate::ml) fn record_split_heads(
	input: &Matrix,
	output: &Matrix,
	batch: usize,
	sequence_length: usize,
	num_heads: usize,
) -> Result<()> {
	record_node(Node::SplitHeads {
		input: input.clone(),
		output_id: output.value_id(),
		batch,
		sequence_length,
		num_heads,
	})
}

pub(in crate::ml) fn record_merge_heads(
	input: &Matrix,
	output: &Matrix,
	batch: usize,
	sequence_length: usize,
	num_heads: usize,
) -> Result<()> {
	record_node(Node::MergeHeads {
		input: input.clone(),
		output_id: output.value_id(),
		batch,
		sequence_length,
		num_heads,
	})
}

pub(in crate::ml) fn record_softmax_scaled_masked(
	input: &Matrix,
	output: &Matrix,
	scale: f32,
) -> Result<()> {
	record_node(Node::SoftmaxScaledMasked {
		input: input.clone(),
		output: output.clone(),
		output_id: output.value_id(),
		scale,
	})
}

pub(in crate::ml) fn record_scaled_dot_product_attention(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	output: &Matrix,
	probabilities: Matrix,
	scale: f32,
) -> Result<()> {
	record_node(Node::ScaledDotProductAttention {
		query: query.clone(),
		key: key.clone(),
		value: value.clone(),
		probabilities,
		output_id: output.value_id(),
		scale,
	})
}

pub(in crate::ml) fn record_flash_attention(
	query: &Matrix,
	key: &Matrix,
	value: &Matrix,
	output: &Matrix,
	log_sum_exp: Matrix,
	scale: f32,
) -> Result<()> {
	record_node(Node::FlashAttention {
		query: query.clone(),
		key: key.clone(),
		value: value.clone(),
		output: output.clone(),
		log_sum_exp,
		output_id: output.value_id(),
		scale,
	})
}
