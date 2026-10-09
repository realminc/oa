//! ML-specialized stateless Matrix operations.
//!
//! This module corresponds to the ML-owned extension of C++ `oa::FnMatrix`.
//! General numerical operations remain in [`crate::matrix`], losses remain in
//! [`crate::ml::loss`], and stateful layers remain in [`crate::ml::nn`].

#[path = "matrix/activation.gen.rs"]
mod activation;
mod attention;
mod batch_norm;
mod channel_norm;
mod conv;
mod embedding;
mod gru;
mod layer_norm;
mod linear;
mod mamba3;
mod moe;
mod pool;
mod recurrent;
mod rms_norm;
mod rnn;
#[path = "matrix/rope.gen.rs"]
mod rope;
#[path = "matrix/swiglu.gen.rs"]
mod swiglu;
#[path = "matrix/upsample.gen.rs"]
mod upsample;
mod vq;

pub use activation::{elu, gelu, leaky_relu, mish, relu, sigmoid, silu, softplus, tanh};
pub(in crate::ml) use activation::{
	elu_backward, gelu_backward, leaky_relu_backward, mish_backward, relu_backward, sigmoid_backward,
	silu_backward, softplus_backward, tanh_backward,
};
pub use attention::{
	bmm, bmm_nt, bmm_tn, flash_attention_causal, merge_heads, scaled_dot_product_attention,
	scaled_dot_product_attention_causal, softmax_scaled_masked, split_heads,
};
pub(in crate::ml) use attention::{
	flash_attention_causal_backward, merge_heads_dispatch, softmax_scaled_masked_backward,
	split_heads_dispatch,
};
pub use batch_norm::{BatchNorm2dResult, batch_norm_2d, batch_norm_2d_with_stats};
pub(in crate::ml) use batch_norm::{
	batch_norm_2d_backward, batch_norm_2d_forward, batch_norm_2d_running_update,
	batch_norm_2d_with_stats_forward,
};
pub use channel_norm::{
	ChannelNormBackward, channel_norm, channel_norm_backward, channel_norm_relu,
	channel_norm_relu_backward,
};
pub(in crate::ml) use channel_norm::{channel_norm_backward_saved, channel_norm_forward};
pub use conv::{conv_1d, conv_2d, conv_transpose_1d, conv_transpose_2d};
pub(in crate::ml) use conv::{
	conv_1d_backward, conv_1d_parameterized, conv_2d_backward, conv_2d_parameterized,
	conv_transpose_1d_backward, conv_transpose_1d_parameterized, conv_transpose_2d_backward,
	conv_transpose_2d_parameterized,
};
pub(in crate::ml) use embedding::{embedding, embedding_backward};
pub use gru::{gru_cell, gru_scan};
pub(in crate::ml) use gru::{
	gru_cell_backward, gru_cell_parameterized, gru_scan_backward, gru_scan_parameterized,
};
pub(in crate::ml) use layer_norm::{layer_norm, layer_norm_backward};
pub(in crate::ml) use linear::{linear, linear_backward};
pub use mamba3::{
	Mamba3MimoBackward, Mamba3PreprocessBackward, Mamba3PreprocessConfig, Mamba3PreprocessResult,
	SsmBackward, SsmConfig, mamba3_mimo, mamba3_mimo_backward, mamba3_mimo_step, mamba3_preprocess,
	mamba3_preprocess_backward, mamba3_siso, mamba3_siso_backward, mamba3_siso_step,
};
pub use moe::{grouped_gemm_m, grouped_linear_m, moe_combine, moe_gather, moe_route_weights};
pub(in crate::ml) use moe::{
	grouped_gemm_m_backward, grouped_linear_m_backward, grouped_linear_m_parameterized,
	moe_combine_backward, moe_gather_backward, moe_route_weights_backward,
};
pub use pool::{MaxPool2dResult, adaptive_avg_pool_2d, avg_pool_2d, max_pool_2d};
pub(in crate::ml) use pool::{
	adaptive_avg_pool_2d_backward, avg_pool_2d_backward, max_pool_2d_backward,
};
pub use rms_norm::{RmsNormGatedBackward, rms_norm, rms_norm_gated, rms_norm_gated_backward};
pub(in crate::ml) use rms_norm::{rms_norm_backward, rms_norm_forward, rms_norm_gated_forward};
pub use rnn::{rnn_cell, rnn_scan};
pub(in crate::ml) use rnn::{
	rnn_cell_backward, rnn_cell_parameterized, rnn_scan_backward, rnn_scan_parameterized,
};
pub use rope::rope;
pub(in crate::ml) use rope::rope_backward;
pub(in crate::ml) use upsample::upsample_2d_backward;
pub use upsample::{UpsampleMode, upsample_2d};
pub use vq::{VqAssignResult, VqEmaState, detach, vq_assign, vq_ema_update, vq_lookup};

pub use swiglu::{silu_mul, swiglu};
pub(in crate::ml) use swiglu::{silu_mul_backward, swiglu_backward};
