//! ML-specialized stateless Matrix operations.
//!
//! This module corresponds to the ML-owned extension of C++ `oa::FnMatrix`.
//! General numerical operations remain in [`crate::matrix`], losses remain in
//! [`crate::ml::loss`], and stateful layers remain in [`crate::ml::nn`].

mod activation;
mod attention;
mod conv;
mod moe;
mod norm;
mod pool;
mod position;
mod recurrent;
mod ssm;
mod upsample;
mod vq;

pub(in crate::ml) use activation::silu_mul_backward;
pub use activation::{
	elu, gelu, leaky_relu, mish, relu, sigmoid, silu, silu_mul, softplus, swiglu, tanh,
};
pub use attention::{
	bmm, bmm_nt, bmm_tn, flash_attention_causal, merge_heads, scaled_dot_product_attention,
	scaled_dot_product_attention_causal, softmax_scaled_masked, split_heads,
};
pub use conv::{conv_1d, conv_2d, conv_transpose_1d, conv_transpose_2d};
pub(in crate::ml) use conv::{
	conv_1d_backward, conv_1d_parameterized, conv_2d_backward, conv_2d_parameterized,
	conv_transpose_1d_backward, conv_transpose_1d_parameterized, conv_transpose_2d_backward,
	conv_transpose_2d_parameterized,
};
pub use moe::{grouped_gemm_m, grouped_linear_m, moe_combine, moe_gather, moe_route_weights};
pub(in crate::ml) use moe::{
	grouped_gemm_m_backward, grouped_linear_m_backward, grouped_linear_m_parameterized,
	moe_combine_backward, moe_gather_backward, moe_route_weights_backward,
};
pub use norm::{
	BatchNorm2dResult, ChannelNormBackward, RmsNormGatedBackward, batch_norm_2d,
	batch_norm_2d_with_stats, channel_norm, channel_norm_backward, channel_norm_relu,
	channel_norm_relu_backward, rms_norm, rms_norm_gated, rms_norm_gated_backward,
};
pub(in crate::ml) use norm::{
	batch_norm_2d_backward, batch_norm_2d_forward, batch_norm_2d_running_update,
	batch_norm_2d_with_stats_forward,
};
pub use pool::{MaxPool2dResult, adaptive_avg_pool_2d, avg_pool_2d, max_pool_2d};
pub(in crate::ml) use pool::{
	adaptive_avg_pool_2d_backward, avg_pool_2d_backward, max_pool_2d_backward,
};
pub use position::rope;
pub use recurrent::{gru_cell, gru_scan, rnn_cell, rnn_scan};
pub(in crate::ml) use recurrent::{
	gru_cell_backward, gru_cell_parameterized, gru_scan_backward, gru_scan_parameterized,
	rnn_cell_backward, rnn_cell_parameterized, rnn_scan_backward, rnn_scan_parameterized,
};
pub use ssm::{
	Mamba3MimoBackward, Mamba3PreprocessBackward, Mamba3PreprocessConfig, Mamba3PreprocessResult,
	SsmBackward, SsmConfig, mamba3_mimo, mamba3_mimo_backward, mamba3_mimo_step, mamba3_preprocess,
	mamba3_preprocess_backward, mamba3_siso, mamba3_siso_backward, mamba3_siso_step,
};
pub(in crate::ml) use upsample::upsample_2d_backward;
pub use upsample::{UpsampleMode, upsample_2d};
pub use vq::{VqAssignResult, VqEmaState, detach, vq_assign, vq_ema_update, vq_lookup};
