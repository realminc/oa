//! Private autograd attachments for ML-owned Matrix operation families.

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
#[path = "matrix/pool.gen.rs"]
mod pool;
mod rms_norm;
mod rnn;
#[path = "matrix/rope.gen.rs"]
mod rope;
#[path = "matrix/upsample.gen.rs"]
mod upsample;

#[path = "matrix/activation.gen.rs"]
mod activation;
pub(in crate::ml) use activation::{
	record_elu, record_gelu, record_leaky_relu, record_mish, record_relu, record_sigmoid,
	record_silu, record_softplus, record_tanh,
};
#[path = "matrix/swiglu.gen.rs"]
mod swiglu;
pub(in crate::ml) use swiglu::{record_silu_mul, record_swiglu};

pub(in crate::ml) use attention::{
	record_bmm, record_bmm_nt, record_bmm_tn, record_flash_attention, record_merge_heads,
	record_scaled_dot_product_attention, record_softmax_scaled_masked, record_split_heads,
};
pub(in crate::ml) use batch_norm::record_batch_norm_2d;
pub(in crate::ml) use channel_norm::record_channel_norm;
pub(in crate::ml) use conv::{
	record_conv_1d, record_conv_2d, record_conv_transpose_1d, record_conv_transpose_2d,
};
pub(in crate::ml) use embedding::record_embedding;
pub(in crate::ml) use gru::{record_gru_cell, record_gru_scan};
pub(in crate::ml) use layer_norm::record_layer_norm;
pub(in crate::ml) use linear::record_linear;
pub(in crate::ml) use mamba3::{record_mamba3_mimo, record_mamba3_preprocess, record_mamba3_siso};
pub(in crate::ml) use moe::{
	record_grouped_gemm_m, record_grouped_linear_m, record_moe_combine, record_moe_gather,
	record_moe_route_weights,
};
pub(in crate::ml) use pool::{record_adaptive_avg_pool_2d, record_avg_pool_2d, record_max_pool_2d};
pub(in crate::ml) use rms_norm::{record_rms_norm, record_rms_norm_gated};
pub(in crate::ml) use rnn::{record_rnn_cell, record_rnn_scan};
pub(in crate::ml) use rope::record_rope;
pub(in crate::ml) use upsample::record_upsample_2d;
