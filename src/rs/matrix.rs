//! Matrix values and stateless numerical operations.

mod alloc;
mod broadcast;
pub(crate) use broadcast::sum_to_shape;
pub(crate) mod autograd;
#[path = "matrix/blas.gen.rs"]
mod blas;
#[path = "matrix/elemwise.gen.rs"]
mod elemwise;
mod index;
#[allow(clippy::module_inception)]
mod matrix;
#[path = "matrix/reduce.gen.rs"]
mod reduce;
mod rng;
mod view;

pub use matrix::Matrix;
pub(crate) use matrix::MatrixSemantic;

pub use alloc::{from_f32, from_f32_on, full, ones};
pub use blas::mat_mul_nt;
pub(crate) use blas::mat_mul_nt_backward;
pub use elemwise::{
	abs, add, add_scalar, clamp_max, clamp_min, copy, cos, div, div_scalar, exp, log, mul, neg, pow,
	reciprocal, scale, sin, sqrt, sub, sub_scalar,
};
pub(crate) use elemwise::{
	abs_backward, add_backward, clamp_max_backward, clamp_min_backward, copy_backward, div_backward,
	exp_backward, log_backward, mul_backward, reciprocal_backward, scale_backward, sqrt_backward,
	sub_backward,
};
pub use index::{
	MoeExpertPlan, TopKResult, concat, equal, gather, gather_last_dim, moe_expert_plan,
	moe_routing_bias_update, repeat_interleave, slice, top_k, top_k_mask, transpose,
};
pub(crate) use index::{
	concat_backward, gather_backward, gather_last_dim_backward, repeat_interleave_backward,
	slice_backward, transpose_backward,
};
pub use reduce::{
	categorical_accuracy_count, log_softmax, masked_categorical_accuracy_count, softmax, sum,
};
pub(crate) use reduce::{log_softmax_backward, softmax_backward, sum_backward};
pub(crate) use rng::dropout_backward;
pub use rng::{dropout, philox_normal, philox_uniform, sample_logits, set_rng_seed};
pub use view::reshape;
pub(crate) use view::{reshape_backward, reshape_semantic_output};
