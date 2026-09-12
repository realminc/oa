//! Selective state-space Matrix operations.

use crate::{Error, Matrix, Result};

use crate::ml::{autograd, lowering::ssm as dispatch};

/// Explicit tensor geometry and optional paths for a selective-state scan.
///
/// A zero num_groups means one Q/K group per head, matching OA C++.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SsmConfig {
	/// Batch size.
	pub batch_size: usize,
	/// Sequence length.
	pub sequence_length: usize,
	/// Number of value heads.
	pub num_heads: usize,
	/// Number of shared Q/K groups, or zero for num_heads.
	pub num_groups: usize,
	/// Value width per head.
	pub head_dim: usize,
	/// Selective recurrent state width.
	pub state_size: usize,
	/// Number of rotary pairs in the state axis.
	pub num_rope_angles: usize,
	/// MIMO rank; SISO callers use one.
	pub mimo_rank: usize,
	/// Apply the SiLU z gate.
	pub has_z: bool,
	/// Apply the learned D skip.
	pub has_d: bool,
	/// Apply MIMO output normalization; SISO requires false.
	pub has_output_norm: bool,
}

impl SsmConfig {
	pub(crate) fn identity(self, operation: &'static str) -> Result<u64> {
		let mut hash = 14_695_981_039_346_656_037_u64;
		for value in [
			self.batch_size,
			self.sequence_length,
			self.num_heads,
			self.num_groups,
			self.head_dim,
			self.state_size,
			self.num_rope_angles,
			self.mimo_rank,
			usize::from(self.has_z),
			usize::from(self.has_d),
			usize::from(self.has_output_norm),
		] {
			let value = u32::try_from(value).map_err(|_| {
				Error::invalid_argument(format!(
					"{operation} configuration value exceeds the stable u32 identity ABI"
				))
			})?;
			for byte in value.to_le_bytes() {
				hash = (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211);
			}
		}
		Ok(hash)
	}
}

/// The eleven explicit adjoints of mamba3_siso.
pub struct SsmBackward {
	/// Gradient for C.
	pub c: Matrix,
	/// Gradient for B.
	pub b: Matrix,
	/// Gradient for x.
	pub x: Matrix,
	/// Gradient for z.
	pub z: Matrix,
	/// Gradient for A times dt.
	pub adt: Matrix,
	/// Gradient for dt.
	pub dt: Matrix,
	/// Gradient for the raw trapezoidal gate.
	pub trap: Matrix,
	/// Gradient for rotary angular velocity.
	pub angle: Matrix,
	/// Gradient for the C bias.
	pub c_bias: Matrix,
	/// Gradient for the B bias.
	pub b_bias: Matrix,
	/// Gradient for the D skip.
	pub d: Matrix,
}

/// Run the Mamba-3 SISO selective-state recurrence.
///
/// Operand layouts are C/B `[B,L,G,N]`, x/z/output `[B,L,H,P]`,
/// A-times-dt/dt/trap `[B,L,H]`, angle `[B,L,A]`, C/B bias `[H,N]`, and D
/// `[H]`. All operands are same-engine F32 matrices.
///
/// # Errors
///
/// Returns an error for mismatched geometry, unsupported dimensions, shape
/// overflow, or failed runtime recording.
#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor eleven-operand selective-scan contract"
)]
pub fn mamba3_siso(
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	let output =
		dispatch::mamba3_siso(c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d, config)?;
	autograd::record_mamba3_siso(
		[c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d],
		&output,
		config,
	)?;
	Ok(output)
}

/// Compute all explicit Mamba-3 SISO adjoints.
///
/// The initial OARS physical route is the donor short-training route and
/// accepts L <= 16, P <= 16, N <= 32, and A <= 8.
///
/// # Errors
///
/// Returns an error for mismatched inputs, a shape outside the admitted short
/// route, overflow, or failed runtime recording.
#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor upstream plus eleven-operand adjoint contract"
)]
pub fn mamba3_siso_backward(
	output_gradient: &Matrix,
	c: &Matrix,
	b: &Matrix,
	x: &Matrix,
	z: &Matrix,
	adt: &Matrix,
	dt: &Matrix,
	trap: &Matrix,
	angle: &Matrix,
	c_bias: &Matrix,
	b_bias: &Matrix,
	d: &Matrix,
	config: SsmConfig,
) -> Result<SsmBackward> {
	let result = dispatch::mamba3_siso_backward(
		output_gradient,
		c,
		b,
		x,
		z,
		adt,
		dt,
		trap,
		angle,
		c_bias,
		b_bias,
		d,
		config,
	)?;
	Ok(SsmBackward {
		c: result.c,
		b: result.b,
		x: result.x,
		z: result.z,
		adt: result.adt,
		dt: result.dt,
		trap: result.trap,
		angle: result.angle,
		c_bias: result.c_bias,
		b_bias: result.b_bias,
		d: result.d,
	})
}
