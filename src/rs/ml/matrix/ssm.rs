//! Selective state-space Matrix operations.

use crate::{Error, Matrix, Result};

use crate::ml::{autograd, lowering::ssm as dispatch};

/// Geometry and numerical bounds for the fused Mamba-3 projection preprocess.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba3PreprocessConfig {
	/// Width of the z and x projection slices.
	pub inner_size: usize,
	/// Width of each B/C state row.
	pub state_size: usize,
	/// Number of selective-state heads.
	pub num_heads: usize,
	/// Number of rotary angle pairs.
	pub num_rope_angles: usize,
	/// Number of shared B/C groups.
	pub num_groups: usize,
	/// Number of B/C rows per group.
	pub mimo_rank: usize,
	/// Epsilon used by B/C RMS normalization.
	pub epsilon: f32,
	/// Inclusive lower bound for transformed dt.
	pub dt_min: f32,
	/// Inclusive upper bound for transformed dt.
	pub dt_max: f32,
	/// Positive floor for the magnitude of the negative A token.
	pub a_floor: f32,
}

impl Mamba3PreprocessConfig {
	pub(crate) fn identity(self, operation: &'static str) -> Result<u64> {
		let mut hash = 14_695_981_039_346_656_037_u64;
		for value in [
			u32::try_from(self.inner_size),
			u32::try_from(self.state_size),
			u32::try_from(self.num_heads),
			u32::try_from(self.num_rope_angles),
			u32::try_from(self.num_groups),
			u32::try_from(self.mimo_rank),
		]
		.into_iter()
		.chain([
			Ok(self.epsilon.to_bits()),
			Ok(self.dt_min.to_bits()),
			Ok(self.dt_max.to_bits()),
			Ok(self.a_floor.to_bits()),
		]) {
			let value = value.map_err(|_| {
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

/// Eight values produced by the fused Mamba-3 projection preprocess.
#[derive(Clone)]
pub struct Mamba3PreprocessResult {
	/// Value projection slice.
	pub x: Matrix,
	/// Gate projection slice.
	pub z: Matrix,
	/// RMS-normalized B rows.
	pub bh: Matrix,
	/// RMS-normalized C rows.
	pub ch: Matrix,
	/// Clamped positive time steps.
	pub dt: Matrix,
	/// Negative A-times-dt log decays.
	pub adt: Matrix,
	/// Raw trapezoidal gates.
	pub trap: Matrix,
	/// Raw rotary angular velocities.
	pub angle: Matrix,
}

impl Mamba3PreprocessResult {
	pub(crate) fn matrices(&self) -> [&Matrix; 8] {
		[
			&self.x,
			&self.z,
			&self.bh,
			&self.ch,
			&self.dt,
			&self.adt,
			&self.trap,
			&self.angle,
		]
	}
}

/// Input adjoints produced by the fused Mamba-3 projection preprocess.
pub struct Mamba3PreprocessBackward {
	/// Gradient of the packed projected input.
	pub projected: Matrix,
	/// Gradient of the per-head dt bias.
	pub dt_bias: Matrix,
}

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

/// Split and transform one packed Mamba-3 input projection.
///
/// `projected` has shape `[rows, projected_width]`, where
/// `projected_width = 2*inner_size + 2*num_groups*mimo_rank*state_size
/// + 3*num_heads + num_rope_angles`. `dt_bias` has one element per head.
///
/// The operation records one semantic node and one physical dispatch.
///
/// # Errors
///
/// Returns an error for invalid configuration, incompatible shape, dtype or
/// Engine ownership, arithmetic overflow, or failed runtime recording.
pub fn mamba3_preprocess(
	projected: &Matrix,
	dt_bias: &Matrix,
	config: Mamba3PreprocessConfig,
) -> Result<Mamba3PreprocessResult> {
	let result = dispatch::mamba3_preprocess(projected, dt_bias, config)?;
	autograd::record_mamba3_preprocess(projected, dt_bias, &result, config)?;
	Ok(result)
}

/// Compute the packed projection and dt-bias adjoints for Mamba-3 preprocess.
///
/// `output_gradients` follows the exact eight shapes returned by
/// [`mamba3_preprocess`]. Its dt-bias reduction is ordered by increasing row.
///
/// # Errors
///
/// Returns an error for invalid configuration, incompatible gradients, shape,
/// dtype or Engine ownership, arithmetic overflow, or failed runtime recording.
pub fn mamba3_preprocess_backward(
	projected: &Matrix,
	dt_bias: &Matrix,
	output_gradients: &Mamba3PreprocessResult,
	config: Mamba3PreprocessConfig,
) -> Result<Mamba3PreprocessBackward> {
	let result = dispatch::mamba3_preprocess_backward(projected, dt_bias, output_gradients, config)?;
	Ok(Mamba3PreprocessBackward {
		projected: result.projected,
		dt_bias: result.dt_bias,
	})
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

/// The fifteen explicit adjoints of [`mamba3_mimo`].
pub struct Mamba3MimoBackward {
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
	/// Gradient for the MIMO input projection.
	pub mimo_x: Matrix,
	/// Gradient for the MIMO gate projection.
	pub mimo_z: Matrix,
	/// Gradient for the MIMO output projection.
	pub mimo_o: Matrix,
	/// Gradient for the optional output-normalization weight.
	pub norm_weight: Matrix,
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
	let output = dispatch::mamba3_siso(c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d, config)?;
	autograd::record_mamba3_siso(
		[c, b, x, z, adt, dt, trap, angle, c_bias, b_bias, d],
		&output,
		config,
	)?;
	Ok(output)
}

/// Advance one Mamba-3 SISO token and update four explicit recurrent states.
///
/// Operand layouts are the length-one forms accepted by [`mamba3_siso`]. State
/// layouts are SSM `[B,H,P,N]`, cumulative angle `[B,H,A]`, previous key
/// `[B,H,N]`, and previous value `[B,H,P]`. The states are mutated only after
/// the complete operation has been validated and recorded.
///
/// # Errors
///
/// Returns an error unless `config.sequence_length == 1`, every operand and
/// state has the configured same-engine F32 shape, or recording fails.
#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor eleven operands plus four explicit recurrent-state buffers"
)]
pub fn mamba3_siso_step(
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
	ssm_state: &mut Matrix,
	angle_state: &mut Matrix,
	k_state: &mut Matrix,
	v_state: &mut Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	dispatch::mamba3_siso_step(
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
		ssm_state,
		angle_state,
		k_state,
		v_state,
		config,
	)
}

/// Run the shared-state Mamba-3 MIMO selective scan.
///
/// C/B use `[B,L,R*G,N]`; the three MIMO projections use `[H,R,P]`; the
/// output-normalization weight always uses `[H,P]` and is consumed only when
/// `config.has_output_norm` is true. The remaining layouts match
/// [`mamba3_siso`].
///
/// # Errors
///
/// Returns an error for mismatched geometry, rank outside `1..=8`, unsupported
/// dimensions, shape overflow, or failed runtime recording.
#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor fifteen-operand MIMO selective-scan contract"
)]
pub fn mamba3_mimo(
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
	mimo_x: &Matrix,
	mimo_z: &Matrix,
	mimo_o: &Matrix,
	norm_weight: &Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	let output = dispatch::mamba3_mimo(
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
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
		config,
	)?;
	autograd::record_mamba3_mimo(
		[
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
			mimo_x,
			mimo_z,
			mimo_o,
			norm_weight,
		],
		&output,
		config,
	)?;
	Ok(output)
}

/// Advance one Mamba-3 MIMO token and update four explicit recurrent states.
///
/// State layouts are SSM `[B,H,P,N]`, angle `[B,H,A]`, key `[B,H,R,N]`,
/// and value `[B,H,R,P]`.
///
/// # Errors
///
/// Returns an error unless `config.sequence_length == 1`, every operand and
/// state matches the configured same-engine F32 layout, or recording fails.
#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor fifteen operands plus four recurrent-state buffers"
)]
pub fn mamba3_mimo_step(
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
	mimo_x: &Matrix,
	mimo_z: &Matrix,
	mimo_o: &Matrix,
	norm_weight: &Matrix,
	ssm_state: &mut Matrix,
	angle_state: &mut Matrix,
	k_state: &mut Matrix,
	v_state: &mut Matrix,
	config: SsmConfig,
) -> Result<Matrix> {
	dispatch::mamba3_mimo_step(
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
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
		ssm_state,
		angle_state,
		k_state,
		v_state,
		config,
	)
}

/// Compute all fifteen explicit Mamba-3 MIMO adjoints.
///
/// The implementation records the donor's deterministic prepare, post,
/// reverse-core, inverse-rotation, and reduction stages as one semantic
/// operation.
///
/// # Errors
///
/// Returns an error for mismatched geometry, unsupported dimensions, shape
/// overflow, or failed runtime recording.
#[allow(
	clippy::too_many_arguments,
	reason = "preserves the donor upstream plus fifteen-operand MIMO adjoint contract"
)]
pub fn mamba3_mimo_backward(
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
	mimo_x: &Matrix,
	mimo_z: &Matrix,
	mimo_o: &Matrix,
	norm_weight: &Matrix,
	config: SsmConfig,
) -> Result<Mamba3MimoBackward> {
	let result = dispatch::mamba3_mimo_backward(
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
		mimo_x,
		mimo_z,
		mimo_o,
		norm_weight,
		config,
	)?;
	Ok(Mamba3MimoBackward {
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
		mimo_x: result.mimo_x,
		mimo_z: result.mimo_z,
		mimo_o: result.mimo_o,
		norm_weight: result.norm_weight,
	})
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
