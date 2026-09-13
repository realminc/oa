// Reusable FP32 formulas for selective state-space kernels.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/math/ssmMath.slang

module ssm_math;

[ForceInline]
public float ssm_softplus(float value) {
	return max(value, 0.0f) + log(1.0f + exp(-abs(value)));
}

[ForceInline]
public float ssm_softplus_clamp(float value, float minimum, float maximum) {
	return clamp(ssm_softplus(value), minimum, maximum);
}

[ForceInline]
public float ssm_heavy_tail(float value) {
	return value >= 0.0f ? 1.0f + value : 1.0f / (1.0f - value);
}

[ForceInline]
public float ssm_a_token(float heavy_tail, float floor) {
	return min(-heavy_tail, -floor);
}

[ForceInline]
public float ssm_adt(float raw_a, float dt, float floor) {
	return ssm_a_token(ssm_heavy_tail(raw_a), floor) * dt;
}

[ForceInline]
public float ssm_rms_norm(float value, float inverse_rms) {
	return value * inverse_rms;
}

[ForceInline]
public float ssm_rms_norm_backward(
	float value,
	float output_gradient,
	float inverse_rms,
	float dot,
	float width
) {
	return inverse_rms * (
		output_gradient - value * dot * inverse_rms * inverse_rms / width);
}
