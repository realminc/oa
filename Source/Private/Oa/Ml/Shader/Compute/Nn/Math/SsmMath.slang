// Oa ML/NN SSM math module — reusable helpers for Mamba-3 / Empyrealm shaders.
//
// Import: `import SsmMath;` (add -I to Source/Private/Oa/Ml/Shader/Compute/Nn/Math
// and the umbrella Source/Public/Oa/Ml/Nn).
//
// Pure math — no bindings, no push constants, no entry points, no groupshared.
// Helpers operate on scalar values so they can cross Slang module boundaries safely.

module SsmMath;

// Numerically stable softplus: log(1 + exp(x))
public float SsmSoftplus(float x) {
	return max(x, 0.0f) + log(1.0f + exp(-abs(x)));
}

// Mamba-3/Empyrealm DT transform: softplus(x) clamped to [dt_min, dt_max]
public float SsmSoftplusClamp(float x, float dt_min, float dt_max) {
	return clamp(SsmSoftplus(x), dt_min, dt_max);
}

// Heavy-tail activation used in Empyrealm / Mamba-3 A·dt transform.
// Always positive and C1 at 0.
//   h(x) = 1 + x        if x >= 0
//   h(x) = 1 / (1 - x)  if x <  0
public float SsmHeavyTail(float x) {
	return (x >= 0.0f) ? (1.0f + x) : (1.0f / (1.0f - x));
}

// aTok = min(-h, -afloor). Always negative.
public float SsmEmpyrealmA(float h, float afloor) {
	return min(-h, -afloor);
}

// Full ADT = aTok * dt, where aTok is computed from dd_A.
public float SsmAdtFromDdA(float ddA, float dt, float afloor) {
	float h = SsmHeavyTail(ddA);
	float aTok = SsmEmpyrealmA(h, afloor);
	return aTok * dt;
}

// RMSNorm forward without learnable weight (weight = 1).
// Returns the normalized value. Caller must compute invRms first.
public float SsmRmsNormForwardNoWeight(float x, float invRms) {
	return x * invRms;
}

// RMSNorm backward without learnable weight.
// dx = inv_rms * (dy - x * c * inv_rms^2 / n), where c = sum_i dy_i * x_i.
public float SsmRmsNormBackwardNoWeight(float x, float dy, float invRms, float c, float n) {
	return invRms * (dy - x * c * invRms * invRms / n);
}
