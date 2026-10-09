// Reusable Core activation formulas shared by executable operation kernels.
// Donor: source/cpp/lib/oa/core/shader/compute/math/activations.slang

module activations;

public float sigmoid(float x) {
	return 1.0f / (1.0f + exp(-x));
}

public float silu(float x) {
	return x * sigmoid(x);
}

public float silu_grad(float x) {
	float sig = sigmoid(x);
	return sig * (1.0f + x * (1.0f - sig));
}

public float swiglu(float gate, float up) {
	return silu(gate) * up;
}
