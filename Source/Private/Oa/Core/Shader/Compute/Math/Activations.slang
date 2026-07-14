// oa::activations — Reusable activation functions for Slang shaders
//
// Import: `import Activations;` (slangc -I to Core/Math or installed lib/Oa/Core/shaders/Math)
// Pure math — no bindings, no push constants, no entry points.
// ML and crypto shaders import this instead of duplicating activation math.

module Activations;

public float Sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

public float Silu(float x) {
	return x * Sigmoid(x);
}

// d(silu)/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
public float SiluGrad(float x) {
	float sig = Sigmoid(x);
	return sig * (1.0 + x * (1.0 - sig));
}

public float Gelu(float x) {
	float cdf = 0.5 * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
	return x * cdf;
}

public float Relu(float x) {
	return max(0.0, x);
}

public float Swiglu(float gate, float up) {
	return Silu(gate) * up;
}
