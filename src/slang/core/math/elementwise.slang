// Shared scalar formulas for source kernels, tape interpretation and specialization.
// Mechanical extraction from matrix/elemwise/fused_chain.slang. Its OA donor
// formulas and operation ordering are retained; this module owns no resources.
module elementwise;
import activations;

[ForceInline]
public float eval_element(uint opcode, float a, float b, float extra) {
	if (opcode == 10u) return a + b;
	if (opcode == 11u) return a - b;
	if (opcode == 12u) return a * b;
	if (opcode == 13u) return a / b;
	if (opcode == 14u) return pow(a, b);
	if (opcode == 15u) return min(a, b);
	if (opcode == 16u) return max(a, b);
	if (opcode == 20u) return -a;
	if (opcode == 21u) return 1.0f / a;
	if (opcode == 22u) return abs(a);
	if (opcode == 23u) return exp(a);
	if (opcode == 24u) return log(a);
	if (opcode == 25u) return sqrt(a);
	if (opcode == 26u) return sin(a);
	if (opcode == 27u) return cos(a);
	if (opcode == 28u) return tanh(a);
	if (opcode == 30u) return 1.0f / (1.0f + exp(-a));
	if (opcode == 31u) return max(a, 0.0f);
	if (opcode == 32u) {
		float inner = 0.7978845608028654f * (a + 0.044715f * a * a * a);
		return 0.5f * a * (1.0f + tanh(inner));
	}
	if (opcode == 33u) return a / (1.0f + exp(-a));
	if (opcode == 40u) return a > 0.0f ? b : 0.0f;
	if (opcode == 41u) {
		float x2 = a * a;
		float inner = 0.7978845608028654f * (a + 0.044715f * x2 * a);
		float t = tanh(inner);
		float derivative = 0.5f * (1.0f + t) + 0.5f * a * (1.0f - t*t) * 0.7978845608028654f * (1.0f + 0.134145f * x2);
		return b * derivative;
	}
	if (opcode == 42u) return b * silu_grad(a);
	if (opcode == 43u) return b * (1.0f - a*a);
	if (opcode == 44u) return b * a * (1.0f - a);
	if (opcode == 45u) return b * (1.0f - exp(-a));
	if (opcode == 46u) {
		float softplus = a > 20.0f ? a : log(1.0f + exp(a));
		float t = tanh(softplus);
		return b * (t + a * sigmoid(a) * (1.0f - t*t));
	}
	if (opcode == 47u) return b * (a > 0.0f ? 1.0f : a + extra);
	if (opcode == 48u) return a > 0.0f ? b : b * extra;
	return a;
}
