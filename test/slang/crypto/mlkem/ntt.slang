// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mlkem_ntt;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_polynomial_components(uint3 tid : SV_DispatchThreadID) {
	uint input_base = tid.x * 512u, output_base = tid.x * 768u;
	uint left[256], right[256], restored[256];
	for (uint j = 0u; j < 256u; ++j) {
		left[j] = inputs[input_base + j];
		right[j] = inputs[input_base + 256u + j];
	}
	mlkem_ntt(left);
	mlkem_ntt(right);
	for (uint j = 0u; j < 256u; ++j) {
		outputs[output_base + j] = left[j];
		restored[j] = left[j];
	}
	mlkem_intt(restored);
	mlkem_multiply_ntts(left, right);
	mlkem_intt(left);
	for (uint j = 0u; j < 256u; ++j) {
		outputs[output_base + 256u + j] = restored[j];
		outputs[output_base + 512u + j] = left[j];
	}
}
