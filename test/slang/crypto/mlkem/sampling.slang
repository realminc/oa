// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mlkem_sampling;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_sampling_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 12u, result_base = tid.x * 512u;
	uint seed[8], matrix[256], noise[256];
	for (uint i = 0u; i < 8u; ++i) seed[i] = inputs[base + i];
	mlkem_sample_ntt(seed, inputs[base + 8u], inputs[base + 9u], matrix);
	mlkem_sample_noise(seed, inputs[base + 11u], inputs[base + 10u], noise);
	for (uint i = 0u; i < 256u; ++i) {
		outputs[result_base + i] = matrix[i];
		outputs[result_base + 256u + i] = noise[i];
	}
}

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_rejection_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 3u, result_base = tid.x * 257u;
	uint count = inputs[base + 2u], poly[256];
	for (uint i = 0u; i < 256u; ++i) poly[i] = 0xa5a5a5a5u;
	mlkem_accept_pair(inputs[base], inputs[base + 1u], count, poly);
	for (uint i = 0u; i < 256u; ++i) outputs[result_base + i] = poly[i];
	outputs[result_base + 256u] = count;
}
