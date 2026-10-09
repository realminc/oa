// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mldsa_sampling;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void sampling_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 22u;
	uint challenge[12], rho[8], c[256], a[256];
	for (uint i = 0u; i < 12u; ++i) challenge[i] = inputs[base + i];
	for (uint i = 0u; i < 8u; ++i) rho[i] = inputs[base + 12u + i];
	mldsa65_sample_in_ball(challenge, c);
	mldsa_expand_a_poly(rho, inputs[base + 20u], inputs[base + 21u], a);
	for (uint i = 0u; i < 256u; ++i) {
		outputs[tid.x * 512u + i] = c[i];
		outputs[tid.x * 512u + 256u + i] = a[i];
	}
}
