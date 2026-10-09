// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mldsa_ntt;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void polynomial_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 512u;
	uint out_base = tid.x * 1024u;
	uint a[256], b[256], c[256], restored[256];
	for (uint j = 0u; j < 256u; ++j) {
		a[j] = inputs[base + j];
		b[j] = inputs[base + 256u + j];
	}
	mldsa_ntt(a);
	mldsa_ntt(b);
	for (uint j = 0u; j < 256u; ++j) {
		outputs[out_base + j] = a[j];
		restored[j] = a[j];
	}
	mldsa_intt(restored);
	mldsa_poly_pointwise_ntt(a, b, c);
	mldsa_intt(c);
	mldsa_poly_scalar(a, 8192u);
	for (uint j = 0u; j < 256u; ++j) {
		outputs[out_base + 256u + j] = restored[j];
		outputs[out_base + 512u + j] = c[j];
		outputs[out_base + 768u + j] = a[j];
	}
}
