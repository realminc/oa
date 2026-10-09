// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import shake;

RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[numthreads(1, 1, 1)]
void streaming_components(uint3 id : SV_DispatchThreadID) {
	uint base = id.x * 2056u;
	uint first_length = inputs.Load(base);
	uint second_length = inputs.Load(base + 4u);
	PqcShake256 sponge;
	pqc_shake256_init(sponge);
	pqc_shake256_absorb_buffer(sponge, inputs, base + 9u, first_length);
	pqc_shake256_absorb_buffer(sponge, inputs, base + 1035u, second_length);
	uint2 digest[8];
	pqc_shake256_finish(sponge, digest);
	for (uint i = 0u; i < 8u; ++i) {
		outputs[id.x * 16u + i * 2u] = digest[i].x;
		outputs[id.x * 16u + i * 2u + 1u] = digest[i].y;
	}
}
