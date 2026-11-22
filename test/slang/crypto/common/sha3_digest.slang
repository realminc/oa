// Test-only transport; production SHA-3 formulas and permutation are imported.
import sha3;

RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[numthreads(1, 1, 1)]
void sha3_components(uint3 id : SV_DispatchThreadID) {
	if (id.x != 0u) return;
	uint bits = inputs.Load(0u);
	uint base = inputs.Load(4u);
	uint length = inputs.Load(8u);
	bool valid = false;
	if (bits == 224u) {
		uint digest[7];
		for (uint i = 0u; i < 7u; ++i) digest[i] = 0xa5a5a5a5u;
		valid = sha3_224(inputs, base, length, digest);
		for (uint i = 0u; i < 7u; ++i) outputs[1u + i] = digest[i];
	} else if (bits == 256u) {
		uint2 digest[4];
		for (uint i = 0u; i < 4u; ++i) digest[i] = uint2(0xa5a5a5a5u, 0xa5a5a5a5u);
		valid = sha3_256(inputs, base, length, digest);
		for (uint i = 0u; i < 4u; ++i) {
			outputs[1u + i * 2u] = digest[i].x;
			outputs[2u + i * 2u] = digest[i].y;
		}
	} else if (bits == 384u) {
		uint2 digest[6];
		for (uint i = 0u; i < 6u; ++i) digest[i] = uint2(0xa5a5a5a5u, 0xa5a5a5a5u);
		valid = sha3_384(inputs, base, length, digest);
		for (uint i = 0u; i < 6u; ++i) {
			outputs[1u + i * 2u] = digest[i].x;
			outputs[2u + i * 2u] = digest[i].y;
		}
	} else if (bits == 512u) {
		uint2 digest[8];
		for (uint i = 0u; i < 8u; ++i) digest[i] = uint2(0xa5a5a5a5u, 0xa5a5a5a5u);
		valid = sha3_512(inputs, base, length, digest);
		for (uint i = 0u; i < 8u; ++i) {
			outputs[1u + i * 2u] = digest[i].x;
			outputs[2u + i * 2u] = digest[i].y;
		}
	}
	outputs[0] = valid ? 1u : 0u;
}

[numthreads(1, 1, 1)]
void sha3_range_components(uint3 id : SV_DispatchThreadID) {
	if (id.x != 0u) return;
	uint2 state[25];
	for (uint i = 0u; i < 25u; ++i) state[i] = uint2(0xa5a5a5a5u, 0xa5a5a5a5u);
	bool valid = sha3_absorb(state, inputs, inputs.Load(4u), inputs.Load(8u), inputs.Load(0u));
	outputs[0] = valid ? 1u : 0u;
	for (uint i = 0u; i < 25u; ++i) {
		outputs[1u + i * 2u] = state[i].x;
		outputs[2u + i * 2u] = state[i].y;
	}
}
