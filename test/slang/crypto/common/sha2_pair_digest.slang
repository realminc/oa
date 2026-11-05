// Test transport imports production arithmetic and digest helpers.
import sha2_u32_pair;
RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[numthreads(1, 1, 1)]
void sha2_u32_pair_components(uint3 id : SV_DispatchThreadID) {
	if (id.x != 0u) return;
	uint bits = inputs.Load(0u);
	uint digest[16];
	for (uint i = 0u; i < 16u; ++i) digest[i] = 0xa5a5a5a5u;
	bool valid = false;
	if (bits == 224u) {
		uint short_digest[7];
		for (uint i = 0u; i < 7u; ++i) short_digest[i] = 0xa5a5a5a5u;
		valid = sha2_512_224(inputs, inputs.Load(4u), inputs.Load(8u), short_digest);
		for (uint i = 0u; i < 7u; ++i) digest[i] = short_digest[i];
	} else if (bits == 256u) {
		uint short_digest[8];
		for (uint i = 0u; i < 8u; ++i) short_digest[i] = 0xa5a5a5a5u;
		valid = sha2_512_256(inputs, inputs.Load(4u), inputs.Load(8u), short_digest);
		for (uint i = 0u; i < 8u; ++i) digest[i] = short_digest[i];
	} else if (bits == 384u) {
		uint short_digest[12];
		for (uint i = 0u; i < 12u; ++i) short_digest[i] = 0xa5a5a5a5u;
		valid = sha2_384(inputs, inputs.Load(4u), inputs.Load(8u), short_digest);
		for (uint i = 0u; i < 12u; ++i) digest[i] = short_digest[i];
	} else if (bits == 512u) {
		uint short_digest[16];
		for (uint i = 0u; i < 16u; ++i) short_digest[i] = 0xa5a5a5a5u;
		valid = sha2_512(inputs, inputs.Load(4u), inputs.Load(8u), short_digest);
		for (uint i = 0u; i < 16u; ++i) digest[i] = short_digest[i];
	} else {
		valid = sha2_u32_pair_digest(inputs, inputs.Load(4u), inputs.Load(8u), bits, digest);
	}
	outputs[0] = valid ? 1u : 0u;
	for (uint i = 0u; i < 16u; ++i) outputs[1u + i] = digest[i];
}

[numthreads(1, 1, 1)]
void sha2_u32_pair_arithmetic_components(uint3 id : SV_DispatchThreadID) {
	if (id.x != 0u) return;
	uint2 left = uint2(inputs.Load(0u), inputs.Load(4u));
	uint2 right = uint2(inputs.Load(8u), inputs.Load(12u));
	uint shift = inputs.Load(16u);
	uint2 sum = sha2_u32_pair_add(left, right);
	uint2 rotated = sha2_u32_pair_rotr(left, shift);
	uint2 shifted = sha2_u32_pair_shr(left, shift);
	outputs[0] = sum.x; outputs[1] = sum.y;
	outputs[2] = rotated.x; outputs[3] = rotated.y;
	outputs[4] = shifted.x; outputs[5] = shifted.y;
}
