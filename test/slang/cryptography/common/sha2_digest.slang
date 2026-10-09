// Test-only transport imports production helpers; each entry owns its output words.
import sha2_u32;
RWByteAddressBuffer inputs;
RWStructuredBuffer<uint> outputs;

[numthreads(1, 1, 1)]
void sha2_u32_components(uint3 id : SV_DispatchThreadID) {
	if (id.x != 0u) return;
	uint digest[8];
	for (uint i = 0u; i < 8u; ++i) digest[i] = 0xa5a5a5a5u;
	uint bits = inputs.Load(0u);
	bool valid = false;
	if (bits == 224u) {
		uint short_digest[7];
		for (uint i = 0u; i < 7u; ++i) short_digest[i] = 0xa5a5a5a5u;
		valid = sha2_224(inputs, inputs.Load(4u), inputs.Load(8u), short_digest);
		for (uint i = 0u; i < 7u; ++i) digest[i] = short_digest[i];
	} else if (bits == 256u) {
		valid = sha2_256(inputs, inputs.Load(4u), inputs.Load(8u), digest);
	} else {
		valid = sha2_u32_digest(inputs, inputs.Load(4u), inputs.Load(8u), bits, digest);
	}
	outputs[0] = valid ? 1u : 0u;
	for (uint i = 0u; i < 8u; ++i) outputs[1u + i] = digest[i];
}

[numthreads(1, 1, 1)]
void sha2_u32_length_components(uint3 id : SV_DispatchThreadID) {
	if (id.x != 0u) return;
	uint2 bits = sha2_u32_bit_length(inputs.Load(0u));
	outputs[0] = bits.x;
	outputs[1] = bits.y;
}
