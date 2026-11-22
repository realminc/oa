// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mlkem_packing;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_packing_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 641u;
	uint bits = inputs[base], poly[256], encoded[384], raw[384], decoded[256];
	for (uint i = 0u; i < 256u; ++i) poly[i] = inputs[base + 1u + i];
	for (uint i = 0u; i < 384u; ++i) {
		encoded[i] = 0xa5u;
		raw[i] = inputs[base + 257u + i];
	}
	mlkem_byte_encode(poly, bits, encoded);
	bool canonical = mlkem_byte_decode(raw, bits, decoded);
	for (uint i = 0u; i < 384u; ++i) outputs[base + i] = encoded[i];
	for (uint i = 0u; i < 256u; ++i) outputs[base + 384u + i] = decoded[i];
	outputs[base + 640u] = canonical ? 1u : 0u;
}
