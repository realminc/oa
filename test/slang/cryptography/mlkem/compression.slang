// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mlkem_compression;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_compression_components(uint3 tid : SV_DispatchThreadID) {
	uint input_base = tid.x * 3u, output_base = tid.x * 4u;
	uint coefficient = inputs[input_base], bits = inputs[input_base + 1u];
	uint encoded = inputs[input_base + 2u];
	uint compressed = mlkem_compress(coefficient, bits);
	uint decompressed = mlkem_decompress(encoded, bits);
	outputs[output_base] = compressed;
	outputs[output_base + 1u] = decompressed;
	outputs[output_base + 2u] = mlkem_decompress(compressed, bits);
	outputs[output_base + 3u] = mlkem_compress(decompressed, bits);
}
