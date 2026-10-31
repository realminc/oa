// Test laboratory calls the production helper; no duplicate bit-count formula.
import bigint_words;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void bigint_bit_components(uint3 tid : SV_DispatchThreadID) {
	uint base = tid.x * 129u;
	uint a[128];
	for (uint i = 0u; i < 128u; ++i) a[i] = inputs[base + 1u + i];
	outputs[tid.x] = bigint_leading_zeros(a, inputs[base]);
}
