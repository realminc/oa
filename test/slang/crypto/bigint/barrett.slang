// Independent laboratory imports the production reciprocal and reduction.
import bigint_barrett;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void bigint_barrett_components(uint3 tid : SV_DispatchThreadID) {
	uint start = tid.x * 385u;
	uint output_base = tid.x * 260u;
	uint words = inputs[start];
	uint a[128], b[128], modulus[128];
	for (uint i = 0u; i < 128u; ++i) {
		a[i] = inputs[start + 1u + i]; b[i] = inputs[start + 129u + i];
		modulus[i] = inputs[start + 257u + i];
	}
	uint result[128], mu[130], k;
	bool valid = bigint_barrett_multiply(a, b, modulus, words, result);
	bigint_barrett_reciprocal(modulus, words, mu, k);
	for (uint i = 0u; i < 128u; ++i) outputs[output_base + i] = result[i];
	for (uint i = 0u; i < 130u; ++i) outputs[output_base + 128u + i] = mu[i];
	outputs[output_base + 258u] = k;
	outputs[output_base + 259u] = uint(valid);
}
