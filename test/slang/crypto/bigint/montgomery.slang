// Independent component harness imports the production Montgomery formulas.
import bigint_montgomery;
import bigint_modular;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void bigint_montgomery_components(uint3 tid : SV_DispatchThreadID) {
	uint start = tid.x * 385u;
	uint output_base = tid.x * 516u;
	uint words = inputs[start];
	uint a[128], b[128], modulus[128];
	for (uint i = 0u; i < 128u; ++i) {
		a[i] = inputs[start + 1u + i];
		b[i] = inputs[start + 129u + i];
		modulus[i] = inputs[start + 257u + i];
	}
	uint encoded[128], decoded[128], product[128];
	bool valid_encode = bigint_to_montgomery(a, modulus, words, encoded);
	bool valid_decode = bigint_from_montgomery(a, modulus, words, decoded);
	bool valid_product = bigint_montgomery_multiply(a, b, modulus, words, product);
	for (uint i = 0u; i < 128u; ++i) {
		outputs[output_base + i] = encoded[i];
		outputs[output_base + 128u + i] = decoded[i];
		outputs[output_base + 256u + i] = product[i];
		outputs[output_base + 384u + i] = 0u;
	}
	outputs[output_base + 512u] = uint(valid_encode);
	outputs[output_base + 513u] = uint(valid_decode);
	outputs[output_base + 514u] = uint(valid_product);
	outputs[output_base + 515u] = bigint_montgomery_factor(modulus[0]);
}
