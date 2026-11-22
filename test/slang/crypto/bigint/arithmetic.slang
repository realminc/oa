// Test laboratory imports the production formulas; no duplicate arithmetic.
import bigint_words;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void bigint_arithmetic_components(uint3 tid : SV_DispatchThreadID) {
	uint input_base = tid.x * 257u;
	uint output_base = tid.x * 772u;
	uint words = inputs[input_base];
	uint a[128], b[128], sum[128], difference[128];
	uint product[256], quotient[128], remainder[128];
	for (uint i = 0u; i < 128u; ++i) {
		a[i] = inputs[input_base + 1u + i];
		b[i] = inputs[input_base + 129u + i];
	}
	uint carry = bigint_add(a, b, words, sum);
	uint borrow = bigint_subtract(a, b, words, difference);
	bigint_multiply(a, b, words, words * 2u, product);
	bool valid = bigint_divide(a, b, words, quotient, remainder);
	for (uint i = 0u; i < 128u; ++i) {
		outputs[output_base + i] = sum[i];
		outputs[output_base + 128u + i] = difference[i];
		outputs[output_base + 512u + i] = quotient[i];
		outputs[output_base + 640u + i] = remainder[i];
	}
	for (uint i = 0u; i < 256u; ++i) outputs[output_base + 256u + i] = product[i];
	outputs[output_base + 768u] = carry;
	outputs[output_base + 769u] = borrow;
	outputs[output_base + 770u] = uint(bigint_compare(a, b, words) + 1);
	outputs[output_base + 771u] = uint(valid);
}
