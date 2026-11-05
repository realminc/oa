// Laboratory entry points import production formulas, never duplicate them.
import bigint_modular;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void bigint_modular_components(uint3 tid : SV_DispatchThreadID) {
	uint start = tid.x * 385u;
	uint output_base = tid.x * 516u;
	uint words = inputs[start];
	uint a[128], b[128], modulus[128];
	for (uint i = 0u; i < 128u; ++i) {
		a[i] = inputs[start + 1u + i];
		b[i] = inputs[start + 129u + i];
		modulus[i] = inputs[start + 257u + i];
	}
	uint reduced[128], sum[128], difference[128], product[128];
	bool valid_reduce = bigint_reduce(a, modulus, words, reduced);
	bool valid_add = bigint_add_mod(a, b, modulus, words, sum);
	bool valid_subtract = bigint_subtract_mod(a, b, modulus, words, difference);
	bool valid_multiply = bigint_multiply_mod(a, b, modulus, words, product);
	for (uint i = 0u; i < 128u; ++i) {
		outputs[output_base + i] = reduced[i];
		outputs[output_base + 128u + i] = sum[i];
		outputs[output_base + 256u + i] = difference[i];
		outputs[output_base + 384u + i] = product[i];
	}
	outputs[output_base + 512u] = uint(valid_reduce);
	outputs[output_base + 513u] = uint(valid_add);
	outputs[output_base + 514u] = uint(valid_subtract);
	outputs[output_base + 515u] = uint(valid_multiply);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bigint_inverse_components(uint3 tid : SV_DispatchThreadID) {
	uint start = tid.x * 385u;
	uint output_base = tid.x * 258u;
	uint words = inputs[start];
	uint a[128], exponent[128], modulus[128];
	for (uint i = 0u; i < 128u; ++i) {
		a[i] = inputs[start + 1u + i];
		exponent[i] = inputs[start + 129u + i];
		modulus[i] = inputs[start + 257u + i];
	}
	uint inverse[128], power[128];
	bool valid_inverse = bigint_inverse_mod(a, modulus, words, inverse);
	bool valid_power = bigint_pow_mod(a, exponent, modulus, words, power);
	for (uint i = 0u; i < 128u; ++i) {
		outputs[output_base + i] = inverse[i];
		outputs[output_base + 128u + i] = power[i];
	}
	outputs[output_base + 256u] = uint(valid_inverse);
	outputs[output_base + 257u] = uint(valid_power);
}
