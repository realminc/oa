// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mldsa_field;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void field_components(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x * 4u;
	uint o = tid.x * 14u;
	uint a = inputs[i], b = inputs[i + 1u];
	uint gamma2 = inputs[i + 2u], hint = inputs[i + 3u];
	outputs[o] = mldsa_mul(a, b);
	outputs[o + 1u] = mldsa_from_mont(mldsa_to_mont(a));
	outputs[o + 2u] = mldsa_to_mont(a);
	outputs[o + 3u] = mldsa_add(a, b);
	outputs[o + 4u] = mldsa_sub(a, b);
	outputs[o + 5u] = mldsa_neg(a);
	outputs[o + 6u] = (uint)mldsa_centered(a);
	outputs[o + 7u] = mldsa_power2round_hi(a);
	outputs[o + 8u] = (uint)mldsa_power2round_lo(a);
	outputs[o + 9u] = mldsa_highbits(a, gamma2);
	outputs[o + 10u] = (uint)mldsa_lowbits(a, gamma2);
	outputs[o + 11u] = mldsa_make_hint(b, a, gamma2);
	outputs[o + 12u] = mldsa_use_hint(hint, a, gamma2);
	outputs[o + 13u] = mldsa_from_mont(mldsa_to_mont(b));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void wide_components(uint3 tid : SV_DispatchThreadID) {
	uint i = tid.x * 2u;
	uint2 product = mul32_wide(inputs[i], inputs[i + 1u]);
	outputs[i] = product.x;
	outputs[i + 1u] = product.y;
}
