// Test-only entry points; import production formulas without copying them.
// Independent host oracles and Vulkan target validation are owned by Python.
import mlkem_field;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[shader("compute")]
[numthreads(1, 1, 1)]
void mlkem_field_components(uint3 tid : SV_DispatchThreadID) {
	uint input_base = tid.x * 2u, output_base = tid.x * 5u;
	uint a = inputs[input_base], b = inputs[input_base + 1u];
	outputs[output_base] = mlkem_add(a, b);
	outputs[output_base + 1u] = mlkem_sub(a, b);
	outputs[output_base + 2u] = mlkem_mul(a, b);
	outputs[output_base + 3u] = mlkem_neg(a);
	outputs[output_base + 4u] = (uint)mlkem_centered(a);
}
