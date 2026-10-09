// Laboratory entry imports the production permutation, without copying it.
import keccak;

StructuredBuffer<uint> inputs;
RWStructuredBuffer<uint> outputs;

[numthreads(1, 1, 1)]
void keccak_components(uint3 id : SV_DispatchThreadID) {
	uint base = id.x * 50u;
	uint2 state[25];
	for (uint lane = 0u; lane < 25u; ++lane) {
		state[lane] = uint2(inputs[base + lane * 2u], inputs[base + lane * 2u + 1u]);
	}
	keccak_f1600_u32(state);
	for (uint lane = 0u; lane < 25u; ++lane) {
		outputs[base + lane * 2u] = state[lane].x;
		outputs[base + lane * 2u + 1u] = state[lane].y;
	}
}
