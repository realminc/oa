// Fused elementwise chain interpreter: bounded op-tape over the storage_buffers ABI.
//
// The tape is a u32 quad per step {opcode, a, b, dst}. Source operands encode
// REGISTER_COUNT as the split: values below it address the per-invocation
// register file; values at or above it carry the resolved storage_buffers[]
// index recorded by the host (local slot on Compatibility, heap index on
// Strict), so no profile-specific push lookup is needed. Immediate fields
// carry raw f32 bits for LOAD_IMM and the clamp pair. STORE steps carry their
// target the same way sources do: arg_b holds REGISTER_COUNT plus the resolved
// storage_buffers[] index of the materialized write, so one dispatch may emit
// several observed intermediates alongside its final output.
//
// The tape is uniform across the workgroup: all threads stage it into shared
// memory once (tail threads included), then tail threads exit after the
// barrier. The register file lives in shared memory too — a private array
// indexed by tape fields would lower to scratch memory, which costs a global
// round-trip per step on this class of device.

import storage;
import elementwise;
import attributes;

static const uint REGISTER_COUNT = 8;
static const uint MAX_OPERAND_SLOTS = 16;
static const uint TAPE_STEP_WORDS = 4;
static const uint MAX_TAPE_STEPS = 64;
static const uint WORKGROUP_SIZE = 256;

static const uint OP_NOP = 0;
static const uint OP_LOAD_IMM = 1;
static const uint OP_STORE = 2;
static const uint OP_ADD = 10;
static const uint OP_SUB = 11;
static const uint OP_MUL = 12;
static const uint OP_DIV = 13;
static const uint OP_POW = 14;
static const uint OP_MIN = 15;
static const uint OP_MAX = 16;
static const uint OP_NEG = 20;
static const uint OP_RECIP = 21;
static const uint OP_ABS = 22;
static const uint OP_EXP = 23;
static const uint OP_LOG = 24;
static const uint OP_SQRT = 25;
static const uint OP_SIN = 26;
static const uint OP_COS = 27;
static const uint OP_TANH = 28;
static const uint OP_SIGMOID = 30;
static const uint OP_RELU = 31;
static const uint OP_GELU = 32;
static const uint OP_SILU = 33;

struct PushConstants {
	uint operand0_index;
	uint operand1_index;
	uint operand2_index;
	uint operand3_index;
	uint operand4_index;
	uint operand5_index;
	uint operand6_index;
	uint operand7_index;
	uint operand8_index;
	uint operand9_index;
	uint operand10_index;
	uint operand11_index;
	uint operand12_index;
	uint operand13_index;
	uint operand14_index;
	uint operand15_index;
	uint store0_index;
	uint store1_index;
	uint store2_index;
	uint store3_index;
	uint store4_index;
	uint store5_index;
	uint tape_index;
	uint element_count;
	uint step_count;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared uint tape_words[MAX_TAPE_STEPS * TAPE_STEP_WORDS];
groupshared float register_file[WORKGROUP_SIZE * REGISTER_COUNT];

float read_source(uint source, uint index, uint register_base) {
	if (source < REGISTER_COUNT) {
		return register_file[register_base + source];
	}
	return load_f32(storage_buffers[source - REGISTER_COUNT], index);
}

[kernel_name("fused_chain")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID, uint3 group_thread_id : SV_GroupThreadID) {
	uint thread = group_thread_id.x;
	uint tape_word_count = min(push.step_count, MAX_TAPE_STEPS) * TAPE_STEP_WORDS;
	for (uint word = thread; word < tape_word_count; word += WORKGROUP_SIZE) {
		tape_words[word] = load_u32(storage_buffers[push.tape_index], word);
	}
	GroupMemoryBarrierWithGroupSync();
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	uint register_base = thread * REGISTER_COUNT;
	for (uint step = 0; step < push.step_count; step++) {
		uint base = step * TAPE_STEP_WORDS;
		uint opcode = tape_words[base];
		uint arg_a = tape_words[base + 1];
		uint arg_b = tape_words[base + 2];
		uint dst = tape_words[base + 3];
		if (opcode == OP_NOP) {
			continue;
		}
		if (opcode == OP_LOAD_IMM) {
			register_file[register_base + dst] = asfloat(arg_a);
			continue;
		}
		if (opcode == OP_STORE) {
			store_f32(
				storage_buffers[arg_b - REGISTER_COUNT],
				index,
				register_file[register_base + arg_a]
			);
			continue;
		}
		if (opcode == 3u) {
			uint remainder = index;
			uint source_index = 0u;
			for (int axis = int(arg_b) - 1; axis >= 0; --axis) {
				uint dim = tape_words[base + 4u + axis];
				uint stride = tape_words[base + 12u + axis];
				source_index += (remainder % dim) * stride;
				remainder /= dim;
			}
			register_file[register_base + dst] = load_f32(storage_buffers[arg_a - REGISTER_COUNT], source_index);
			step += 4u;
			continue;
		}
		if (opcode == 4u) {
			uint rows = tape_words[base + 4u];
			uint width = tape_words[base + 5u];
			uint dtype = tape_words[base + 6u];
			uint position = index / width;
			uint row = dtype == 0u ? load_u8(storage_buffers[arg_b - REGISTER_COUNT], position)
				: storage_buffers[arg_b - REGISTER_COUNT].Load(position * 4u);
			register_file[register_base + dst] = row < rows ? load_f32(storage_buffers[arg_a - REGISTER_COUNT], row * width + index % width) : asfloat(0x7fc00000u);
			step += 1u;
			continue;
		}
		float a = read_source(arg_a, index, register_base);
		float b = (opcode >= 10u && opcode <= 16u) || (opcode >= 40u && opcode <= 48u)
			? read_source(arg_b, index, register_base) : 0.0f;
		float extra = 0.0f;
		if (opcode == 47u || opcode == 48u) {
			extra = asfloat(tape_words[base + 4u]);
			step += 1u;
		}
		register_file[register_base + dst] = eval_element(opcode, a, b, extra);
	}
}
