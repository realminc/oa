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
		float a = read_source(arg_a, index, register_base);
		// Named constants only bind in comparisons: a `case NAME:` label in a
		// Slang switch is a pattern, not constant equality, so the opcode
		// dispatch must stay an if/else chain.
		if (opcode >= OP_ADD && opcode <= OP_MAX) {
			float b = read_source(arg_b, index, register_base);
			if (opcode == OP_ADD) {
				register_file[register_base + dst] = a + b;
			} else if (opcode == OP_SUB) {
				register_file[register_base + dst] = a - b;
			} else if (opcode == OP_MUL) {
				register_file[register_base + dst] = a * b;
			} else if (opcode == OP_DIV) {
				register_file[register_base + dst] = a / b;
			} else if (opcode == OP_POW) {
				register_file[register_base + dst] = pow(a, b);
			} else if (opcode == OP_MIN) {
				register_file[register_base + dst] = min(a, b);
			} else {
				register_file[register_base + dst] = max(a, b);
			}
			continue;
		}
		if (opcode == OP_NEG) {
			register_file[register_base + dst] = -a;
		} else if (opcode == OP_RECIP) {
			register_file[register_base + dst] = 1.0f / a;
		} else if (opcode == OP_ABS) {
			register_file[register_base + dst] = abs(a);
		} else if (opcode == OP_EXP) {
			register_file[register_base + dst] = exp(a);
		} else if (opcode == OP_LOG) {
			register_file[register_base + dst] = log(a);
		} else if (opcode == OP_SQRT) {
			register_file[register_base + dst] = sqrt(a);
		} else if (opcode == OP_SIN) {
			register_file[register_base + dst] = sin(a);
		} else if (opcode == OP_COS) {
			register_file[register_base + dst] = cos(a);
		} else if (opcode == OP_TANH) {
			register_file[register_base + dst] = tanh(a);
		} else if (opcode == OP_SIGMOID) {
			register_file[register_base + dst] = 1.0f / (1.0f + exp(-a));
		} else if (opcode == OP_RELU) {
			register_file[register_base + dst] = max(a, 0.0f);
		} else if (opcode == OP_GELU) {
			float inner = 0.7978845608028654f * (a + 0.044715f * a * a * a);
			register_file[register_base + dst] = 0.5f * a * (1.0f + tanh(inner));
		} else if (opcode == OP_SILU) {
			register_file[register_base + dst] = a / (1.0f + exp(-a));
		} else {
			register_file[register_base + dst] = a;
		}
	}
}
