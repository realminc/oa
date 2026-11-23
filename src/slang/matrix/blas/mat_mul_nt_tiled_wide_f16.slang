// Wide-tile F16-storage specialization of `mat_mul_nt` for large canonical
// problems: 128x64x16 workgroup tiles, 256 threads, 8x4 register tiles,
// double-buffered shared memory staged through registers (one barrier per
// K tile), and half-precision vector K reads inside the shared tiles. Operands stay
// in binary16 storage end to end — global loads, the shared tiles, and the
// inner fragments — and widen to F32 only at the FMA, so both global
// bandwidth and the shared footprint are half the F32 schedule's. Edge
// rows, columns, and K tails stay bounds-checked so the variant covers any
// shape; the router structurally prefers it only where the tile is
// profitable under the explicit-FP16 mode.

import storage;
import attributes;

struct PushConstants {
	uint left_index;
	uint right_index;
	uint output_index;
	uint m;
	uint n;
	uint k;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

static const uint BM = 128;
static const uint BN = 64;
static const uint BK = 16;
static const uint TM = 8;
static const uint TN = 4;
static const uint THREADS_ROW = BN / TN;
static const uint LOAD_WIDTH = 4;
static const uint KVEC = BK / LOAD_WIDTH;
static const uint BN_F4 = BN / LOAD_WIDTH + 1;

groupshared float4 shared_left[2][BM * KVEC];
groupshared float4 shared_right[2][BK * BN_F4];

vector<float16_t, 4> load_slice(
	uint row,
	uint k_begin,
	uint buffer_index,
	uint row_limit,
	bool aligned
) {
	uint element_index = row * push.k + k_begin;
	if (aligned && row < row_limit && k_begin + LOAD_WIDTH <= push.k) {
		return storage_buffers[buffer_index].Load<vector<float16_t, 4>>(element_index * 2u);
	}
	vector<float16_t, 4> values;
	[[unroll]] for (uint element = 0; element < LOAD_WIDTH; ++element) {
		values[element] = row < row_limit && k_begin + element < push.k
			? storage_buffers[buffer_index].Load<float16_t>(
				(row * push.k + k_begin + element) * 2u)
			: float16_t(0.0);
	}
	return values;
}

[kernel_name("mat_mul_nt_tiled_wide_f16")]
[domain("matrix")]
[variant("tiled_wide_f16")]
[dtype("f16")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint thread = group_thread_id.x;
	uint thread_row = thread / THREADS_ROW;
	uint thread_column = thread % THREADS_ROW;
	uint row_base = group_id.x * BM;
	uint column_base = group_id.y * BN;

	uint load_row = (thread * LOAD_WIDTH) / BK;
	uint load_column = (thread * LOAD_WIDTH) % BK;
	// vector<float16_t, 4> global loads need 8-byte alignment: with K a multiple of four
	// every row start and K-tile offset is aligned; odd K falls back to
	// scalar half loads everywhere.
	bool aligned = (push.k & 3u) == 0u;

	vector<float16_t, 4> stage_a[2];
	vector<float16_t, 4> stage_b[1];

	uint k_tile_count = (push.k + BK - 1u) / BK;
	uint k_base = 0u;

	stage_a[0] = load_slice(
		row_base + load_row, k_base + load_column, push.left_index, push.m, aligned);
	stage_a[1] = load_slice(
		row_base + load_row + 64u, k_base + load_column, push.left_index, push.m, aligned);
	stage_b[0] = load_row < BN
		? load_slice(
			column_base + load_row, k_base + load_column, push.right_index, push.n, aligned)
		: vector<float16_t, 4>(float16_t(0.0));

	uint load_slot = load_row * KVEC + load_column / LOAD_WIDTH;
	uint right_slot = load_row / LOAD_WIDTH;
	uint right_lane = load_row % LOAD_WIDTH;
	shared_left[0][load_slot] = float4(stage_a[0]);
	shared_left[0][load_slot + 64u * KVEC] = float4(stage_a[1]);
	if (load_row < BN) {
		shared_right[0][(load_column + 0u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].x);
		shared_right[0][(load_column + 1u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].y);
		shared_right[0][(load_column + 2u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].z);
		shared_right[0][(load_column + 3u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].w);
	}
	GroupMemoryBarrierWithGroupSync();

	uint buffer = 0u;
	float accumulators[TM * TN];
	[[unroll]] for (uint index = 0; index < TM * TN; ++index) {
		accumulators[index] = 0.0f;
	}

	for (uint tile = 0; tile < k_tile_count; ++tile) {
		k_base += BK;
		bool prefetch = tile + 1u < k_tile_count;
		if (prefetch) {
			stage_a[0] = load_slice(
				row_base + load_row, k_base + load_column, push.left_index, push.m, aligned);
			stage_a[1] = load_slice(
				row_base + load_row + 64u, k_base + load_column, push.left_index, push.m, aligned);
			stage_b[0] = load_row < BN
				? load_slice(
					column_base + load_row, k_base + load_column, push.right_index, push.n, aligned)
				: vector<float16_t, 4>(float16_t(0.0));
		}

		[[unroll]] for (uint kk = 0; kk < BK / LOAD_WIDTH; ++kk) {
			float4 right_fragment[TN];
			[[unroll]] for (uint lane = 0; lane < LOAD_WIDTH; ++lane) {
				right_fragment[lane] =
					shared_right[buffer][(kk * LOAD_WIDTH + lane) * BN_F4 + thread_column];
			}
			[[unroll]] for (uint row = 0; row < TM; ++row) {
				float4 left_fragment =
					shared_left[buffer][(thread_row * TM + row) * KVEC + kk];
				[[unroll]] for (uint column = 0; column < TN; ++column) {
					[[unroll]] for (uint lane = 0; lane < LOAD_WIDTH; ++lane) {
						accumulators[row * TN + column] +=
							left_fragment[lane] * right_fragment[lane][column];
					}
				}
			}
		}

		if (prefetch) {
			uint next = buffer ^ 1u;
			shared_left[next][load_slot] = float4(stage_a[0]);
			shared_left[next][load_slot + 64u * KVEC] = float4(stage_a[1]);
			if (load_row < BN) {
				shared_right[next][(load_column + 0u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].x);
				shared_right[next][(load_column + 1u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].y);
				shared_right[next][(load_column + 2u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].z);
				shared_right[next][(load_column + 3u) * BN_F4 + right_slot][right_lane] = float(stage_b[0].w);
			}
			buffer = next;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	[[unroll]] for (uint row = 0; row < TM; ++row) {
		[[unroll]] for (uint column = 0; column < TN; ++column) {
			uint output_row = row_base + thread_row * TM + row;
			uint output_column = column_base + thread_column * TN + column;
			if (output_row < push.m && output_column < push.n) {
				store_f32(
					storage_buffers[push.output_index],
					output_row * push.n + output_column,
					accumulators[row * TN + column]
				);
			}
		}
	}
}
