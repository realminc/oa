// Batched transpose of the last two axes of a rank-3 tensor:
//   in [B, R, C]  →  out [B, C, R],   out[b, c, r] = in[b, r, c]
// Tiled: SMEM-staged per batch slice, coalesced reads AND writes.
// SMEM[TILE][TILE+1] — +1 column padding breaks bank conflicts on transposed reads.
// One workgroup per (col-tile, row-tile, batch); dispatch = (ceil(C/TILE), ceil(R/TILE), B).
// (The rank-2 case is TransposeTiled; this is the [B,T,C]↔[B,C,T] channels-first conv case.)

import Storage;

static const uint TILE     = 32;
static const uint TILE_PAD = TILE + 1;

groupshared float smTile[TILE * TILE_PAD];

struct PushConstants {
	uint x_idx;      // bufs[0] — runtime prepends
	uint out_idx;    // bufs[1] — runtime prepends
	uint batch;      // B
	uint rows;       // R (input axis 1)
	uint cols;       // C (input axis 2)
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint b = gid.z;                            // batch slice
	uint in_row = gid.y * TILE + lid.y;        // row in input [R, C]
	uint in_col = gid.x * TILE + lid.x;        // col in input [R, C]

	uint inBase = b * pc.rows * pc.cols;       // offset of slice b in [B, R, C]
	if (in_row < pc.rows && in_col < pc.cols) {
		smTile[lid.y * TILE_PAD + lid.x] = OaLoad(heap[pc.x_idx], inBase + in_row * pc.cols + in_col);
	} else {
		smTile[lid.y * TILE_PAD + lid.x] = 0.0f;
	}

	GroupMemoryBarrierWithGroupSync();

	uint out_row = gid.x * TILE + lid.y;       // row in output [C, R]
	uint out_col = gid.y * TILE + lid.x;       // col in output [C, R]
	uint outBase = b * pc.cols * pc.rows;      // offset of slice b in [B, C, R]

	if (out_row < pc.cols && out_col < pc.rows) {
		OaStore(heap[pc.out_idx], outBase + out_row * pc.rows + out_col, smTile[lid.x * TILE_PAD + lid.y]);
	}
}
