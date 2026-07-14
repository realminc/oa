// Tiled transpose: out[j, i] = in[i, j]
// Replaces Transpose.slang — coalesced reads AND writes via SMEM staging.
// SMEM[TILE][TILE+1] — +1 column padding breaks bank conflicts on transposed reads.
// Dispatch: (ceil(cols/TILE), ceil(rows/TILE), 1)

import Storage;

static const uint TILE     = 32;
static const uint TILE_PAD = TILE + 1;

groupshared float smTile[TILE * TILE_PAD];

struct PushConstants {
	uint x_idx;
	uint out_idx;
	uint rows;
	uint cols;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void main(uint3 gid : SV_GroupID, uint3 lid : SV_GroupThreadID) {
	uint in_row = gid.y * TILE + lid.y;
	uint in_col = gid.x * TILE + lid.x;

	if (in_row < pc.rows && in_col < pc.cols) {
		smTile[lid.y * TILE_PAD + lid.x] = OaLoad(heap[pc.x_idx], in_row * pc.cols + in_col);
	} else {
		smTile[lid.y * TILE_PAD + lid.x] = 0.0f;
	}

	GroupMemoryBarrierWithGroupSync();

	uint out_row = gid.x * TILE + lid.y;
	uint out_col = gid.y * TILE + lid.x;

	if (out_row < pc.cols && out_col < pc.rows) {
		OaStore(heap[pc.out_idx], out_row * pc.rows + out_col, smTile[lid.x * TILE_PAD + lid.y]);
	}
}
