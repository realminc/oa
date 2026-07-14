// EmpyrealmPreprocess — fused in_proj split + RMSNorm + dt + A·dt (forward).
//
// Renamed copy of Mamba3Preprocess.slang; identical SPIR-V today. Reserved for
// future Empyrealm-specific divergence while Mamba3 stays pristine.
//
// Replaces 11-13 small dispatches (8x Slice, 2x RmsNorm, 1x Add, 1x Dt, 1x Adt)
// with a single kernel. One workgroup per row of the projected [B*S, dInProj]
// tensor. Each workgroup:
//   1. RmsNorm the b slice (DState cols) and c slice (DState cols)
//   2. Copy z, x, dd_dt, dd_A, trap, angle slices to output buffers
//   3. Compute DT = clamp(softplus(dd_dt + dt_bias), dt_min, dt_max)
//   4. Compute ADT = min(-heavy_tail(dd_A), -afloor) * DT
//
// All outputs are written as flat [B*S, ...] buffers (the caller reshapes).
// The RmsNorm weight is all-ones (bias added in-kernel by the SSM scan).
// RmsNorm is applied per sub-row of DState elements (NGroups*MimoRank sub-rows).

import Reductions;
import Storage;
import SsmMath;

struct PushConstants {
	uint proj_idx;       // bufs[0]  projected [B*S, dInProj]
	uint dt_bias_idx;    // bufs[1]  dt_bias [H]
	uint z_idx;          // bufs[2]  z output [B*S, DInner]
	uint x_idx;          // bufs[3]  x output [B*S, DInner]
	uint bh_idx;         // bufs[4]  Bh output (rmsnorm'd) [B*S, nBCRows*DState]
	uint ch_idx;         // bufs[5]  Ch output (rmsnorm'd) [B*S, nBCRows*DState]
	uint dt_idx;         // bufs[6]  DT output [B*S, H]
	uint adt_idx;        // bufs[7]  ADT output [B*S, H]
	uint trap_idx;       // bufs[8]  trap output [B*S, H]
	uint angle_idx;      // bufs[9]  angle output [B*S, NumRopeAngles]
	uint rows;           // B*S
	uint d_inner;        // DInner
	uint d_state;        // DState
	uint n_heads;        // H
	uint n_rope_angles;  // A
	uint n_bc_rows;      // NGroups * MimoRank (sub-rows for RmsNorm)
	uint bc_width;       // n_bc_rows * d_state
	// Offsets within each row of the projected tensor:
	uint z_offset;       // 0
	uint x_offset;       // DInner
	uint b_offset;       // 2*DInner
	uint c_offset;       // b_offset + n_bc_rows * d_state
	uint dd_dt_offset;   // c_offset + n_bc_rows * d_state
	uint dd_a_offset;    // dd_dt_offset + NHeads
	uint trap_offset;    // dd_a_offset + NHeads
	uint angle_offset;   // trap_offset + NHeads
	uint d_in_proj;      // total row width
	float eps;           // RmsNorm epsilon
	float dt_min;
	float dt_max;
	float afloor;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;
	uint lane = ltid.x;
	uint rowBase = row * pc.d_in_proj;

	// --- Phase 1: RmsNorm for b slice (n_bc_rows sub-rows of DState) ---
	for (uint gr = 0; gr < pc.n_bc_rows; ++gr) {
		uint subBase = rowBase + pc.b_offset + gr * pc.d_state;
		float invRms = InvRms(heap[pc.proj_idx], subBase, pc.d_state, pc.eps, lane);
		GroupMemoryBarrierWithGroupSync();
		for (uint i = lane; i < pc.d_state; i += 256) {
			float v = OaLoad(heap[pc.proj_idx], subBase + i);
			OaStore(heap[pc.bh_idx], row * pc.bc_width + gr * pc.d_state + i,
				SsmRmsNormForwardNoWeight(v, invRms));
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// --- Phase 2: RmsNorm for c slice (same structure) ---
	for (uint gr = 0; gr < pc.n_bc_rows; ++gr) {
		uint subBase = rowBase + pc.c_offset + gr * pc.d_state;
		float invRms = InvRms(heap[pc.proj_idx], subBase, pc.d_state, pc.eps, lane);
		GroupMemoryBarrierWithGroupSync();
		for (uint i = lane; i < pc.d_state; i += 256) {
			float v = OaLoad(heap[pc.proj_idx], subBase + i);
			OaStore(heap[pc.ch_idx], row * pc.bc_width + gr * pc.d_state + i,
				SsmRmsNormForwardNoWeight(v, invRms));
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// --- Phase 3: Copy z, x, trap, angle ---
	for (uint i = lane; i < pc.d_inner; i += 256) {
		OaStore(heap[pc.z_idx], row * pc.d_inner + i, OaLoad(heap[pc.proj_idx], rowBase + pc.z_offset + i));
	}
	for (uint i = lane; i < pc.d_inner; i += 256) {
		OaStore(heap[pc.x_idx], row * pc.d_inner + i, OaLoad(heap[pc.proj_idx], rowBase + pc.x_offset + i));
	}
	for (uint i = lane; i < pc.n_heads; i += 256) {
		OaStore(heap[pc.trap_idx], row * pc.n_heads + i, OaLoad(heap[pc.proj_idx], rowBase + pc.trap_offset + i));
	}
	for (uint i = lane; i < pc.n_rope_angles; i += 256) {
		OaStore(heap[pc.angle_idx], row * pc.n_rope_angles + i, OaLoad(heap[pc.proj_idx], rowBase + pc.angle_offset + i));
	}

	// --- Phase 4: dt + adt (one thread per head) ---
	// DT = clamp(softplus(dd_dt + dt_bias), dt_min, dt_max)
	// ADT = min(-heavy_tail(dd_A), -afloor) * DT
	for (uint h = lane; h < pc.n_heads; h += 256) {
		float ddDt = OaLoad(heap[pc.proj_idx], rowBase + pc.dd_dt_offset + h);
		float dtBias = OaLoad(heap[pc.dt_bias_idx], h);
		float dt = SsmSoftplusClamp(ddDt + dtBias, pc.dt_min, pc.dt_max);
		OaStore(heap[pc.dt_idx], row * pc.n_heads + h, dt);

		float ddA = OaLoad(heap[pc.proj_idx], rowBase + pc.dd_a_offset + h);
		OaStore(heap[pc.adt_idx], row * pc.n_heads + h,
			SsmAdtFromDdA(ddA, dt, pc.afloor));
	}
}
