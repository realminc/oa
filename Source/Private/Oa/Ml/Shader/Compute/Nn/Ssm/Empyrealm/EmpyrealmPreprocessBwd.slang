// EmpyrealmPreprocessBwd — fused in_proj split + RMSNorm + dt + A·dt backward.
//
// Renamed copy of Mamba3PreprocessBwd.slang; identical SPIR-V today. Inverts
// EmpyrealmPreprocess.slang in one dispatch.
//
// Receives dZ, dX, dBh, dCh, dDT, dADT, dTrap, dAngle and accumulates into dProj.
// Also computes dDtBias via a floating-point atomic add across all rows.
//
// Backward math:
//   Copy ops (z, x, trap, angle): dProj[offset+i] += dOut[i]
//   RmsNorm (b, c): dX = invRms * (dY - x * c * invRms^2 / n),  c = sum(dY * x)
//   Dt:  dDtRaw = dDT * mask * sigmoid(dtRaw)
//   Adt: dDdA = dADT * dt * mask_ht * (-h'(ddA))
//        dDt_from_adt = dADT * aTok  (accumulates into dDT)
//   dt_bias: dDtBias[h] = sum over B*S of dDtRaw[row, h]

import Reductions;
import Storage;
import SsmMath;

struct PushConstants {
	uint proj_idx;        // bufs[0]  projected [B*S, dInProj]  (forward input, read-only)
	uint dt_bias_idx;     // bufs[1]  dt_bias [H]  (forward input, read-only)
	uint dz_idx;          // bufs[2]  dZ [B*S, DInner]
	uint dx_idx;          // bufs[3]  dX [B*S, DInner]
	uint dbh_idx;         // bufs[4]  dBh [B*S, nBCRows*DState]
	uint dch_idx;         // bufs[5]  dCh [B*S, nBCRows*DState]
	uint ddt_idx;         // bufs[6]  dDT [B*S, H]
	uint dadt_idx;        // bufs[7]  dADT [B*S, H]
	uint dtrap_idx;       // bufs[8]  dTrap [B*S, H]
	uint dangle_idx;      // bufs[9]  dAngle [B*S, NumRopeAngles]
	uint dproj_idx;       // bufs[10] dProj [B*S, dInProj]  (write, zeroed by caller)
	uint ddt_bias_idx;    // bufs[11] dDtBias [H]  (write, atomic add)
	uint rows;
	uint d_inner;
	uint d_state;
	uint n_heads;
	uint n_rope_angles;
	uint n_bc_rows;      // NGroups * MimoRank
	uint bc_width;       // n_bc_rows * d_state
	uint z_offset;
	uint x_offset;
	uint b_offset;
	uint c_offset;
	uint dd_dt_offset;
	uint dd_a_offset;
	uint trap_offset;
	uint angle_offset;
	uint d_in_proj;
	float eps;
	float dt_min;
	float dt_max;
	float afloor;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float g_reduce[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
	uint row = gid.x;
	if (row >= pc.rows) return;
	uint lane = ltid.x;
	uint rowBase = row * pc.d_in_proj;

	// --- RmsNorm backward for b slice (n_bc_rows sub-rows of DState) ---
	for (uint gr = 0; gr < pc.n_bc_rows; ++gr) {
		uint subBase = rowBase + pc.b_offset + gr * pc.d_state;
		uint outBase = row * pc.bc_width + gr * pc.d_state;
		float invRms = InvRms(heap[pc.proj_idx], subBase, pc.d_state, pc.eps, lane);
		GroupMemoryBarrierWithGroupSync();

		// c = sum(dBh[i] * b[i]) over i (weight is all-ones)
		float localC = 0.0f;
		for (uint i = lane; i < pc.d_state; i += 256) {
			float xi = OaLoad(heap[pc.proj_idx], subBase + i);
			float dyi = OaLoad(heap[pc.dbh_idx], outBase + i);
			localC += dyi * xi;
		}
		g_reduce[lane] = localC;
		GroupMemoryBarrierWithGroupSync();
		for (uint s = 128; s > 0; s >>= 1) {
			if (lane < s) g_reduce[lane] += g_reduce[lane + s];
			GroupMemoryBarrierWithGroupSync();
		}
		float cb = g_reduce[0];
		GroupMemoryBarrierWithGroupSync();

		float dn = float(pc.d_state);
		for (uint i = lane; i < pc.d_state; i += 256) {
			float xi = OaLoad(heap[pc.proj_idx], subBase + i);
			float dyi = OaLoad(heap[pc.dbh_idx], outBase + i);
			OaStore(heap[pc.dproj_idx], subBase + i,
				SsmRmsNormBackwardNoWeight(xi, dyi, invRms, cb, dn));
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// --- RmsNorm backward for c slice (same structure) ---
	for (uint gr = 0; gr < pc.n_bc_rows; ++gr) {
		uint subBase = rowBase + pc.c_offset + gr * pc.d_state;
		uint outBase = row * pc.bc_width + gr * pc.d_state;
		float invRms = InvRms(heap[pc.proj_idx], subBase, pc.d_state, pc.eps, lane);
		GroupMemoryBarrierWithGroupSync();

		float localC = 0.0f;
		for (uint i = lane; i < pc.d_state; i += 256) {
			float xi = OaLoad(heap[pc.proj_idx], subBase + i);
			float dyi = OaLoad(heap[pc.dch_idx], outBase + i);
			localC += dyi * xi;
		}
		g_reduce[lane] = localC;
		GroupMemoryBarrierWithGroupSync();
		for (uint s = 128; s > 0; s >>= 1) {
			if (lane < s) g_reduce[lane] += g_reduce[lane + s];
			GroupMemoryBarrierWithGroupSync();
		}
		float cc = g_reduce[0];
		GroupMemoryBarrierWithGroupSync();

		float dn = float(pc.d_state);
		for (uint i = lane; i < pc.d_state; i += 256) {
			float xi = OaLoad(heap[pc.proj_idx], subBase + i);
			float dyi = OaLoad(heap[pc.dch_idx], outBase + i);
			OaStore(heap[pc.dproj_idx], subBase + i,
				SsmRmsNormBackwardNoWeight(xi, dyi, invRms, cc, dn));
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// --- Copy backward: z, x, trap, angle ---
	for (uint i = lane; i < pc.d_inner; i += 256) {
		OaStore(heap[pc.dproj_idx], rowBase + pc.z_offset + i,
			OaLoad(heap[pc.dz_idx], row * pc.d_inner + i));
	}
	for (uint i = lane; i < pc.d_inner; i += 256) {
		OaStore(heap[pc.dproj_idx], rowBase + pc.x_offset + i,
			OaLoad(heap[pc.dx_idx], row * pc.d_inner + i));
	}
	for (uint i = lane; i < pc.n_heads; i += 256) {
		OaStore(heap[pc.dproj_idx], rowBase + pc.trap_offset + i,
			OaLoad(heap[pc.dtrap_idx], row * pc.n_heads + i));
	}
	for (uint i = lane; i < pc.n_rope_angles; i += 256) {
		OaStore(heap[pc.dproj_idx], rowBase + pc.angle_offset + i,
			OaLoad(heap[pc.dangle_idx], row * pc.n_rope_angles + i));
	}

	// --- dt + adt backward ---
	// Forward: dtRaw = dd_dt + dt_bias; sp = softplus(dtRaw); DT = clamp(sp, dt_min, dt_max)
	//          h = heavy_tail(ddA); aTok = min(-h, -afloor); ADT = aTok * DT
	// dDT comes from two sources: direct dDT + dADT * aTok
	// dDtRaw = dDT_total * mask * sigmoid(dtRaw)
	// dDdA = dADT * DT * mask_ht * (-h'(ddA))
	// dDtBias[h] += dDtRaw (accumulated across all rows via atomic add)
	for (uint h = lane; h < pc.n_heads; h += 256) {
		float ddDt = OaLoad(heap[pc.proj_idx], rowBase + pc.dd_dt_offset + h);
		float dtBias = OaLoad(heap[pc.dt_bias_idx], h);
		float dtRaw = ddDt + dtBias;
		float sp = max(dtRaw, 0.0f) + log(1.0f + exp(-abs(dtRaw)));
		float dt = clamp(sp, pc.dt_min, pc.dt_max);

		float ddA = OaLoad(heap[pc.proj_idx], rowBase + pc.dd_a_offset + h);
		float heavyH = (ddA >= 0.0f) ? (1.0f + ddA) : (1.0f / (1.0f - ddA));
		float aTok = min(-heavyH, -pc.afloor);

		// dDT_total = dDT + dADT * aTok
		float dDT = OaLoad(heap[pc.ddt_idx], row * pc.n_heads + h);
		float dADT = OaLoad(heap[pc.dadt_idx], row * pc.n_heads + h);
		float dDTtotal = dDT + dADT * aTok;

		// dDtRaw = dDTtotal * mask * sigmoid(dtRaw)
		float dtMask = (sp >= pc.dt_min && sp <= pc.dt_max) ? 1.0f : 0.0f;
		float sig = 1.0f - exp(-sp);
		float dDtRaw = dDTtotal * dtMask * sig;

		// Write dDtRaw to dProj at dd_dt position
		OaStore(heap[pc.dproj_idx], rowBase + pc.dd_dt_offset + h, dDtRaw);

		// Accumulate dDtBias via floating-point atomic add (CAS loop).
		OaAtomicAdd(heap[pc.ddt_bias_idx], h, dDtRaw);

		// dDdA = dADT * dt * mask_ht * (-h'(ddA))
		float htMask = (heavyH > pc.afloor) ? 1.0f : 0.0f;
		float hp = (ddA >= 0.0f) ? 1.0f : (1.0f / ((1.0f - ddA) * (1.0f - ddA)));
		float dDdA = dADT * dt * htMask * (-hp);
		OaStore(heap[pc.dproj_idx], rowBase + pc.dd_a_offset + h, dDdA);
	}
}
