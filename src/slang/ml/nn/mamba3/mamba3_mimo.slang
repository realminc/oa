// Reference-faithful Mamba-3 MIMO selective scan.
//
// One workgroup owns (batch, head). Rank channels share one [P,N] state:
//   U[p,n] = sum_r V[r,p] * K[r,n]
//   H       = decay * H + scale * U
// Each Q[r] queries that same state. Rank outputs are gated (and optionally
// RMS-normalized) before MIMO_Out contracts them back to [B,L,H,P].

import storage;
import activations;
import attributes;

static const uint THREADS = 128u;
static const uint MAXN = 128u;
static const uint MAXA = 64u;
static const uint MAXR = 8u;
static const uint MAXP = 128u;

struct PushConstants {
	uint c_idx, b_idx, x_idx, z_idx, adt_idx, dt_idx, trap_idx, angle_idx;
	uint cbias_idx, bbias_idx, d_idx, mx_idx, mz_idx, mo_idx, norm_idx, y_idx;
	uint b, l, h, g, p, n, a, r;
	uint has_z, has_d, has_norm;
	float eps;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

groupshared float gsQ[MAXR * MAXN];
groupshared float gsK[MAXR * MAXN];
groupshared float gsTheta[MAXA];
groupshared float gsPre[MAXR * MAXP];
groupshared float gsInv[MAXR];

[kernel_name("mamba3_mimo")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint bh = gid.x;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint g = h / (push.h / push.g);
	uint p = tid.x;
	bool activeP = p < push.p;
	float state[MAXN];
	for (uint n = 0u; n < push.n; ++n) state[n] = 0.0f;
	for (uint a = tid.x; a < push.a; a += THREADS) gsTheta[a] = 0.0f;
	GroupMemoryBarrierWithGroupSync();

	float dh = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;
	for (uint t = 0u; t < push.l; ++t) {
		uint scalar = (b * push.l + t) * push.h + h;
		float dt = load_f32(storage_buffers[push.dt_idx], scalar);
		float trap = sigmoid(load_f32(storage_buffers[push.trap_idx], scalar));
		float decay = exp(load_f32(storage_buffers[push.adt_idx], scalar));
		float gamma = dt * trap;
		float shifted = 0.0f;
		if (t + 1u < push.l) {
			uint next = (b * push.l + t + 1u) * push.h + h;
			float nextTrap = sigmoid(load_f32(storage_buffers[push.trap_idx], next));
			shifted = load_f32(storage_buffers[push.dt_idx], next) * (1.0f - nextTrap);
		}
		float scale = gamma + shifted;
		uint angleBase = (b * push.l + t) * push.a;
		for (uint a = tid.x; a < push.a; a += THREADS)
			gsTheta[a] += load_f32(storage_buffers[push.angle_idx], angleBase + a) * dt;
		for (uint linear = tid.x; linear < push.r * push.n; linear += THREADS) {
			uint r = linear / push.n;
			uint n = linear % push.n;
			uint rg = r * push.g + g;
			uint input = (((b * push.l + t) * (push.r * push.g) + rg) * push.n) + n;
			uint bias = (h * push.r + r) * push.n + n;
			gsQ[linear] = load_f32(storage_buffers[push.c_idx], input) + load_f32(storage_buffers[push.cbias_idx], bias);
			gsK[linear] = load_f32(storage_buffers[push.b_idx], input) + load_f32(storage_buffers[push.bbias_idx], bias);
		}
		GroupMemoryBarrierWithGroupSync();
		for (uint linear = tid.x; linear < push.r * push.a; linear += THREADS) {
			uint r = linear / push.a;
			uint a = linear % push.a;
			uint i0 = r * push.n + 2u * a;
			uint i1 = i0 + 1u;
			float cs = cos(gsTheta[a]), sn = sin(gsTheta[a]);
			float q0 = gsQ[i0], q1 = gsQ[i1];
			float k0 = gsK[i0], k1 = gsK[i1];
			gsQ[i0] = q0 * cs - q1 * sn;
			gsQ[i1] = q0 * sn + q1 * cs;
			gsK[i0] = k0 * cs - k1 * sn;
			gsK[i1] = k0 * sn + k1 * cs;
		}
		GroupMemoryBarrierWithGroupSync();

		if (activeP) {
			uint xp = ((b * push.l + t) * push.h + h) * push.p + p;
			float xv = load_f32(storage_buffers[push.x_idx], xp);
			float u[MAXN];
			for (uint n = 0u; n < push.n; ++n) {
				float value = 0.0f;
				for (uint r = 0u; r < push.r; ++r) {
					uint rp = (h * push.r + r) * push.p + p;
					value += xv * load_f32(storage_buffers[push.mx_idx], rp) * gsK[r * push.n + n];
				}
				u[n] = value;
			}
			for (uint r = 0u; r < push.r; ++r) {
				float pre = 0.0f;
				for (uint n = 0u; n < push.n; ++n) {
					float hd = decay * state[n];
					pre += gsQ[r * push.n + n] * (hd + gamma * u[n]);
				}
				uint rp = (h * push.r + r) * push.p + p;
				pre += dh * xv * load_f32(storage_buffers[push.mx_idx], rp);
				gsPre[r * MAXP + p] = pre;
			}
			for (uint n = 0u; n < push.n; ++n)
				state[n] = decay * state[n] + scale * u[n];
		}
		GroupMemoryBarrierWithGroupSync();
		if (push.has_norm != 0u && tid.x < push.r) {
			uint r = tid.x;
			float sumSq = 0.0f;
			for (uint j = 0u; j < push.p; ++j) {
				float value = gsPre[r * MAXP + j];
				sumSq += value * value;
			}
			gsInv[r] = rsqrt(sumSq / float(push.p) + push.eps);
		}
		GroupMemoryBarrierWithGroupSync();
		if (activeP) {
			uint xp = ((b * push.l + t) * push.h + h) * push.p + p;
			float z = load_f32(storage_buffers[push.z_idx], xp);
			float outValue = 0.0f;
			for (uint r = 0u; r < push.r; ++r) {
				uint rp = (h * push.r + r) * push.p + p;
				float core = gsPre[r * MAXP + p];
				if (push.has_norm != 0u)
					core *= gsInv[r] * load_f32(storage_buffers[push.norm_idx], h * push.p + p);
				float gate = push.has_z != 0u
					? silu(z * load_f32(storage_buffers[push.mz_idx], rp)) : 1.0f;
				outValue += core * gate * load_f32(storage_buffers[push.mo_idx], rp);
			}
			store_f32(storage_buffers[push.y_idx], xp, outValue);
		}
		GroupMemoryBarrierWithGroupSync();
	}
}
