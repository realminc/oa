// Reference-faithful single-token Mamba-3 MIMO decode.
// The [P,N] SSM state is shared across rank channels. K/V caches retain every
// rank so the deferred trapezoid contribution can complete the previous state.

import storage;
import activations;

static const uint THREADS = 128u;
static const uint MAXN = 128u;
static const uint MAXA = 64u;
static const uint MAXR = 8u;
static const uint MAXP = 128u;

struct PushConstants {
	uint c_idx, b_idx, x_idx, z_idx, adt_idx, dt_idx, trap_idx, angle_idx;
	uint cbias_idx, bbias_idx, d_idx, mx_idx, mz_idx, mo_idx, norm_idx, y_idx;
	uint ssm_idx, angle_state_idx, k_state_idx, v_state_idx;
	uint B, H, G, P, N, A, R;
	uint has_z, has_d, has_norm;
	float eps;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

groupshared float gsQ[MAXR * MAXN];
groupshared float gsK[MAXR * MAXN];
groupshared float gsTheta[MAXA];
groupshared float gsPre[MAXR * MAXP];
groupshared float gsInv[MAXR];

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
	uint bh = gid.x;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint g = h / (pc.H / pc.G);
	uint p = tid.x;
	bool activeP = p < pc.P;
	uint scalar = b * pc.H + h;
	float dt = oaLoadF32(heap[pc.dt_idx], scalar);
	float trap = Sigmoid(oaLoadF32(heap[pc.trap_idx], scalar));
	float decay = exp(oaLoadF32(heap[pc.adt_idx], scalar));
	float gamma = dt * trap;
	float deferred = dt * (1.0f - trap);

	uint angleInput = b * pc.A;
	uint angleState = (b * pc.H + h) * pc.A;
	for (uint a = tid.x; a < pc.A; a += THREADS) {
		float theta = oaLoadF32(heap[pc.angle_state_idx], angleState + a)
			+ oaLoadF32(heap[pc.angle_idx], angleInput + a) * dt;
		gsTheta[a] = theta;
		oaStoreF32(heap[pc.angle_state_idx], angleState + a, theta);
	}
	for (uint linear = tid.x; linear < pc.R * pc.N; linear += THREADS) {
		uint r = linear / pc.N;
		uint n = linear % pc.N;
		uint rg = r * pc.G + g;
		uint input = ((b * (pc.R * pc.G) + rg) * pc.N) + n;
		uint bias = (h * pc.R + r) * pc.N + n;
		gsQ[linear] = oaLoadF32(heap[pc.c_idx], input) + oaLoadF32(heap[pc.cbias_idx], bias);
		gsK[linear] = oaLoadF32(heap[pc.b_idx], input) + oaLoadF32(heap[pc.bbias_idx], bias);
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint linear = tid.x; linear < pc.R * pc.A; linear += THREADS) {
		uint r = linear / pc.A;
		uint a = linear % pc.A;
		uint i0 = r * pc.N + 2u * a;
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
		uint xp = (b * pc.H + h) * pc.P + p;
		uint stateBase = (b * pc.H + h) * pc.P * pc.N + p * pc.N;
		float xv = oaLoadF32(heap[pc.x_idx], xp);
		float u[MAXN];
		for (uint n = 0u; n < pc.N; ++n) {
			float complete = oaLoadF32(heap[pc.ssm_idx], stateBase + n);
			float currentU = 0.0f;
			for (uint r = 0u; r < pc.R; ++r) {
				uint rankState = ((b * pc.H + h) * pc.R + r);
				float prevV = oaLoadF32(heap[pc.v_state_idx], rankState * pc.P + p);
				float prevK = oaLoadF32(heap[pc.k_state_idx], rankState * pc.N + n);
				complete += deferred * prevV * prevK;
				uint rp = (h * pc.R + r) * pc.P + p;
				currentU += xv * oaLoadF32(heap[pc.mx_idx], rp) * gsK[r * pc.N + n];
			}
			float hd = decay * complete;
			u[n] = currentU;
			oaStoreF32(heap[pc.ssm_idx], stateBase + n, hd + gamma * currentU);
		}
		float dh = pc.has_d != 0u ? oaLoadF32(heap[pc.d_idx], h) : 0.0f;
		for (uint r = 0u; r < pc.R; ++r) {
			float pre = 0.0f;
			for (uint n = 0u; n < pc.N; ++n) {
				float stateNow = oaLoadF32(heap[pc.ssm_idx], stateBase + n);
				pre += gsQ[r * pc.N + n] * stateNow;
			}
			uint rp = (h * pc.R + r) * pc.P + p;
			pre += dh * xv * oaLoadF32(heap[pc.mx_idx], rp);
			gsPre[r * MAXP + p] = pre;
			uint rankState = ((b * pc.H + h) * pc.R + r);
			oaStoreF32(heap[pc.v_state_idx], rankState * pc.P + p,
				xv * oaLoadF32(heap[pc.mx_idx], rp));
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if (pc.has_norm != 0u && tid.x < pc.R) {
		uint r = tid.x;
		float sumSq = 0.0f;
		for (uint j = 0u; j < pc.P; ++j) {
			float value = gsPre[r * MAXP + j];
			sumSq += value * value;
		}
		gsInv[r] = rsqrt(sumSq / float(pc.P) + pc.eps);
	}
	GroupMemoryBarrierWithGroupSync();
	if (activeP) {
		uint xp = (b * pc.H + h) * pc.P + p;
		float z = oaLoadF32(heap[pc.z_idx], xp);
		float output = 0.0f;
		for (uint r = 0u; r < pc.R; ++r) {
			uint rp = (h * pc.R + r) * pc.P + p;
			float core = gsPre[r * MAXP + p];
			if (pc.has_norm != 0u)
				core *= gsInv[r] * oaLoadF32(heap[pc.norm_idx], h * pc.P + p);
			float gate = pc.has_z != 0u
				? Silu(z * oaLoadF32(heap[pc.mz_idx], rp)) : 1.0f;
			output += core * gate * oaLoadF32(heap[pc.mo_idx], rp);
		}
		oaStoreF32(heap[pc.y_idx], xp, output);
	}
	GroupMemoryBarrierWithGroupSync();
	for (uint linear = tid.x; linear < pc.R * pc.N; linear += THREADS) {
		uint r = linear / pc.N;
		uint n = linear % pc.N;
		uint rankState = ((b * pc.H + h) * pc.R + r);
		oaStoreF32(heap[pc.k_state_idx], rankState * pc.N + n, gsK[linear]);
	}
}
