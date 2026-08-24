// Rankwise gate/RMSNorm/MIMO-out adjoint. One invocation owns (batch, head),
// producing deterministic parameter partials and dPre for the recurrent core.

import storage;
import activations;

static const uint MAXR = 8u;
static const uint MAXP = 128u;
static const uint MAXPN = 4096u;
static const uint MAXRP = MAXR * MAXP;

struct PushConstants {
	uint dout_idx, x_idx, z_idx, d_idx, mx_idx, mz_idx, mo_idx, norm_idx;
	uint q_idx, k_idx, hprev_idx, decay_idx, gamma_idx;
	uint dpre_idx, dz_idx, dmz_batch_idx, dmo_batch_idx, dnorm_batch_idx;
	uint B, L, H, P, N, R, has_z, has_d, has_norm;
	float eps;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
	uint bh = gid.x;
	uint h = bh % pc.H;
	uint b = bh / pc.H;
	uint mimoBatch = bh * pc.R * pc.P;
	uint normBatch = bh * pc.P;
	float u[MAXPN];
	float pre[MAXRP];
	for (uint i = 0u; i < pc.R * pc.P; ++i) {
		oaStoreF32(heap[pc.dmz_batch_idx], mimoBatch + i, 0.0f);
		oaStoreF32(heap[pc.dmo_batch_idx], mimoBatch + i, 0.0f);
	}
	for (uint p = 0u; p < pc.P; ++p)
		oaStoreF32(heap[pc.dnorm_batch_idx], normBatch + p, 0.0f);
	float dh = pc.has_d != 0u ? oaLoadF32(heap[pc.d_idx], h) : 0.0f;

	for (uint t = 0u; t < pc.L; ++t) {
		uint scalar = (b * pc.L + t) * pc.H + h;
		uint xBase = scalar * pc.P;
		float decay = oaLoadF32(heap[pc.decay_idx], scalar);
		float gamma = oaLoadF32(heap[pc.gamma_idx], scalar);
		for (uint p = 0u; p < pc.P; ++p) {
			float xv = oaLoadF32(heap[pc.x_idx], xBase + p);
			for (uint n = 0u; n < pc.N; ++n) {
				float value = 0.0f;
				for (uint r = 0u; r < pc.R; ++r) {
					uint rp = (h * pc.R + r) * pc.P + p;
					uint qk = (((scalar * pc.R + r) * pc.N) + n);
					value += xv * oaLoadF32(heap[pc.mx_idx], rp)
						* oaLoadF32(heap[pc.k_idx], qk);
				}
				u[p * pc.N + n] = value;
			}
			uint history = ((bh * pc.L + t) * pc.P + p) * pc.N;
			for (uint r = 0u; r < pc.R; ++r) {
				float value = 0.0f;
				for (uint n = 0u; n < pc.N; ++n) {
					uint qk = (((scalar * pc.R + r) * pc.N) + n);
					float state = decay * oaLoadF32(heap[pc.hprev_idx], history + n)
						+ gamma * u[p * pc.N + n];
					value += oaLoadF32(heap[pc.q_idx], qk) * state;
				}
				uint rp = (h * pc.R + r) * pc.P + p;
				pre[r * pc.P + p] = value
					+ dh * xv * oaLoadF32(heap[pc.mx_idx], rp);
			}
		}

		for (uint r = 0u; r < pc.R; ++r) {
			float inv = 1.0f;
			if (pc.has_norm != 0u) {
				float sumSq = 0.0f;
				for (uint p = 0u; p < pc.P; ++p) {
					float value = pre[r * pc.P + p];
					sumSq += value * value;
				}
				inv = rsqrt(sumSq / float(pc.P) + pc.eps);
			}
			float normDot = 0.0f;
			for (uint p = 0u; p < pc.P; ++p) {
				uint rp = (h * pc.R + r) * pc.P + p;
				float z = oaLoadF32(heap[pc.z_idx], xBase + p);
				float gateInput = z * oaLoadF32(heap[pc.mz_idx], rp);
				float gate = pc.has_z != 0u ? Silu(gateInput) : 1.0f;
				float weight = pc.has_norm != 0u
					? oaLoadF32(heap[pc.norm_idx], h * pc.P + p) : 1.0f;
				float core = pre[r * pc.P + p] * inv * weight;
				float dout = oaLoadF32(heap[pc.dout_idx], xBase + p);
				float mo = oaLoadF32(heap[pc.mo_idx], rp);
				float dCore = dout * mo * gate;
				uint batchRp = mimoBatch + r * pc.P + p;
				oaStoreF32(heap[pc.dmo_batch_idx], batchRp,
					oaLoadF32(heap[pc.dmo_batch_idx], batchRp) + dout * gate * core);
				if (pc.has_z != 0u) {
					float du = dout * mo * core * SiluGrad(gateInput);
					float dz = r == 0u ? 0.0f : oaLoadF32(heap[pc.dz_idx], xBase + p);
					oaStoreF32(heap[pc.dz_idx], xBase + p,
						dz + du * oaLoadF32(heap[pc.mz_idx], rp));
					oaStoreF32(heap[pc.dmz_batch_idx], batchRp,
						oaLoadF32(heap[pc.dmz_batch_idx], batchRp) + du * z);
				} else if (r == 0u) {
					oaStoreF32(heap[pc.dz_idx], xBase + p, 0.0f);
				}
				if (pc.has_norm != 0u) {
					oaStoreF32(heap[pc.dnorm_batch_idx], normBatch + p,
						oaLoadF32(heap[pc.dnorm_batch_idx], normBatch + p)
						+ dCore * pre[r * pc.P + p] * inv);
					normDot += dCore * weight * pre[r * pc.P + p];
				}
				oaStoreF32(heap[pc.dpre_idx], (scalar * pc.R + r) * pc.P + p,
					dCore * inv * weight);
			}
			if (pc.has_norm != 0u) {
				float correction = inv * inv * inv * normDot / float(pc.P);
				for (uint p = 0u; p < pc.P; ++p) {
					uint outIndex = (scalar * pc.R + r) * pc.P + p;
					oaStoreF32(heap[pc.dpre_idx], outIndex,
						oaLoadF32(heap[pc.dpre_idx], outIndex)
						- pre[r * pc.P + p] * correction);
				}
			}
		}
	}
}
