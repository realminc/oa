// Rankwise gate/RMSNorm/MIMO-out adjoint. One invocation owns (batch, head),
// producing deterministic parameter partials and dPre for the recurrent core.

import storage;
import activations;
import attributes;

static const uint MAXR = 8u;
static const uint MAXP = 128u;
static const uint MAXPN = 4096u;
static const uint MAXRP = MAXR * MAXP;

struct PushConstants {
	uint dout_idx, x_idx, z_idx, d_idx, mx_idx, mz_idx, mo_idx, norm_idx;
	uint q_idx, k_idx, hprev_idx, decay_idx, gamma_idx;
	uint dpre_idx, dz_idx, dmz_batch_idx, dmo_batch_idx, dnorm_batch_idx;
	uint b, l, h, p, n, r, has_z, has_d, has_norm;
	float eps;
};
[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("mamba3_mimo_backward_post")]
[domain("ml")]
[variant("post")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 gid : SV_GroupID) {
	uint bh = gid.x;
	uint h = bh % push.h;
	uint b = bh / push.h;
	uint mimoBatch = bh * push.r * push.p;
	uint normBatch = bh * push.p;
	float u[MAXPN];
	float pre[MAXRP];
	for (uint i = 0u; i < push.r * push.p; ++i) {
		store_f32(storage_buffers[push.dmz_batch_idx], mimoBatch + i, 0.0f);
		store_f32(storage_buffers[push.dmo_batch_idx], mimoBatch + i, 0.0f);
	}
	for (uint p = 0u; p < push.p; ++p)
		store_f32(storage_buffers[push.dnorm_batch_idx], normBatch + p, 0.0f);
	float dh = push.has_d != 0u ? load_f32(storage_buffers[push.d_idx], h) : 0.0f;

	for (uint t = 0u; t < push.l; ++t) {
		uint scalar = (b * push.l + t) * push.h + h;
		uint xBase = scalar * push.p;
		float decay = load_f32(storage_buffers[push.decay_idx], scalar);
		float gamma = load_f32(storage_buffers[push.gamma_idx], scalar);
		for (uint p = 0u; p < push.p; ++p) {
			float xv = load_f32(storage_buffers[push.x_idx], xBase + p);
			for (uint n = 0u; n < push.n; ++n) {
				float value = 0.0f;
				for (uint r = 0u; r < push.r; ++r) {
					uint rp = (h * push.r + r) * push.p + p;
					uint qk = (((scalar * push.r + r) * push.n) + n);
					value += xv * load_f32(storage_buffers[push.mx_idx], rp)
						* load_f32(storage_buffers[push.k_idx], qk);
				}
				u[p * push.n + n] = value;
			}
			uint history = ((bh * push.l + t) * push.p + p) * push.n;
			for (uint r = 0u; r < push.r; ++r) {
				float value = 0.0f;
				for (uint n = 0u; n < push.n; ++n) {
					uint qk = (((scalar * push.r + r) * push.n) + n);
					float state = decay * load_f32(storage_buffers[push.hprev_idx], history + n)
						+ gamma * u[p * push.n + n];
					value += load_f32(storage_buffers[push.q_idx], qk) * state;
				}
				uint rp = (h * push.r + r) * push.p + p;
				pre[r * push.p + p] = value
					+ dh * xv * load_f32(storage_buffers[push.mx_idx], rp);
			}
		}

		for (uint r = 0u; r < push.r; ++r) {
			float inv = 1.0f;
			if (push.has_norm != 0u) {
				float sumSq = 0.0f;
				for (uint p = 0u; p < push.p; ++p) {
					float value = pre[r * push.p + p];
					sumSq += value * value;
				}
				inv = rsqrt(sumSq / float(push.p) + push.eps);
			}
			float normDot = 0.0f;
			for (uint p = 0u; p < push.p; ++p) {
				uint rp = (h * push.r + r) * push.p + p;
				float z = load_f32(storage_buffers[push.z_idx], xBase + p);
				float gateInput = z * load_f32(storage_buffers[push.mz_idx], rp);
				float gate = push.has_z != 0u ? silu(gateInput) : 1.0f;
				float weight = push.has_norm != 0u
					? load_f32(storage_buffers[push.norm_idx], h * push.p + p) : 1.0f;
				float core = pre[r * push.p + p] * inv * weight;
				float dout = load_f32(storage_buffers[push.dout_idx], xBase + p);
				float mo = load_f32(storage_buffers[push.mo_idx], rp);
				float dCore = dout * mo * gate;
				uint batchRp = mimoBatch + r * push.p + p;
				store_f32(storage_buffers[push.dmo_batch_idx], batchRp,
					load_f32(storage_buffers[push.dmo_batch_idx], batchRp) + dout * gate * core);
				if (push.has_z != 0u) {
					float du = dout * mo * core * silu_grad(gateInput);
					float dz = r == 0u ? 0.0f : load_f32(storage_buffers[push.dz_idx], xBase + p);
					store_f32(storage_buffers[push.dz_idx], xBase + p,
						dz + du * load_f32(storage_buffers[push.mz_idx], rp));
					store_f32(storage_buffers[push.dmz_batch_idx], batchRp,
						load_f32(storage_buffers[push.dmz_batch_idx], batchRp) + du * z);
				} else if (r == 0u) {
					store_f32(storage_buffers[push.dz_idx], xBase + p, 0.0f);
				}
				if (push.has_norm != 0u) {
					store_f32(storage_buffers[push.dnorm_batch_idx], normBatch + p,
						load_f32(storage_buffers[push.dnorm_batch_idx], normBatch + p)
						+ dCore * pre[r * push.p + p] * inv);
					normDot += dCore * weight * pre[r * push.p + p];
				}
				store_f32(storage_buffers[push.dpre_idx], (scalar * push.r + r) * push.p + p,
					dCore * inv * weight);
			}
			if (push.has_norm != 0u) {
				float correction = inv * inv * inv * normDot / float(push.p);
				for (uint p = 0u; p < push.p; ++p) {
					uint outIndex = (scalar * push.r + r) * push.p + p;
					store_f32(storage_buffers[push.dpre_idx], outIndex,
						load_f32(storage_buffers[push.dpre_idx], outIndex)
						- pre[r * push.p + p] * correction);
				}
			}
		}
	}
}
