// GradScaler-owned replay-safe dynamic scale update. Reads the finite-check
// accumulator for the current step, writes the finite flag consumed by gated
// optimizer kernels, and applies scale backoff or interval growth in place.

import storage;
import attributes;

struct PushConstants {
	uint check_index;
	uint scale_index;
	uint counter_index;
	uint flag_index;
	float growth_factor;
	float backoff_factor;
	uint growth_interval;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("grad_scaler_advance")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	if (dispatch_thread_id.x != 0u) {
		return;
	}
	float check = load_f32(storage_buffers[push.check_index], 0u);
	float scale = load_f32(storage_buffers[push.scale_index], 0u);
	float counter = load_f32(storage_buffers[push.counter_index], 0u);
	bool finite = isfinite(check);
	if (finite) {
		counter += 1.0f;
		if (counter >= float(push.growth_interval)) {
			scale *= push.growth_factor;
			counter = 0.0f;
		}
		store_f32(storage_buffers[push.flag_index], 0u, 1.0f);
	} else {
		scale *= push.backoff_factor;
		counter = 0.0f;
		store_f32(storage_buffers[push.flag_index], 0u, 0.0f);
	}
	store_f32(storage_buffers[push.scale_index], 0u, scale);
	store_f32(storage_buffers[push.counter_index], 0u, counter);
}
