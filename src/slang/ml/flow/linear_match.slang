// OARS single-pass provider for OA's composite FnFlow::linearMatch contract.

import storage;
import attributes;

struct PushConstants {
	uint clean_index;
	uint noise_index;
	uint time_index;
	uint state_index;
	uint velocity_index;
	uint element_count;
	uint rank;
	uint state_stride_0; uint state_stride_1; uint state_stride_2; uint state_stride_3;
	uint state_stride_4; uint state_stride_5; uint state_stride_6; uint state_stride_7;
	uint time_dim_0; uint time_dim_1; uint time_dim_2; uint time_dim_3;
	uint time_dim_4; uint time_dim_5; uint time_dim_6; uint time_dim_7;
	uint time_stride_0; uint time_stride_1; uint time_stride_2; uint time_stride_3;
	uint time_stride_4; uint time_stride_5; uint time_stride_6; uint time_stride_7;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[ForceInline]
uint stateStride(uint dim) {
	if (dim == 0u) return push.state_stride_0;
	if (dim == 1u) return push.state_stride_1;
	if (dim == 2u) return push.state_stride_2;
	if (dim == 3u) return push.state_stride_3;
	if (dim == 4u) return push.state_stride_4;
	if (dim == 5u) return push.state_stride_5;
	if (dim == 6u) return push.state_stride_6;
	return push.state_stride_7;
}

[ForceInline]
uint timeDim(uint dim) {
	if (dim == 0u) return push.time_dim_0;
	if (dim == 1u) return push.time_dim_1;
	if (dim == 2u) return push.time_dim_2;
	if (dim == 3u) return push.time_dim_3;
	if (dim == 4u) return push.time_dim_4;
	if (dim == 5u) return push.time_dim_5;
	if (dim == 6u) return push.time_dim_6;
	return push.time_dim_7;
}

[ForceInline]
uint timeStride(uint dim) {
	if (dim == 0u) return push.time_stride_0;
	if (dim == 1u) return push.time_stride_1;
	if (dim == 2u) return push.time_stride_2;
	if (dim == 3u) return push.time_stride_3;
	if (dim == 4u) return push.time_stride_4;
	if (dim == 5u) return push.time_stride_5;
	if (dim == 6u) return push.time_stride_6;
	return push.time_stride_7;
}

[ForceInline]
uint timeIndex(uint flat_index) {
	uint remaining = flat_index;
	uint time_index = 0u;
	[unroll] for (uint dim = 0u; dim < 8u; ++dim) {
		if (dim >= push.rank) {
			break;
		}
		uint stride = stateStride(dim);
		uint coordinate = remaining / stride;
		remaining -= coordinate * stride;
		if (timeDim(dim) != 1u) {
			time_index += coordinate * timeStride(dim);
		}
	}
	return time_index;
}

[kernel_name("linear_match")]
[domain("ml")]
[variant("broadcast_time")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatch_thread_id : SV_DispatchThreadID) {
	uint index = dispatch_thread_id.x;
	if (index >= push.element_count) {
		return;
	}
	float clean = load_f32(storage_buffers[push.clean_index], index);
	float noise = load_f32(storage_buffers[push.noise_index], index);
	float velocity = noise - clean;
	float time = load_f32(storage_buffers[push.time_index], timeIndex(index));
	store_f32(storage_buffers[push.state_index], index, clean + velocity * time);
	store_f32(storage_buffers[push.velocity_index], index, velocity);
}
