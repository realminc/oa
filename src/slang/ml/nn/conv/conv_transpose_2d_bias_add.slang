// Mechanical adaptation of OA's ConvTranspose2d channel-bias epilogue.
// Donor: source/cpp/lib/oa/ml/fnmatrix/modules/fnMatrixModules.cpp

import storage;
import attributes;

struct PushConstants {
	uint values_index;
	uint bias_index;
	uint batch_size;
	uint channels;
	uint height;
	uint width;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

[kernel_name("conv_transpose_2d_bias_add")]
[domain("ml")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
	uint index = thread_id.x;
	uint element_count = push.batch_size * push.channels * push.height * push.width;
	if (index >= element_count) {
		return;
	}
	uint plane_size = push.height * push.width;
	uint channel = (index / plane_size) % push.channels;
	float value = load_f32(storage_buffers[push.values_index], index);
	value += load_f32(storage_buffers[push.bias_index], channel);
	store_f32(storage_buffers[push.values_index], index, value);
}
