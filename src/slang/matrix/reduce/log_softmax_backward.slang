// Mechanical adaptation of OA's output-saved LogSoftmax adjoint.
// Donor: source/cpp/lib/oa/ml/shader/compute/nn/softmax/logSoftmaxBwd.slang

import storage;
import attributes;

struct PushConstants {
	uint forward_output_index;
	uint output_gradient_index;
	uint input_gradient_index;
	uint outer_size;
	uint dim_size;
	uint inner_size;
};

[[vk::push_constant]] PushConstants push;
[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];

// Separate read and write banks preserve the donor's barrier-safe reduction.
groupshared float reduction[2][256];

[kernel_name("log_softmax_backward")]
[domain("matrix")]
[variant("generic")]
[dtype("f32")]
[status("experimental")]
[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID) {
	uint group = group_id.x;
	uint lane = group_thread_id.x;
	if (group >= push.outer_size * push.inner_size) {
		return;
	}

	uint outer = group / push.inner_size;
	uint inner = group % push.inner_size;
	uint base = outer * push.dim_size * push.inner_size + inner;
	uint read_bank = 0;
	uint write_bank = 1;

	float local_sum = 0.0f;
	for (uint index = lane; index < push.dim_size; index += 256) {
		local_sum += load_f32(
			storage_buffers[push.output_gradient_index], base + index * push.inner_size
		);
	}
	reduction[read_bank][lane] = local_sum;
	GroupMemoryBarrierWithGroupSync();
	for (uint width = 128; width > 0; width >>= 1) {
		if (lane < width) {
			reduction[write_bank][lane] =
				reduction[read_bank][lane] + reduction[read_bank][lane + width];
		}
		GroupMemoryBarrierWithGroupSync();
		uint previous_read_bank = read_bank;
		read_bank = write_bank;
		write_bank = previous_read_bank;
	}
	float gradient_sum = reduction[read_bank][0];
	for (uint index = lane; index < push.dim_size; index += 256) {
		uint element = base + index * push.inner_size;
		float output = load_f32(storage_buffers[push.forward_output_index], element);
		float upstream = load_f32(storage_buffers[push.output_gradient_index], element);
		store_f32(
			storage_buffers[push.input_gradient_index],
			element,
			upstream - exp(output) * gradient_sum
		);
	}
}
