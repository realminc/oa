import Storage;

struct PushConstants {
	uint x_idx;
	uint d_out_idx;
	uint d_x_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx >= pc.count) return;

	OaStore(heap[pc.d_x_idx], idx, OaLoad(heap[pc.x_idx], idx) > 0.0 ? OaLoad(heap[pc.d_out_idx], idx) : 0.0);
}
