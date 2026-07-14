import Storage;

struct PushConstants {
	uint a_idx;
	uint b_idx;
	uint dout_idx;
	uint da_idx;
	uint db_idx;
	uint count;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint idx = tid.x;
	if (idx < pc.count) {
		float a_val = OaLoad(heap[pc.a_idx], idx);
		float b_val = OaLoad(heap[pc.b_idx], idx);
		float dout_val = OaLoad(heap[pc.dout_idx], idx);
		
		// Gradient of C = A * B
		OaStore(heap[pc.da_idx], idx, dout_val * b_val);  // dA = dC * B
		OaStore(heap[pc.db_idx], idx, dout_val * a_val);  // dB = dC * A
	}
}