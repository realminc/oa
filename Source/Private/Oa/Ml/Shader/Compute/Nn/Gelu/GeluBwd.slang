import Activations;
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

	float xi = OaLoad(heap[pc.x_idx], idx);
	float kSqrt2OverPi = 0.7978845608;
	float kCoeff = 0.044715;
	float inner = kSqrt2OverPi * (xi + kCoeff * xi * xi * xi);
	float tanhInner = tanh(inner);
	float sech2 = 1.0 - tanhInner * tanhInner;
	float dInner = kSqrt2OverPi * (1.0 + 3.0 * kCoeff * xi * xi);
	float grad = 0.5 * (1.0 + tanhInner) + 0.5 * xi * sech2 * dInner;

	OaStore(heap[pc.d_x_idx], idx, OaLoad(heap[pc.d_out_idx], idx) * grad);
}
