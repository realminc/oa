import Storage;

struct PushConstants {
	uint input_idx;
	uint output_idx;
	uint Batch;
	uint SeqLen;
	uint NumHeads;
	uint HeadDim;
};

[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint outIndex = tid.x;
	uint count = pc.Batch * pc.SeqLen * pc.NumHeads * pc.HeadDim;
	if (outIndex >= count) return;
	uint p = outIndex % pc.HeadDim;
	uint q = outIndex / pc.HeadDim;
	uint h = q % pc.NumHeads;
	q /= pc.NumHeads;
	uint s = q % pc.SeqLen;
	uint b = q / pc.SeqLen;
	uint inputIndex = ((b * pc.NumHeads + h) * pc.SeqLen + s) * pc.HeadDim + p;
	OaStore(heap[pc.output_idx], outIndex, OaLoad(heap[pc.input_idx], inputIndex));
}
