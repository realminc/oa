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
	uint count = pc.Batch * pc.NumHeads * pc.SeqLen * pc.HeadDim;
	if (outIndex >= count) return;
	uint p = outIndex % pc.HeadDim;
	uint q = outIndex / pc.HeadDim;
	uint s = q % pc.SeqLen;
	q /= pc.SeqLen;
	uint h = q % pc.NumHeads;
	uint b = q / pc.NumHeads;
	uint inputIndex = ((b * pc.SeqLen + s) * pc.NumHeads + h) * pc.HeadDim + p;
	OaStore(heap[pc.output_idx], outIndex, OaLoad(heap[pc.input_idx], inputIndex));
}
