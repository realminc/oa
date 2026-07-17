struct PushConstants {
	uint image_idx, mask_idx, palette_idx, out_idx;
	uint batch, channels, height, width, classes;
	float alpha;
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWByteAddressBuffer heap[];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
	uint element = tid.x;
	uint spatial = pc.height * pc.width;
	uint elements = pc.batch * pc.channels * spatial;
	if (element >= elements) return;
	uint n = element / (pc.channels * spatial);
	uint channelOffset = element - n * pc.channels * spatial;
	uint channel = channelOffset / spatial;
	uint pixel = channelOffset - channel * spatial;
	float source = asfloat(heap[pc.image_idx].Load(element * 4u));
	int label = asint(heap[pc.mask_idx].Load((n * spatial + pixel) * 4u));
	float result = source;
	if (channel < 3u && label >= 0 && label < int(pc.classes)) {
		float color = asfloat(heap[pc.palette_idx].Load(
			(uint(label) * 3u + channel) * 4u));
		result = source + (color - source) * pc.alpha;
	}
	heap[pc.out_idx].Store(element * 4u, asuint(result));
}
