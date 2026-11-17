// UI raster clear — opaque fullscreen write of one RGBA color.
//
// Raster port of clear_compose.slang: the pipeline's blend state is disabled,
// so the returned value replaces attachment contents exactly like the compute
// shader's unconditional store.

struct PushConstants {
	uint rgba;
};

[[vk::push_constant]] PushConstants pc;

[shader("fragment")]
float4 main() : SV_Target0 {
	return float4(
		float((pc.rgba >> 24) & 0xffu) / 255.0,
		float((pc.rgba >> 16) & 0xffu) / 255.0,
		float((pc.rgba >> 8) & 0xffu) / 255.0,
		float(pc.rgba & 0xffu) / 255.0);
}
