// UI raster fullscreen triangle — shared vertex stage for every per-pixel
// compositor family. Geometry comes from SV_VertexID; the pipeline's dynamic
// scissor limits coverage to the element rect intersected with its clip.

[shader("vertex")]
float4 main(uint vertex_id : SV_VertexID) : SV_Position {
	// (-1,-1) (3,-1) (-1,3) — covers the viewport in one triangle.
	const float2 positions[3] = {
		float2(-1.0, -1.0),
		float2(3.0, -1.0),
		float2(-1.0, 3.0),
	};
	return float4(positions[vertex_id % 3U], 0.0, 1.0);
}
