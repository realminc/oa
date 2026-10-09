// Flat-color vertex shader — Tier 1 material, UI and debug geometry.
//
// Push constants carry a 4-component linear RGBA color and a final opacity
// multiplier in addition to the MVP matrix.  No texture bindings.  Vertex
// color from the MeshVertex buffer is multiplied with the material color so
// solid-color override (color=[1,1,1,1]) is a no-op.
//
// Push layout (96 bytes):
//   offset  0 : row_major float4x4  model_view_projection  (64 bytes)
//   offset 64 : float4              material_color          (16 bytes)
//   offset 80 : float               opacity                 ( 4 bytes)
// Bytes 84–95 are unused CPU-side padding; the shader block ends at byte 84.
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/flatColor.vert.slang
// Adaptation: push layout extended with material_color; uv passthrough added
// for forward compatibility with textured overlay.

#include "material_common.slang"

struct FlatColorPush {
	row_major float4x4 model_view_projection;
	float4 material_color;
	float  opacity;
};

[[vk::push_constant]] FlatColorPush pc;

[shader("vertex")]
SimpleVertexOutput main(MeshVertexInput input) {
	SimpleVertexOutput output;
	output.sv_position = mul(float4(input.position, 1.0), pc.model_view_projection);
	// Multiply vertex color by material color so geometry authored with white
	// vertex colors becomes the solid material color.
	float4 combined = input.color * pc.material_color;
	output.color = float4(combined.rgb, combined.a * pc.opacity);
	output.uv = input.uv;
	return output;
}
