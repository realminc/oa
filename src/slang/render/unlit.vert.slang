// Unlit vertex shader — Tier 2 material, image and video display.
//
// Transforms position with the MVP matrix and forwards UV (with scale/offset
// applied) and tint color to the fragment stage.
//
// Push layout (96 bytes):
//   offset  0 : row_major float4x4  model_view_projection  (64 bytes)
//   offset 64 : float4              tint                    (16 bytes)
//   offset 80 : float2              uv_scale                ( 8 bytes)
//   offset 88 : float2              uv_offset               ( 8 bytes)
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/unlit.vert.slang

#include "material_common.slang"

struct UnlitPush {
	row_major float4x4 model_view_projection;
	float4 tint;
	float2 uv_scale;
	float2 uv_offset;
};

[[vk::push_constant]] UnlitPush pc;

[shader("vertex")]
SimpleVertexOutput main(MeshVertexInput input) {
	SimpleVertexOutput output;
	output.sv_position = mul(float4(input.position, 1.0), pc.model_view_projection);
	output.uv    = apply_uv_transform(input.uv, pc.uv_scale, pc.uv_offset);
	output.color = pc.tint;
	return output;
}
