// Unlit fragment shader — Tier 2 material, image and video display.
//
// Samples the base-color texture (binding 0) and multiplies with the tint
// color forwarded from the vertex stage.  When no texture is bound, the
// caller binds the shared 1×1 white default texture so the tint alone is
// returned.
//
// Descriptor set 0, binding 0 : Sampler2D base_color_texture (sRGB RGBA8)
//
// Donor: oacpp/source/cpp/lib/oa/render/shader/graphics/unlit.frag.slang

#include "material_common.slang"

[[vk::binding(0, 0)]] Sampler2D base_color_texture;

[shader("fragment")]
float4 main(SimpleVertexOutput input) : SV_Target0 {
	float4 texel = base_color_texture.Sample(input.uv);
	return texel * input.color;
}
