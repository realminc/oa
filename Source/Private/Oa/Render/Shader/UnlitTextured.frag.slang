// UnlitTextured.frag.slang — Minimal fragment shader for textured quad rendering.
//
// Input:   UV from vertex stage
// Output:  RGBA color
// Push:    tint color

struct FragmentInput {
	float2 uv : TEXCOORD0;
};

struct PushConstants {
	float4x4 mvp;
	float4 tint;
};

[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] Texture2D<float4> texture_image;
[[vk::binding(1, 0)]] SamplerState texture_sampler;

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0 {
	float4 color = texture_image.Sample(texture_sampler, input.uv);
	return color * pc.tint;
}
