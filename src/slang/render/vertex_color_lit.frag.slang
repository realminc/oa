// Donor path: oacpp/source/cpp/lib/oa/render/shader/graphics/vertexColorLit.frag.slang
// Adaptation: none — shader body is identical.

struct FragmentInput {
	float4 color : COLOR0;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0 {
	return input.color;
}
