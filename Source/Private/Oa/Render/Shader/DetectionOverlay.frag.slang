struct FragmentInput {
	float4 color : COLOR0;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0 {
	return input.color;
}
