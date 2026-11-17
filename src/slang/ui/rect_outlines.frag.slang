// UI raster batched normalized rectangle borders — fragment stage.
//
// Per-pixel counterpart of the draw_rect_outlines.slang border layers: a
// pixel is written when it lies inside the record rect within `border` of an
// edge — the same set the compute path's layered top/bottom/left/right stores
// covered. The pipeline is replace (blend-disabled) to match `store_pixel`.

struct OutlineVaryings {
	float4 position : SV_Position;
	nointerpolation float4 rect : OUTLINE_RECT;
	nointerpolation float4 color : OUTLINE_COLOR;
	nointerpolation uint border : OUTLINE_BORDER;
};

[shader("fragment")]
float4 main(OutlineVaryings input) : SV_Target0 {
	int x = int(input.position.x);
	int y = int(input.position.y);
	int x0 = int(input.rect.x);
	int y0 = int(input.rect.y);
	int x1 = int(input.rect.z);
	int y1 = int(input.rect.w);
	int border = int(input.border);
	if (x < x0 || x >= x1 || y < y0 || y >= y1) discard;
	bool covered = x - x0 < border
		|| x1 - 1 - x < border
		|| y - y0 < border
		|| y1 - 1 - y < border;
	if (!covered) discard;
	return input.color;
}
