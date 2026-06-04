[[vk::binding(0, 0)]]
[[vk::image_format("rgba8")]]
RWTexture2D<float4> tex;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID) {
	uint w;
	uint h;
	tex.GetDimensions(w, h);

	float2 uv = (float2(id.xy) + 0.5) / float2(w, h);
	float2 ndc = (uv - 0.5) * 2.0 - float2(1.0, 0.0);

	float2 z = 0.0;
	float i;
	for (i = 0.0; i < 1.0; i += 0.0005) {
		z = float2(z.x * z.x - z.y * z.y + ndc.x, z.y * z.x + z.x * z.y + ndc.y);

		if (length(z) > 4.0) {
			break;
		}
	}

	tex[int2(id.xy)] = float4(i, i, i, 1.0);
}
