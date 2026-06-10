static const uint CASCADES = 3;

[[vk::binding(2, 0)]]
Sampler2D shadowmaps[CASCADES];

[[vk::binding(3, 0)]]
Texture2D color_output;

static const float2 POSITIONS[3] = {
    float2(-1.0, -1.0),
    float2(3.0, -1.0),
    float2(-1.0, 3.0),
};
static const float2 UVS[3] = {
    float2(0.0, 0.0),
    float2(2.0, 0.0),
    float2(0.0, 2.0),
};

struct VertexOutput
{
    float4 position : SV_Position;
    float2 uv;
};

[shader("vertex")]
VertexOutput vertex(uint vertex_index: SV_VertexID)
{
    VertexOutput res;
    res.position = float4(POSITIONS[vertex_index], 0.0, 1.0);
    res.uv = UVS[vertex_index];
    return res;
}

[shader("fragment")]
float4 fragment(VertexOutput in) : SV_Target
{
    uint width;
    uint height;
    color_output.GetDimensions(width, height);

    let pos = uint2(float2(width, height) * in.uv + 0.5);

    var color = color_output.Load(uint3(pos, 0)).rgb;

    if (all(in.uv > float2(0.8)) && all(in.uv <= float2(1.0)))
    {
        uint shadow_width;
        uint shadow_height;
        shadowmaps[0].GetDimensions(shadow_width, shadow_height);

        let uv = (in.uv - 0.8) / 0.2;
        let pos = uint2(float2(shadow_width, shadow_height) * uv + 0.5);
        color = float3(shadowmaps[0].Load(uint3(pos, 0)).r);
    }

    return float4(color, 1.0);
}
