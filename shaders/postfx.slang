static const uint CASCADES = 3;

[[vk::binding(2, 0)]]
Sampler2D shadowmaps[CASCADES];

[[vk::binding(3, 0)]]
Texture2D color_output;

struct FrameData
{
    float4x4 view_proj;
    float4x4 view;
    float4x4 proj;
    float3 camera_pos;
    float3 light_dir;
    Cascade cascades[CASCADES];
};

struct Cascade
{
    float4x4 view_proj;
    float2 texel_size;
    float near;
    float far;
};

struct PushConstants
{
    FrameData *frame;
    uint debug_view;
    uint _pad;
};

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
float4 fragment(VertexOutput in, uniform PushConstants data) : SV_Target
{
    uint width;
    uint height;
    color_output.GetDimensions(width, height);

    let pos = uint2(float2(width, height) * in.uv + 0.5);

    var color = color_output.Load(uint3(pos, 0)).rgb;

    if (data.debug_view > 0 && data.debug_view <= CASCADES)
    {
        color = float3(shadowmaps[data.debug_view - 1].Sample(in.uv).r);
    }

    return float4(color, 1.0);
}
