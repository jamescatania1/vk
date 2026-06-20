import "utils/common";

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
    let pos = uint2(float2(data.frame.size) * in.uv + 0.5);

    if (data.debug_view == 0)
    {
        let color = tex_hdr_output.Load(int3(pos, 0)).rgb;
        return float4(color, 1.0);
    }
    else
    {
        let color = debug(pos, data);
        return float4(color, 1.0);
    }
}

float3 debug(uint2 pos, PushConstants data)
{
    let frag = load_fragment(pos, data.frame);

    switch (data.debug_view)
    {
    case 1:
        return frag.depth;
    case 2:
        return frag.normal;
    case 3:
        return frag.roughness;
    case 4:
        return frag.metallic;
    case 5:
    case 6:
        return tex_hdr_output.Load(int3(pos, 0)).rgb;
    default:
        return float3(shadowmaps[data.debug_view - 6].Sample(frag.uv).r);
    };
}
