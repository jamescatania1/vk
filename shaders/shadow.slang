import "utils/common";

struct PushConstants
{
    FrameData *frame;
    ObjectData *objects;
    MaterialData *materials;
    uint object_id;
    uint material_id;
    uint cascade_index;
    uint _pad;
};

struct VertexInput
{
    float3 position;
    float3 normal;
    float4 tangent;
    float3 color;
    float2 uv;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput vertex(VertexInput input, uniform PushConstants data)
{
    VertexOutput res;

    let object = data.objects[data.object_id];
    let world_pos = mul(object.transform, float4(input.position, 1.0));

    res.position = mul(data.frame.cascades[data.cascade_index].view_proj, world_pos);
    return res;
}

[shader("fragment")]
void fragment(VertexOutput in) : SV_Target
{
}
