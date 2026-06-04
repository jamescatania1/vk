
struct FrameData
{
    float4x4 view_proj;
    // float3 camera_pos;
};

// struct ObjectData
// {
//     float4x4 transform;
//     float3x3 normal_transform;
// };

// struct PushConstants
// {
//     FrameData* frame;
//     ObjectData* objects;
//     uint object_id;
//     uint material_id;
// };

// [vk::push_constant]
// uniform PushConstants push;

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
    float3 color;
    float3 normal;
};

[shader("vertex")]
VertexOutput vertex(VertexInput input, uniform FrameData *shader_data)
{
    VertexOutput res;

    // let object = push.objects[push.object_id];
    // let world_pos = mul(object.transform, float4(input.position, 1.0));

    res.color = input.color;
    res.position = mul(shader_data.view_proj, float4(input.position, 1.0));
    res.position = float4(input.position, 1.0);
    res.normal = input.normal;
    return res;
}

[shader("fragment")]
float4 fragment(VertexOutput in) : SV_Target
{
    return float4(in.normal, 1.0);
}
