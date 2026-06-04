
struct FrameData
{
    float4x4 view_proj;
    // float3 camera_pos;
};

struct ObjectData
{
    float4x4 transform;
    float3x3 normal_transform;
};

struct MaterialData
{
    float4 albedo_factor;
    float3 emissive_factor;
    float metallic_factor;
    float roughness_factor;
    float alpha_cutoff;
    int base_color_texture;
    int metallic_roughness_texture;
    int normal_texture;
    int occlusion_texture;
    int emissive_texture;
    uint albedo_sampler_index;
    uint metallic_roughness_sampler_index;
    uint normal_sampler_index;
    uint occlusion_sampler_index;
    uint emissive_sampler_index;
};

struct PushConstants
{
    FrameData *frame;
    ObjectData *objects;
    MaterialData *materials;
    uint object_id;
    uint material_id;
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
    float3 color;
    float3 normal;
};

[shader("vertex")]
VertexOutput vertex(VertexInput input, uniform PushConstants data)
{
    VertexOutput res;

    // let object = push.objects[push.object_id];
    let world_pos = mul(data.objects[data.object_id].transform, float4(input.position, 1.0));

    res.color = input.color;
    res.position = mul(data.frame.view_proj, world_pos);
    // res.position = float4(input.position, 1.0);
    res.normal = input.normal;
    return res;
}

[shader("fragment")]
float4 fragment(VertexOutput in, uniform PushConstants data) : SV_Target
{
    let material = data.materials[data.material_id];

    let albedo = in.color * material.albedo_factor.rgb;
    return float4(albedo, 1.0);
}
