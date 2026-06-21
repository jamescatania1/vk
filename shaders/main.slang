import "utils/common";

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
    float4 cur_cs_pos;
    float4 prev_cs_pos;
    float3 world_pos;
    float3 normal;
    float4 tangent;
    float3 color;
    float2 uv;
};

[shader("vertex")]
VertexOutput vertex(VertexInput input, uniform PushConstants data)
{
    VertexOutput res;

    let object = data.objects[data.object_id];

    let cur_world_pos = mul(object.transform, float4(input.position, 1.0));
    let cur_cs_pos = mul(data.frame.view_proj, cur_world_pos);

    let prev_world_pos = cur_world_pos; // use prev object matrix once dynamic
    let prev_cs_pos = mul(data.frame.prev_view_proj, prev_world_pos);

    res.position = cur_cs_pos;

    res.cur_cs_pos = cur_cs_pos;
    res.prev_cs_pos = prev_cs_pos;
    res.world_pos = cur_world_pos.xyz;

    res.normal = mul(object.normal_transform, input.normal);
    res.tangent = float4(normalize(mul((float3x3)object.transform, input.tangent.xyz)), input.tangent.w);
    res.color = input.color;
    res.uv = input.uv;

    return res;
}

[shader("fragment")]
uint4 fragment(VertexOutput in, uniform PushConstants data) : SV_Target
{
    let material = data.materials[data.material_id];
    let surface = gather_surface(in, material);

    if (surface.alpha < material.alpha_cutoff)
    {
        discard;
    }

    GBufferFragment res;
    res.albedo = surface.albedo;
    res.metallic = surface.metallic;
    res.roughness = surface.roughness;
    res.normal = surface.normal;
    res.geometry_normal = surface.geometry_normal;
    res.velocity = motion_vector(in);
    return res.packed;
}

float2 motion_vector(VertexOutput in)
{
    let cur_ndc = in.cur_cs_pos.xy / in.cur_cs_pos.w;
    let prev_ndc = in.prev_cs_pos.xy / in.prev_cs_pos.w;

    let cur_uv = cur_ndc * 0.5 + 0.5;
    let prev_uv = prev_ndc * 0.5 + 0.5;

    return cur_uv - prev_uv;
}

struct SurfaceProperties
{
    float3 albedo;
    float alpha;
    float metallic;
    float roughness;
    float3 normal;
    float3 geometry_normal;
};
SurfaceProperties gather_surface(VertexOutput in, MaterialData material)
{
    var res : SurfaceProperties;

    res.albedo = in.color * material.albedo_factor.rgb;
    res.alpha = material.albedo_factor.a;
    if (material.base_color_texture >= 0)
    {
        let tex = textures[material.base_color_texture];
        let color = tex.Sample(samplers[material.albedo_sampler_index], in.uv);
        res.alpha *= color.a;
        res.albedo *= color.rgb;
    }

    res.metallic = material.metallic_factor;
    res.roughness = material.roughness_factor;
    if (material.metallic_roughness_texture >= 0)
    {
        let tex = textures[material.metallic_roughness_texture];
        let mr = tex.Sample(samplers[material.metallic_roughness_sampler_index], in.uv);
        res.metallic *= mr.b;
        res.roughness *= mr.g;
    }
    res.roughness = clamp(res.roughness, 0.04, 1.0);
    res.metallic = saturate(res.metallic);

    res.geometry_normal = normalize(in.normal);
    res.normal = res.geometry_normal;
    if (material.normal_texture >= 0)
    {
        let tex = textures[material.normal_texture];
        let ts_normal = normalize(tex.Sample(samplers[material.normal_sampler_index], in.uv).rgb * 2.0 - 1.0);

        var tangent = normalize(in.tangent.xyz);
        tangent = normalize(tangent - res.normal * dot(res.normal, tangent));
        var bitangent = normalize(cross(res.normal, tangent)) * in.tangent.w;

        res.normal = normalize(ts_normal.x * tangent + ts_normal.y * bitangent + ts_normal.z * res.normal);
    }

    return res;
}
