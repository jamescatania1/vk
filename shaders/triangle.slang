struct FrameUniform
{
    float4x4 view_proj;
    float3 camera_pos;
};

struct ObjectData
{
    float4x4 transform;
    float3x3 normal_transform;
};

[[vk::binding(0, 0)]]
ConstantBuffer<FrameUniform> frame;

[[vk::binding(1, 0)]]
StructuredBuffer<ObjectData> objects;

struct Material
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

[[vk::binding(0, 1)]]
StructuredBuffer<Material> materials;

[[vk::binding(1, 1)]]
Texture2D<float4> textures[256];

[[vk::binding(2, 1)]]
SamplerState samplers[16];

struct PushConstants
{
    uint object_id;
    uint material_id;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> push;

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]]
    float3 color;
    [[vk::location(1)]]
    float3 normal;
    [[vk::location(2)]]
    float4 tangent;
    [[vk::location(3)]]
    float3 world_pos;
    [[vk::location(4)]]
    float2 uv;
};

[shader("vertex")]
VertexOutput vertex([[vk::location(0)]] float3 position: POSITION0, [[vk::location(1)]] float3 normal: POSITION1,
                    [[vk::location(2)]] float4 tangent: POSITION2,
                    [[vk::location(3)]] float3 color: POSITION3, [[vk::location(4)]] float2 uv: TEXCOORD0)

{
    VertexOutput res;

    let object = objects[push.object_id];
    let world_pos = mul(object.transform, float4(position, 1.0));

    res.color = color;
    res.position = mul(frame.view_proj, world_pos);
    res.normal = mul(object.normal_transform, normal);
    res.tangent = float4(normalize(mul((float3x3)object.transform, tangent.xyz)), tangent.w);
    res.world_pos = world_pos.xyz;
    res.uv = uv;
    return res;
}

[shader("fragment")]
float4 fragment(VertexOutput in) : SV_Target
{
    let material = materials[push.material_id];
    let surface = gather_surface(in, material);

    if (surface.alpha < material.alpha_cutoff)
    {
        // discard;
        return float4(0.0, 0.0, 0.0, 0.0);
    }

    float3 light_dir = normalize(float3(0.2, -0.1, 1.4));
    float3 view_dir = normalize(frame.camera_pos - in.world_pos);
    float3 half_dir = normalize(view_dir + light_dir);

    let ndh = max(dot(surface.normal, half_dir), 0.0);
    let vdh = max(dot(view_dir, half_dir), 0.0);
    let ndv = max(dot(surface.normal, view_dir), 0.0);
    let ndl = max(dot(surface.normal, light_dir), 0.0);

    let f0 = lerp(float3(0.04), surface.albedo, surface.metallic);
    let ks = fresnel(f0, vdh);
    let kd = (1.0 - ks) * (1.0 - surface.metallic);

    let ndf = normal_distribution(ndh, surface.roughness);
    let geom = geometry_smith(ndv, ndl, surface.roughness);

    let diffuse = kd * surface.albedo / PI;
    let specular = ks * ndf * geom / max(4.0 * ndv * ndl, 0.000001);
    let bsdf = diffuse + specular;

    let direct = bsdf * float3(1.0) * 6.0 * ndl;
    let indirect = 0.5 * surface.albedo;

    var res = direct + indirect;

    return float4(res, surface.alpha);
}

struct SurfaceProperties
{
    float3 albedo;
    float alpha;
    float metallic;
    float roughness;
    float3 normal;
};
SurfaceProperties gather_surface(VertexOutput in, Material material)
{
    var res : SurfaceProperties;

    res.albedo = in.color * material.albedo_factor.rgb;
    res.alpha = material.albedo_factor.a;
    if (material.base_color_texture >= 0)
    {
        let tex = textures[material.base_color_texture];
        let sampler = samplers[material.albedo_sampler_index];
        let color = tex.Sample(sampler, in.uv);
        res.alpha *= color.a;
        res.albedo *= color.rgb;
    }

    res.metallic = material.metallic_factor;
    res.roughness = material.roughness_factor;
    if (material.metallic_roughness_texture >= 0)
    {
        let tex = textures[material.metallic_roughness_texture];
        let sampler = samplers[material.metallic_roughness_sampler_index];
        let mr = tex.Sample(sampler, in.uv);
        res.metallic *= mr.b;
        res.roughness *= mr.g;
    }
    res.roughness = clamp(res.roughness, 0.04, 1.0);
    res.metallic = saturate(res.metallic);

    res.normal = normalize(in.normal);
    if (material.normal_texture >= 0)
    {
        let tex = textures[material.normal_texture];
        let sampler = samplers[material.normal_sampler_index];
        let ts_normal = normalize(tex.Sample(sampler, in.uv).rgb * 2.0 - 1.0);

        var tangent = normalize(in.tangent.xyz);
        tangent = normalize(tangent - res.normal * dot(res.normal, tangent));
        var bitangent = normalize(cross(res.normal, tangent)) * in.tangent.w;

        res.normal = normalize(ts_normal.x * tangent + ts_normal.y * bitangent + ts_normal.z * res.normal);
    }

    return res;
}

static const float PI = 3.14159265359;

float normal_distribution(float ndh, float roughness)
{
    let alpha = roughness * roughness;
    let alpha_sq = alpha * alpha;
    let d = ndh * ndh * (alpha_sq - 1.0) + 1.0;
    return (alpha_sq) / max((PI * d * d), 0.000001);
}

float geometry_smith(float ndv, float ndl, float roughness)
{
    return geometry_ggx(ndv, roughness) * geometry_ggx(ndl, roughness);
}

float geometry_ggx(float ndx, float roughness)
{
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let d = ndx * (1.0 - k) + k;
    return ndx / max(d, 0.000001);
}

float3 fresnel(float3 f0, float vdh)
{
    return f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
}

float3 fresnel_roughness(float3 f0, float vdh, float roughness)
{
    return f0 + (max(1.0 - roughness, f0) - f0) * pow(1.0 - vdh, 5.0);
}
