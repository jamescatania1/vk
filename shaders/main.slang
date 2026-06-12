
static const uint CASCADES = 4;

[[vk::binding(0, 0)]]
Texture2D textures[];

[[vk::binding(1, 0)]]
SamplerState samplers[];

[[vk::binding(2, 0)]]
Sampler2D shadowmaps[CASCADES];

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
    float2 inv_texel_size;
    float near;
    float far;
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
    let world_pos = mul(object.transform, float4(input.position, 1.0));

    res.color = input.color;
    res.world_pos = world_pos.xyz;
    res.position = mul(data.frame.view_proj, world_pos);
    res.normal = mul(object.normal_transform, input.normal);
    res.tangent = float4(normalize(mul((float3x3)object.transform, input.tangent.xyz)), input.tangent.w);
    res.uv = input.uv;
    return res;
}

[shader("fragment")]
float4 fragment(VertexOutput in, uniform PushConstants data) : SV_Target
{
    let material = data.materials[data.material_id];
    let surface = gather_surface(in, material);

    if (surface.alpha < material.alpha_cutoff)
    {
        discard;
    }

    float3 view_dir = normalize(data.frame.camera_pos - in.world_pos);
    float3 half_dir = normalize(view_dir + data.frame.light_dir);

    let ndh = max(dot(surface.normal, half_dir), 0.0);
    let vdh = max(dot(view_dir, half_dir), 0.0);
    let ndv = max(dot(surface.normal, view_dir), 0.0);
    let ndl = max(dot(surface.normal, data.frame.light_dir), 0.0);

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

    let view_pos = mul(data.frame.view, float4(in.world_pos, 1.0));
    let depth = abs(view_pos.z);

    var shadow = 0.0;
    var sampled_cascade = false;
    var cascade_index = CASCADES - 1;
    for (int i = 0; i < CASCADES; i++)
    {
        let cascade = &data.frame.cascades[i];
        if (depth < cascade.far && depth > cascade.near)
        {
            let cascade_shadow = sample_shadow(in.world_pos, cascade, i);

            if (sampled_cascade)
            {
                let blend_factor = (depth - cascade.near) / (data.frame.cascades[i - 1].far - cascade.near);
                shadow = shadow * blend_factor + cascade_shadow * (1.0 - blend_factor);
                break;
            }
            else
            {
                shadow += cascade_shadow;
            }
            sampled_cascade = true;
        }
    }

    var res = direct * shadow + indirect;
    // res = float3(shadow * 0.5 + 0.5);
    // res[cascade_index] *= 0.8;

    return float4(res, surface.alpha);
}

float sample_shadow(float3 world_pos, Cascade *cascade, int cascade_index)
{
    let shadowmap = shadowmaps[cascade_index];

    uint width;
    uint height;
    shadowmap.GetDimensions(width, height);
    let dimensions = float2(width, height);
    let texel_size = 1.0 / dimensions;

    let light_pos = mul(cascade.view_proj, float4(world_pos, 1.0));
    let ndc = light_pos.xyz / light_pos.w;
    let uv = ndc.xy * 0.5 + 0.5;
    let depth = ndc.z;

    if (any(uv < float2(0.0)) || any(uv > float2(1.0)) || depth < 0.0 || depth > 1.0)
    {
        return 1.0;
    }

    let world_radius = 0.01;
    let texel_radius = clamp(int2(world_radius * cascade.inv_texel_size), 1, 16);

    var res = 0.0;
    var sample_count = 0.0;
    [unroll]
    for (int x = -texel_radius.x; x <= texel_radius.x; x += 2)
    {
        [unroll]
        for (int y = -texel_radius.y; y <= texel_radius.y; y += 2)
        {
            let offset = float2(x, y) * texel_size;
            let sample_pos = uv + offset;

            let samples = shadowmap.GatherRed(sample_pos);

            [unroll]
            for (int i = 0; i < 4; i++)
            {
                let bias = 0.0001;
                res += depth < samples[i] + bias ? 1.0 : 0.0;
            }
            sample_count += 4.0;
        }
    }
    res /= sample_count;
    return res;
}

struct SurfaceProperties
{
    float3 albedo;
    float alpha;
    float metallic;
    float roughness;
    float3 normal;
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

    res.normal = normalize(in.normal);
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
