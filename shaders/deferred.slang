import "utils/common";

struct PushConstants
{
    FrameData *frame;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform PushConstants data)
{
    uint2 pos = id.xy;

    if (any(pos >= data.frame.size))
    {
        return;
    }

    let frag = load_fragment(pos, data.frame);

    float3 view_dir = normalize(data.frame.camera_pos - frag.ws_pos);
    float3 half_dir = normalize(view_dir + data.frame.light_dir);

    let ndh = max(dot(frag.normal, half_dir), 0.0);
    let vdh = max(dot(view_dir, half_dir), 0.0);
    let ndv = max(dot(frag.normal, view_dir), 0.0);
    let ndl = max(dot(frag.normal, data.frame.light_dir), 0.0);

    let f0 = lerp(float3(0.04), frag.albedo, frag.metallic);
    let ks = fresnel(f0, vdh);
    let kd = (1.0 - ks) * (1.0 - frag.metallic);

    let ndf = normal_distribution(ndh, frag.roughness);
    let geom = geometry_smith(ndv, ndl, frag.roughness);

    let diffuse = kd * frag.albedo / PI;
    let specular = ks * ndf * geom / max(4.0 * ndv * ndl, 0.000001);
    let bsdf = diffuse + specular;

    let direct = bsdf * float3(1.0) * 6.0 * ndl;
    let indirect = 0.5 * frag.albedo;

    float shadow = compute_shadow(frag, data);

    // let ao = compute_ao(frag, data);
    // let ao = gtao(frag, data);
    let ao = tex_ao[data.frame.swap_b].SampleLevel(sampler_linear, frag.uv, 0.0);

    var res = direct * shadow + indirect * ao;

    // res = float3(shadow * 0.5 + 0.5);
    // res[cascade_index] *= 0.8;

    if (data.frame.debug_view == 7)
    {
        res = float3(ao);
    }
    else if (data.frame.debug_view == 8)
    {
        res = float3(shadow);
    }
    tex_hdr_output_storage.Store(pos, float4(res, 1.0));
}

float compute_shadow(Fragment frag, PushConstants data)
{
    let view_pos = mul(data.frame.view, float4(frag.ws_pos, 1.0));
    let depth = abs(view_pos.z);

    var shadow = 0.0;
    var sampled_cascade = false;
    var cascade_index = CASCADES - 1;
    for (int i = 0; i < CASCADES; i++)
    {
        let cascade = &data.frame.cascades[i];
        if (depth < cascade.far && depth > cascade.near)
        {
            let cascade_shadow = sample_cascade(frag.ws_pos, cascade, i);

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
    return shadow;
}

float sample_cascade(float3 world_pos, Cascade *cascade, int cascade_index)
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
