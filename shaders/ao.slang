import "utils/common";

struct PushConstants
{
    FrameData *frame;
    uint debug_view;
    uint _pad;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform PushConstants data)
{
    uint2 pos = id.xy * 2;

    if (any(pos >= data.frame.size))
    {
        return;
    }

    let frag = load_fragment(pos, data.frame);

    let ao = gtao(id.xy, frag, data);

    tex_ao_storage.Store(id.xy, ao);
}

float gtao(uint2 id, Fragment frag, PushConstants data)
{
    const uint slice_count = 4;
    const uint sample_count = 8;
    const float falloff = 0.5;

    let vs_normal = normalize(mul((float3x3)data.frame.view, frag.normal));
    let vs_ctr_pos = recover_view_pos(frag.uv, frag.depth, data.frame);
    let vs_ctr_dir = normalize(-vs_ctr_pos);

    let vs_texel_size = abs(vs_ctr_pos.z) * data.frame.ndc_view_pixel_size;

    let radius = data.frame.ao_radius / vs_texel_size.x;
    let falloff_start = data.frame.ao_radius * (1.0 - falloff);

    let noise = hash22(id);

    var visibility = 0.0;

    [unroll]
    for (int slice = 0; slice < slice_count; slice++)
    {
        let phi = (float(slice) + noise.x) * PI / float(slice_count);
        let omega = float2(cos(phi), sin(phi));

        let direction = float3(omega, 0.0);
        let ortho_direction = direction - dot(direction, vs_ctr_dir) * vs_ctr_dir;
        let axis = normalize(cross(direction, vs_ctr_dir));
        let proj_normal = vs_normal - axis * dot(vs_normal, axis);

        let sgn = sign(dot(ortho_direction, proj_normal));
        let cos_n = saturate(dot(proj_normal, vs_ctr_dir) / length(proj_normal));
        let n = sgn * acos(cos_n);

        [unroll]
        for (int side = 0; side <= 1; side++)
        {
            var horizon_cos = -1.0;

            [unroll]
            for (int sample = 0; sample < sample_count; sample++)
            {
                let step_base_noise = float(slice + sample * sample_count) * 0.6180339887498948482;
                let step_noise = fract(step_base_noise + noise.y);

                let s = (float(sample) + step_noise) / float(sample_count);
                let offset = (side == 0 ? -1.0 : 1.0) * s * radius * float2(omega.x, -omega.y);
                if (max(abs(offset.x), abs(offset.y)) <= 1.0)
                {
                    continue;
                }

                let uv = frag.uv + offset * data.frame.texel_size;
                if (any(uv < 0.0) || any(uv >= 1.0))
                {
                    continue;
                }
                let pos = clamp(uint2(uv * float2(data.frame.size) + 0.5), 0, data.frame.size - 1);

                let hs_sample_depth = tex_depth.Load(int3(pos, 0));
                let vs_sample_pos = recover_view_pos(uv, hs_sample_depth, data.frame);

                let sample_delta = vs_sample_pos - vs_ctr_pos;
                let sample_distance = length(sample_delta);
                let sample_weight = 1.0 - saturate((sample_distance - falloff_start) / max(data.frame.ao_radius - falloff_start, 1e-5));

                let sample_horizon = normalize(sample_delta);

                let sample_horizon_cos = lerp(-1.0, dot(sample_horizon, vs_ctr_dir), sample_weight);
                horizon_cos = max(horizon_cos, sample_horizon_cos);
            }

            let horizon_angle = n + clamp((-1.0 + 2.0 * float(side)) * acos(horizon_cos) - n, -PI * 0.5, PI * 0.5);
            visibility += length(proj_normal) * (cos_n + 2.0 * horizon_angle * sin(n) - cos(2.0 * horizon_angle - n)) * 0.25;
        }
    }
    visibility /= float(slice_count);

    return saturate(visibility);
}

uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float hash_to_float(uint x)
{
    return float(x & 0x00ffffffu) / float(0x01000000u);
}

float2 hash22(uint2 p)
{
    uint h0 = hash_u32(p.x * 1664525u + p.y * 1013904223u);
    uint h1 = hash_u32(h0 ^ 0x9e3779b9u);
    return float2(hash_to_float(h0), hash_to_float(h1));
}
