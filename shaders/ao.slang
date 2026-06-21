import "utils/common";

struct PushConstants
{
    FrameData *frame;
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

    // let frag = load_fragment(pos, data.frame);

    let ao = gtao(id.xy, data.frame);

    tex_ao_storage[data.frame.swap_a].Store(id.xy, ao);
}

const static uint2[4] OFFSETS = {
    uint2(0, 0),
    uint2(0, 1),
    uint2(1, 0),
    uint2(1, 1),
};

float gtao(uint2 id, FrameData *frame)
{
    // let pos = id * 2 + OFFSETS[frame.frame_id % 4];
    let pos = id * 2;
    let ctr_uv = (float2(pos) + 0.5) * frame.texel_size;

    let ws_normal = GBufferFragment(tex_gbuffer.Load(int3(pos, 0))).normal;
    let vs_normal = normalize(mul((float3x3)frame.view, ws_normal));

    var ctr_depth = tex_depth.Load(int3(pos, 0));
    let vs_ctr_pos = recover_view_pos(ctr_uv, ctr_depth, frame);
    let vs_ctr_dir = normalize(-vs_ctr_pos);
    let vs_texel_size = abs(vs_ctr_pos.z * 0.99999) * frame.ndc_view_pixel_size;

    let screen_radius = frame.ao_radius / vs_texel_size.x;
    let min_s = 1.3 / screen_radius;

    let falloff_range = frame.ao_radius * frame.ao_falloff_range;
    let falloff_start = frame.ao_radius * (1.0 - frame.ao_falloff_range);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_start / falloff_range + 1.0;

    let noise = hash22(id);

    var visibility = 0.0;

    for (int slice = 0; slice < frame.ao_slices; slice++)
    {
        let phi = (float(slice) + noise.x) * PI / float(frame.ao_slices);
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        let omega = float2(cos(phi), -sin(phi)) * screen_radius;

        let direction = float3(cos_phi, sin_phi, 0.0);
        let ortho_direction = direction - dot(direction, vs_ctr_dir) * vs_ctr_dir;
        let axis = normalize(cross(ortho_direction, vs_ctr_dir));

        let proj_normal = vs_normal - axis * dot(vs_normal, axis);
        var proj_normal_length = length(proj_normal);
        let cos_normal = saturate(dot(proj_normal, vs_ctr_dir) / proj_normal_length);
        let sign_normal = sign(dot(ortho_direction, proj_normal));
        let n = sign_normal * fast_acos(cos_normal);

        let min_horizon_cos_0 = cos(n + PI * 0.5);
        let min_horizon_cos_1 = cos(n - PI * 0.5);

        var horizon_cos_0 = min_horizon_cos_0;
        var horizon_cos_1 = min_horizon_cos_1;

        [unroll]
        for (int sample = 0; sample < frame.ao_samples; sample++)
        {
            let sample_base_noise = float(slice + sample * frame.ao_samples) * 0.6180339887498948482;
            let sample_noise = fract(sample_base_noise + noise.y);

            var s = (float(sample) + sample_noise) / float(frame.ao_samples);
            s = pow(s, frame.ao_sample_distribution_power);
            s += min_s; // avoids sampling center pixel

            var offset = int2(round(s * omega));

            let sample_pos_0 = clamp(int2(pos) + offset, 0, frame.size - 1);
            let sample_pos_1 = clamp(int2(pos) - offset, 0, frame.size - 1);

            let sample_depth_0 = tex_depth.Load(int3(sample_pos_0, 0));
            let sample_depth_1 = tex_depth.Load(int3(sample_pos_1, 0));

            let vs_sample_pos_0 = recover_view_pos((float2(sample_pos_0) + 0.5) * frame.texel_size, sample_depth_0, frame);
            let vs_sample_pos_1 = recover_view_pos((float2(sample_pos_1) + 0.5) * frame.texel_size, sample_depth_1, frame);

            let sample_delta_0 = vs_sample_pos_0 - vs_ctr_pos;
            let sample_delta_1 = vs_sample_pos_1 - vs_ctr_pos;

            let sample_distance_0 = length(sample_delta_0);
            let sample_distance_1 = length(sample_delta_1);

            let sample_horizon_0 = sample_delta_0 / sample_distance_0;
            let sample_horizon_1 = sample_delta_1 / sample_distance_1;

            let falloff_base_0 = length(float3(sample_delta_0.x, sample_delta_0.y, sample_delta_0.z * (1.0 + frame.ao_thin_occluder_compensation)));
            let falloff_base_1 = length(float3(sample_delta_1.x, sample_delta_1.y, sample_delta_1.z * (1.0 + frame.ao_thin_occluder_compensation)));

            let weight_0 = saturate(falloff_base_0 * falloff_mul + falloff_add);
            let weight_1 = saturate(falloff_base_1 * falloff_mul + falloff_add);

            var sample_horizon_cos_0 = dot(vs_ctr_dir, sample_horizon_0);
            var sample_horizon_cos_1 = dot(vs_ctr_dir, sample_horizon_1);

            sample_horizon_cos_0 = lerp(min_horizon_cos_0, sample_horizon_cos_0, weight_0);
            sample_horizon_cos_1 = lerp(min_horizon_cos_1, sample_horizon_cos_1, weight_1);

            horizon_cos_0 = max(horizon_cos_0, sample_horizon_cos_0);
            horizon_cos_1 = max(horizon_cos_1, sample_horizon_cos_1);
        }

        // apparently fixes overdarkining on high slopes
        // trust those smarter than myself
        proj_normal_length = lerp(proj_normal_length, 1.0, 0.05);

        let h_0 = -fast_acos(horizon_cos_1);
        let h_1 = fast_acos(horizon_cos_0);

        let arc_0 = 0.25 * (cos_normal + 2.0 * h_0 * sin(n) - cos(2.0 * h_0 - n));
        let arc_1 = 0.25 * (cos_normal + 2.0 * h_1 * sin(n) - cos(2.0 * h_1 - n));

        visibility += proj_normal_length * (arc_0 + arc_1);
    }
    visibility /= float(frame.ao_slices);
    visibility = pow(visibility, frame.ao_final_value_power);
    visibility = max(0.03, visibility);

    return saturate(visibility);
}

float fast_sqrt(float x)
{
    return asfloat(0x1fbd1df5 + (asint(x) >> 1));
}

float fast_acos(float x)
{
    x = clamp(x, -1.0, 1.0);
    let x_abs = abs(x);
    var res = -0.156583 * x_abs + (0.5 * PI);
    res *= fast_sqrt(1.0 - x_abs);
    return x >= 0.0 ? res : PI - res;
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
