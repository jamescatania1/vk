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
    if (any(pos >= data.frame.size_half))
    {
        return;
    }

    let frag = load_fragment(pos * 2, data.frame);

    let cur_uv = frag.uv;
    let prev_uv = cur_uv - frag.velocity;

    let cur_ao = tex_ao.Load(int3(pos, 0));
    var prev_ao = tex_ao_history[data.frame.swap_a].SampleLevel(sampler_linear, prev_uv, 0.0);

    var min_ao = 0.0;
    var max_ao = 0.0;
    var mean = 0.0;
    var mean_sq = 0.0;
    var count = 0.0;

    [unroll]
    for (int x = -2; x <= 2; x++)
    {
        [unroll]
        for (int y = -2; y <= 2; y++)
        {
            let sample_pos = int2(pos) + int2(x, y);
            if (any(sample_pos < 0) || any(sample_pos >= data.frame.size_half))
            {
                continue;
            }

            let ao = tex_ao.Load(int3(sample_pos, 0));

            min_ao = min(min_ao, ao);
            max_ao = max(max_ao, ao);
            mean += ao;
            mean_sq += ao * ao;
            count += 1.0;
        }
    }
    mean /= count;
    mean_sq /= count;

    let variance = max(mean_sq - mean * mean, 0.0);
    let sigma = sqrt(variance);

    const float gamma = 1.3;

    let var_min = mean - gamma * sigma;
    let var_max = mean + gamma * sigma;

    prev_ao = clamp(prev_ao, var_min, var_max);

    if (any(prev_uv < 0.0) || any(prev_uv >= 1.0))
    {
        prev_ao = cur_ao;
    }

    let res = lerp(prev_ao, cur_ao, 0.2);
    tex_ao_history_storage[data.frame.swap_b].Store(pos, res);
}
