import "utils/common";

struct PushConstants
{
    FrameData *frame;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID, uniform PushConstants data)
{
    int2 pos = id.xy * 2;

    let d_00 = tex_depth.Load(int3(pos + int2(0, 0), 0));
    let d_01 = tex_depth.Load(int3(pos + int2(0, 1), 0));
    let d_10 = tex_depth.Load(int3(pos + int2(1, 0), 0));
    let d_11 = tex_depth.Load(int3(pos + int2(1, 1), 0));

    let max_depth = min(min(min(d_00, d_01), d_10), d_11);

    tex_depth_half_storage.Store(id.xy, max_depth);
}
