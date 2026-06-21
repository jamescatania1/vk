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

    // let frag = load_fragment(pos, data.frame);

    let ao = gtao(id.xy, data.frame);

    tex_ao_storage.Store(id.xy, ao);
}
