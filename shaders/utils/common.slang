module common;

public static const uint CASCADES = 4;

public static const float PI = 3.14159265359;

[[vk::binding(0, 0)]]
public Texture2D textures[];

[[vk::binding(1, 0)]]
public SamplerState samplers[];

[[vk::binding(2, 0)]]
public Sampler2D shadowmaps[CASCADES];

[[vk::binding(3, 0)]]
public Texture2D<float> tex_depth;

[[vk::binding(4, 0)]]
public Texture2D<uint4> tex_gbuffer;

[[vk::binding(5, 0)]]
public RWTexture2D<float4> tex_hdr_output_storage;

[[vk::binding(6, 0)]]
public Texture2D<float4> tex_hdr_output;

[[vk::binding(7, 0)]]
public SamplerState sampler_linear;

[[vk::binding(8, 0)]]
public RWTexture2D<float> tex_ao_storage;

[[vk::binding(9, 0)]]
public Texture2D<float> tex_ao;

public struct FrameData
{
    public float4x4 view_proj;
    public float4x4 view;
    public float4x4 proj;
    public float4x4 inv_view;
    public float4x4 inv_proj;
    public float4x4 inv_view_proj;
    public float3 camera_pos;
    public uint2 size;
    public float2 texel_size;
    public float2 ndc_view_pixel_size;
    public float3 light_dir;
    public Cascade cascades[CASCADES];
    public float ao_radius;
};

public struct Cascade
{
    public float4x4 view_proj;
    public float2 inv_texel_size;
    public float near;
    public float far;
};

public struct ObjectData
{
    public float4x4 transform;
    public float3x3 normal_transform;
};

public struct MaterialData
{
    public float4 albedo_factor;
    public float3 emissive_factor;
    public float metallic_factor;
    public float roughness_factor;
    public float alpha_cutoff;
    public int base_color_texture;
    public int metallic_roughness_texture;
    public int normal_texture;
    public int occlusion_texture;
    public int emissive_texture;
    public uint albedo_sampler_index;
    public uint metallic_roughness_sampler_index;
    public uint normal_sampler_index;
    public uint occlusion_sampler_index;
    public uint emissive_sampler_index;
};

public struct Fragment
{
    public float2 uv;
    public float3 albedo;
    public float3 normal;
    public float roughness;
    public float metallic;
    public float depth;
    public float3 ws_pos;
};

public Fragment load_fragment(uint2 pos, FrameData *frame)
{
    let depth = tex_depth.Load(int3(pos, 0));
    let gbuffer = GBufferFragment(tex_gbuffer.Load(int3(pos, 0)));

    let uv = (float2(pos) + 0.5) * frame.texel_size;
    let ws_pos = recover_world_pos(uv, depth, frame);

    Fragment res;
    res.uv = uv;
    res.albedo = gbuffer.albedo;
    res.metallic = gbuffer.metallic;
    res.normal = gbuffer.normal;
    res.roughness = gbuffer.roughness;
    res.depth = depth;
    res.ws_pos = ws_pos;
    return res;
}

float3 recover_world_pos(float2 uv, float depth, FrameData *frame)
{
    let ndc = uv * 2.0 - 1.0;

    let cs_pos = float4(ndc, depth, 1.0);
    let ws_pos_h = mul(frame.inv_view_proj, cs_pos);
    return ws_pos_h.xyz / ws_pos_h.w;
}

public float3 recover_view_pos(float2 uv, float depth, FrameData *frame)
{
    let ndc = uv * 2.0 - 1.0;

    let cs_pos = float4(ndc, depth, 1.0);
    let vs_pos_h = mul(frame.inv_proj, cs_pos);
    return vs_pos_h.xyz / vs_pos_h.w;
}

public struct GBufferFragment
{
    public float3 albedo;
    public float3 normal;
    public float roughness;
    public float metallic;

    public __init(uint4 packed)
    {
        albedo = float3(
                     (packed.x >> 24) & 0xFF,
                     (packed.x >> 16) & 0xFF,
                     (packed.x >> 8) & 0xFF) /
                 255.0;
        normal = decode_normal_octahedral(packed.y);
        roughness = float((packed.z >> 16) & 0xFFFF) / 65535.0;
        metallic = float((packed.y >> 31) & 1);
    }

    public property uint4 packed
    {
        [__ref]
        get
        {
            uint4 res;
            let rgb = uint3(saturate(albedo) * 255.0 + 0.5);
            res.x = rgb.r << 24 | rgb.g << 16 | rgb.b << 8;
            res.y = (metallic > 0.5 ? 1 : 0) << 31 | encode_normal_octahedral(normal);
            res.z = uint(roughness * 65535.0 + 0.5) << 16;

            return res;
        }
    }
};

// lower 31 bits of u32
// uses John White's octahedral packing strategy https://johnwhite3d.blogspot.com/2017/10/signed-octahedron-normal-encoding.html
float3 decode_normal_octahedral(uint packed)
{
    let x = float((packed >> 16) & 0x7FFF) / 32767.0;
    let y = float((packed >> 1) & 0x7FFF) / 32767.0;
    let sgn = float(packed & 1) * 2.0 - 1.0;
    var res = float3(0.0);
    res.x = x - y;
    res.y = x + y - 1.0;
    res.z = sgn * (1.0 - abs(res.x) - abs(res.y));
    return normalize(res);
}

uint encode_normal_octahedral(float3 normal)
{
    let n = normal / dot(abs(normal), 1.0);
    var nrm = float2(0.0);
    nrm.y = n.y * 0.5 + 0.5;
    nrm.x = n.x * 0.5 + nrm.y;
    nrm.y = n.x * -0.5 + nrm.y;
    uint sgn = n.z >= 0.0 ? 1 : 0;
    return (uint(nrm.x * 32767.0 + 0.5) << 16) | (uint(nrm.y * 32767.0 + 0.5) << 1) | sgn;
}
