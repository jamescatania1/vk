#[derive(Debug)]
pub struct Config {
    pub render_scale: f32,
    pub debug_view: DebugView,
    pub sun_azimuth: f32,
    pub sun_altitude: f32,
    pub cascade_lambda: f32,
    pub ao_slices: u32,
    pub ao_samples: u32,
    pub ao_radius: f32,
    pub ao_falloff_range: f32,
    pub ao_sample_distribution_power: f32,
    pub ao_thin_occluder_compensation: f32,
    pub ao_final_value_power: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            render_scale: 0.5,
            debug_view: Default::default(),
            sun_azimuth: 45.0,
            sun_altitude: 80.0,
            cascade_lambda: 0.7,
            ao_slices: 4,
            ao_samples: 8,
            ao_radius: 0.35,
            ao_falloff_range: 0.615,
            ao_sample_distribution_power: 2.0,
            ao_thin_occluder_compensation: 0.9,
            ao_final_value_power: 2.2,
        }
    }
}

#[derive(Debug, PartialEq, Default, Copy, Clone)]
pub enum DebugView {
    #[default]
    Composite = 0,
    Albedo = 1,
    Depth = 2,
    Normal = 3,
    Roughness = 4,
    Metallic = 5,
    Velocity = 6,
    AmbientOcclusion = 7,
    Shadow = 8,
}

pub const DEBUG_VIEW_NAMES: &'static [&'static str] = &[
    "Composite",
    "Albedo",
    "Depth",
    "Normal",
    "Roughness",
    "Metallic",
    "Velocity",
    "Ambient Occlusion",
    "Shadow",
];

pub const DEBUG_VIEWS: &'static [DebugView] = &[
    DebugView::Composite,
    DebugView::Albedo,
    DebugView::Depth,
    DebugView::Normal,
    DebugView::Roughness,
    DebugView::Metallic,
    DebugView::Velocity,
    DebugView::AmbientOcclusion,
    DebugView::Shadow,
];
