use std::{
    f32::{EPSILON, consts::PI},
    time::Duration,
};

use glam::{Mat4, UVec2, Vec2, Vec3, vec3};
use winit::keyboard::KeyCode;

use crate::{input::Input, renderer::SHADOWMAP_SIZE, scene::SceneResources};

pub const CASCADES: usize = 4;
const CASCADES_FAR: f32 = 40.0;

#[derive(Debug)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub view_proj: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
    pub cascades: [Cascade; CASCADES],
    velocity: Vec3,
    rotation: Vec3,
    fov: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Cascade {
    pub view_proj: Mat4,
    pub texel_size: Vec2,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    const UP: Vec3 = Vec3::Z;
    const MOUSE_SENSITIVITY: f64 = 0.001;

    pub fn new() -> Self {
        Self {
            position: vec3(-2.0, -2.0, 0.5),
            forward: vec3(1.0, 0.0, 0.0),
            velocity: Vec3::ZERO,
            rotation: Vec3::ZERO,
            cascades: [Cascade::default(); CASCADES],
            fov: 90.0f32.to_radians(),
            near: 0.01,
            far: 200.0,
            view_proj: Mat4::IDENTITY,
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
        }
    }

    pub fn update(
        &mut self,
        size: UVec2,
        delta_time: &Duration,
        input: &Input,
        cursor_locked: bool,
        sun_dir: glam::Vec3,
        cascade_lambda: f32,
        scene: &SceneResources,
    ) {
        let mut in_vec = glam::ivec2(
            input.key_down(KeyCode::KeyD) as i32 - input.key_down(KeyCode::KeyA) as i32,
            input.key_down(KeyCode::KeyW) as i32 - input.key_down(KeyCode::KeyS) as i32,
        )
        .as_vec2();
        in_vec /= in_vec.length().max(1.0);
        if !cursor_locked {
            in_vec = glam::Vec2::ZERO;
        }

        if cursor_locked {
            self.rotation.z -= (input.mouse.delta.x * Self::MOUSE_SENSITIVITY) as f32;
            self.rotation.x -= (input.mouse.delta.y * Self::MOUSE_SENSITIVITY) as f32;
            self.rotation.x = self
                .rotation
                .x
                .clamp(-PI * 0.5 + EPSILON, PI * 0.5 - EPSILON);
        }
        self.forward = vec3(
            self.rotation.z.cos() * self.rotation.x.cos(),
            self.rotation.z.sin() * self.rotation.x.cos(),
            self.rotation.x.sin(),
        )
        .normalize();
        let right = -Self::UP.cross(self.forward);

        let targ_velocity = in_vec.x * right + in_vec.y * self.forward;
        let delta_ms = (delta_time.as_secs_f64() * 1000.0).clamp(0.1, 1000.0);
        self.velocity = self.velocity.lerp(targ_velocity, (delta_ms * 0.01) as f32);
        self.position += self.velocity * (delta_ms * 0.005) as f32;

        self.view = Mat4::look_at_rh(self.position, self.position + self.forward, Self::UP);
        self.proj = Mat4::perspective_rh(
            self.fov,
            size.x as f32 / size.y.max(1) as f32,
            self.near,
            self.far,
        );
        self.proj.y_axis.y *= -1.0;

        self.view_proj = self.proj * self.view;

        let near = self.near;
        let far = self.far.min(CASCADES_FAR);
        for i in 0..=CASCADES {
            let z = i as f32 / CASCADES as f32;

            let split_unif = near + (far - near) * z;
            let split_log = near * (far / near).powf(z);
            let split = split_unif + (split_log - split_unif) * cascade_lambda;

            if i < CASCADES {
                self.cascades[i].near = split;
            }
            if i > 0 {
                self.cascades[i - 1].far = split;
            }
        }
        for i in 0..(CASCADES - 1) {
            let range = self.cascades[i].far - self.cascades[i].near;
            let blend_factor = 0.1;

            self.cascades[i + 1].near -= range * blend_factor;
        }

        for i in 0..CASCADES {
            let near = self.cascades[i].near;
            let far = self.cascades[i].far;

            let mut frustum_proj =
                Mat4::perspective_rh(self.fov, size.x as f32 / size.y.max(1) as f32, near, far);
            frustum_proj.y_axis.y *= -1.0;

            let inv_view_proj = (frustum_proj * self.view).inverse_or_zero();

            let frustum_corners = [
                glam::vec3(-1.0, 1.0, 0.0),
                glam::vec3(1.0, 1.0, 0.0),
                glam::vec3(-1.0, -1.0, 0.0),
                glam::vec3(1.0, -1.0, 0.0),
                glam::vec3(-1.0, 1.0, 1.0),
                glam::vec3(1.0, 1.0, 1.0),
                glam::vec3(-1.0, -1.0, 1.0),
                glam::vec3(1.0, -1.0, 1.0),
            ]
            .map(|corner| inv_view_proj.project_point3(corner));

            let mut center = glam::Vec3::ZERO;
            for corner in frustum_corners {
                center += corner;
            }
            center /= 8.0;

            let view = glam::Mat4::look_at_rh(center, center - sun_dir, glam::Vec3::Z);

            let mut min_bd = Vec3::MAX;
            let mut max_bd = Vec3::MIN;
            for corner in frustum_corners {
                let vs_corner = view.transform_point3(corner);
                min_bd = min_bd.min(vs_corner);
                max_bd = max_bd.max(vs_corner);
            }
            for primitive in &scene.primitives {
                for corner in primitive.bounds {
                    let vs_corner = view.transform_point3(corner);
                    min_bd.z = min_bd.z.min(vs_corner.z);
                    max_bd.z = max_bd.z.max(vs_corner.z);
                }
            }

            self.cascades[i].texel_size.x = SHADOWMAP_SIZE as f32 / (max_bd.x - min_bd.x);
            self.cascades[i].texel_size.y = SHADOWMAP_SIZE as f32 / (max_bd.y - min_bd.y);

            let near = -max_bd.z - 1.0;
            let far = -min_bd.z + 1.0;

            let proj =
                glam::Mat4::orthographic_rh(min_bd.x, max_bd.x, min_bd.y, max_bd.y, near, far);

            self.cascades[i].view_proj = proj * view;
        }
    }
}
