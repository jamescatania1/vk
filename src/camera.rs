use std::{
    f64::{EPSILON, consts::PI},
    time::Duration,
};

use glam::{DMat4, DVec3, Mat4, UVec2, dvec3};
use winit::keyboard::KeyCode;

use crate::input::Input;

#[derive(Debug)]
pub struct Camera {
    pub position: DVec3,
    pub view_proj: Mat4,
    velocity: DVec3,
    rotation: DVec3,
    fov: f64,
    near: f64,
    far: f64,
}

impl Camera {
    const UP: DVec3 = DVec3::Z;
    const MOUSE_SENSITIVITY: f64 = 0.001;

    pub fn new() -> Self {
        Self {
            position: dvec3(-2.0, -2.0, 0.5),
            velocity: DVec3::ZERO,
            rotation: DVec3::ZERO,
            fov: 90.0f64.to_radians(),
            near: 0.01,
            far: 100.0,
            view_proj: Mat4::IDENTITY,
        }
    }

    pub fn update(
        &mut self,
        size: UVec2,
        delta_time: &Duration,
        input: &Input,
        cursor_locked: bool,
    ) {
        let mut in_vec = glam::ivec2(
            input.key_down(KeyCode::KeyD) as i32 - input.key_down(KeyCode::KeyA) as i32,
            input.key_down(KeyCode::KeyW) as i32 - input.key_down(KeyCode::KeyS) as i32,
        )
        .as_dvec2();
        in_vec /= in_vec.length().max(1.0);
        if !cursor_locked {
            in_vec = glam::DVec2::ZERO;
        }

        if cursor_locked {
            self.rotation.z -= input.mouse.delta.x * Self::MOUSE_SENSITIVITY;
            self.rotation.x -= input.mouse.delta.y * Self::MOUSE_SENSITIVITY;
            self.rotation.x = self
                .rotation
                .x
                .clamp(-PI / 2.0 + EPSILON, PI / 2.0 - EPSILON);
        }
        let forward = dvec3(
            self.rotation.z.cos() * self.rotation.x.cos(),
            self.rotation.z.sin() * self.rotation.x.cos(),
            self.rotation.x.sin(),
        )
        .normalize();
        let right = -Self::UP.cross(forward);

        let targ_velocity = in_vec.x * right + in_vec.y * forward;
        let delta_ms = (delta_time.as_secs_f64() * 1000.0).clamp(0.1, 1000.0);
        self.velocity = self.velocity.lerp(targ_velocity, delta_ms * 0.01);
        self.position += self.velocity * (delta_ms * 0.005);

        let view = DMat4::look_at_rh(self.position, self.position + forward, Self::UP);
        let mut proj = DMat4::perspective_rh(
            self.fov,
            size.x as f64 / size.y.max(1) as f64,
            self.near,
            self.far,
        );
        proj.y_axis.y *= -1.0;

        self.view_proj = (proj * view).as_mat4();
    }
}
