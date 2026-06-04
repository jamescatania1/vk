#![allow(unused)]

use std::collections::HashSet;

use glam::DVec2;
use winit::{
    event::{DeviceEvent, ElementState},
    keyboard::{KeyCode, PhysicalKey},
};

#[derive(Debug, Default)]
pub struct Input {
    pub mouse: Mouse,
    pub scroll: f64,
    pressed_keys: HashSet<KeyCode>,
    released_keys: HashSet<KeyCode>,
}

impl Input {
    pub fn handle_mouse(&mut self, state: &ElementState, button: &winit::event::MouseButton) {
        if let Some(mb) = match button {
            winit::event::MouseButton::Left => Some(&mut self.mouse.left),
            winit::event::MouseButton::Right => Some(&mut self.mouse.right),
            _ => None,
        } {
            if *state == ElementState::Pressed {
                mb.clicked = true;
                mb.down = true;
            } else {
                mb.released = true;
                mb.down = false;
            }
        }
    }

    pub fn handle_keyboard(
        &mut self,
        window: &winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::KeyEvent,
    ) {
        if event.repeat {
            return;
        }

        if let winit::keyboard::Key::Character(key) = &event.logical_key {
            match event.state {
                ElementState::Pressed => {
                    if key == "f" {
                        if window.fullscreen().is_some() {
                            window.set_fullscreen(None);
                        } else {
                            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(
                                event_loop.primary_monitor(),
                            )));
                        }
                    }
                }
                ElementState::Released => {}
            }
        }

        if let PhysicalKey::Code(code) = event.physical_key {
            match event.state {
                ElementState::Pressed => {
                    self.pressed_keys.insert(code);
                }
                ElementState::Released => {
                    self.pressed_keys.remove(&code);
                }
            }
        }
    }

    pub fn handle_device_input(
        &mut self,
        window: &winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: &winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::Motion { axis: 0, value } => {
                self.mouse.delta.x += *value;
            }
            DeviceEvent::Motion { axis: 1, value } => {
                self.mouse.delta.y += *value;
            }
            _ => {}
        }
    }

    pub fn key_down(&self, code: KeyCode) -> bool {
        self.pressed_keys.contains(&code)
    }

    // Called at the end of each frame
    pub fn update(&mut self) {
        self.mouse.left.clicked = false;
        self.mouse.left.released = false;
        self.mouse.right.clicked = false;
        self.mouse.right.released = false;
        self.mouse.delta *= 0.0;
        self.mouse.scroll_delta = 0.0;
    }
}

#[derive(Debug, Default)]
pub struct Key {
    /// Whether the key was pressed down this specific frame
    pub pressed: bool,
    /// Whether the key was released this specific frame
    pub released: bool,
    /// Whether the key is currently held down
    pub down: bool,
}

#[derive(Debug, Default)]
pub struct Mouse {
    pub delta: DVec2,
    pub scroll_delta: f64,
    pub left: MouseButton,
    pub right: MouseButton,
}

#[derive(Debug, Default)]
pub struct MouseButton {
    /// Whether the button was clicked this specific frame
    pub clicked: bool,
    /// Whether the button was released this specific frame
    pub released: bool,
    /// Whether the button is currently held down
    pub down: bool,
}
