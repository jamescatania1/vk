use std::{concat, env};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

mod app;
mod camera;
mod input;
use app::App;

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

struct Program {
    app: Option<App>,
}

impl ApplicationHandler for Program {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }

        let cfg = Window::default_attributes()
            .with_inner_size(LogicalSize::new(1920.0, 1080.0))
            .with_title("vulkan");
        let window = event_loop.create_window(cfg).unwrap();
        self.app = Some(App::new(window));
        self.app.as_ref().unwrap().window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(app) = &mut self.app else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                app.frame();
                app.window.request_redraw();
            }
            WindowEvent::MouseInput { state, button, .. } => {
                app.input.handle_mouse(&state, &button);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if !event.repeat {
                    app.input.handle_keyboard(&app.window, event_loop, &event);
                }
            }
            WindowEvent::Resized(_) => {
                app.recreate_swapchain = true;
            }
            _ => {}
        };
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let Some(app) = &mut self.app else {
            return;
        };
        app.input
            .handle_device_input(&app.window, event_loop, &event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(app) = &mut self.app else {
            return;
        };
        // app.about_to_wait(event_loop);
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    event_loop.run_app(&mut Program { app: None }).unwrap();
}
