pub mod color;
pub mod fractal;
pub mod image;
pub mod number;
pub mod render;
pub mod view;

pub use color::viridis;
pub use fractal::{Escapee, Fractal, Julia, Mandelbrot, Seed};
pub use image::write_ppm;
pub use number::{Complex, Float};
pub use render::{render_pixel, smooth};
pub use view::{Bounds, View};
