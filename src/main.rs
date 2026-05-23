use clap::Parser;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel};
use cuda_host::cuda_module;
use fractal_fanatic::{Bounds, Complex, Float, Fractal, Julia, View, render_pixel, write_ppm};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn compute<T: Float, F: Fractal<T>>(
        fractal: F,
        width: u32,
        height: u32,
        bounds: Bounds<T>,
        max_iter: u32,
        samples: u32,
        mut out: DisjointSlice<f32>,
    ) {
        if let Some((pixel, idx)) = out.get_mut_indexed() {
            *pixel = render_pixel(
                fractal,
                idx.get() as u32,
                width,
                height,
                bounds,
                max_iter,
                samples,
            );
        }
    }
}

/// GPU Mandelbrot renderer (cuda-oxide).
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Output image path (binary PPM, P6).
    #[arg(short, long, default_value = "julia.ppm")]
    output: String,

    /// Image width in pixels.
    #[arg(long, default_value_t = 1920)]
    width: u32,

    /// Image height in pixels.
    #[arg(long, default_value_t = 1080)]
    height: u32,

    /// View center, real axis.
    #[arg(long, default_value_t = -0.5, allow_negative_numbers = true)]
    center_x: f64,

    /// View center, imaginary axis.
    #[arg(long, default_value_t = -0.5, allow_negative_numbers = true)]
    center_y: f64,

    /// Half-width of the view in complex-plane units. Smaller = deeper zoom.
    /// (1.35 frames the whole set.)
    #[arg(long, default_value_t = 1.35)]
    half_x: f64,

    /// Supersampling factor per axis (1 = off, 2 = 4x, 3 = 9x).
    #[arg(short, long, default_value_t = 1)]
    ssaa_factor: u32,

    /// Max number of iterations before a point is deemed to have not escaped.
    #[arg(long, default_value_t = 1024)]
    max_iter: u32,
}

fn main() {
    let args = Args::parse();

    let output = args.output;
    let width = args.width;
    let height = args.height;
    let samples = args.ssaa_factor;
    let max_iter = args.max_iter;

    let view = View {
        center_x: args.center_x,
        center_y: args.center_y,
        half_x: args.half_x,
    };
    let bounds = view.get_bounds(width, height);

    let n: usize = (width as usize) * (height as usize);

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let mut c_device = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .compute(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
            Julia {
                c: Complex {
                    re: -0.7,
                    im: 0.27015,
                },
            },
            width,
            height,
            bounds,
            max_iter,
            samples,
            &mut c_device,
        )
        .expect("Kernel launch failed");

    let c_host = c_device
        .to_host_vec(&stream)
        .expect("Failed to copy back to host");

    write_ppm(&output, width, height, &c_host, max_iter).expect("Failed to write ppm");

    println!("Wrote {} ({}x{})", &output, width, height);
}
