mod fractal;
mod number;

use crate::fractal::{Fractal, Julia, Mandlebrot};
use crate::number::{Complex, EscapedPoint, Float};
use clap::Parser;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel};
use cuda_host::cuda_module;

#[inline(always)]
fn smooth<T: Float>(e: EscapedPoint<T>, max_iter: u32) -> T {
    if e.iter >= max_iter {
        return T::ZERO;
    }

    let log_zn = T::HALF * e.z.modulus_sq().approximate_ln();
    let nu = (log_zn / T::LN_2).approximate_ln() / T::LN_2;
    let mu = T::from_u32(e.iter) + T::ONE - nu;
    if mu > T::ZERO { mu } else { T::ZERO }
}

#[cuda_module]
mod kernels {
    use super::*;
    use number::Float;

    #[kernel]
    pub fn compute<T: Float, F: Fractal<T>>(
        fractal: F,
        width: u32,
        height: u32,
        min_x: T,
        max_x: T,
        min_y: T,
        max_y: T,
        max_iter: u32,
        samples: u32,
        mut out: DisjointSlice<f32>,
    ) {
        if let Some((pixel, idx)) = out.get_mut_indexed() {
            let i = idx.get();
            let total = T::from_u32(samples * samples);

            let px = (i as u32) % width;
            let py = (i as u32) / height;

            let span_x = max_x - min_x;
            let span_y = max_y - min_y;
            let px_w = span_x / T::from_u32(width - 1);
            let px_h = span_y / T::from_u32(height - 1);

            let base_x = min_x + T::from_u32(px) * px_w;
            let base_y = min_y + T::from_u32(py) * px_h;

            let mut acc = T::ZERO;
            let mut sy = 0u32;
            let inv_n = T::ONE / T::from_u32(samples);

            while sy < samples {
                let oy = (T::from_u32(sy) + T::HALF) * inv_n - T::HALF;
                let mut sx = 0u32;
                while sx < samples {
                    let ox = (T::from_u32(sx) + T::HALF) * inv_n - T::HALF;
                    let c = Complex {
                        re: base_x + ox * px_w,
                        im: base_y + oy * px_h,
                    };
                    // Two phases: escape-iterate, then smooth the raw result.
                    let e = fractal.iterate_until_escape(c, max_iter);
                    acc = acc + smooth::<T>(e, max_iter);
                    sx += 1;
                }
                sy += 1;
            }

            *pixel = (acc / total).to_f32();
        }
    }
}

/// GPU Mandelbrot renderer (cuda-oxide).
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Output image path (binary PPM, P6).
    #[arg(short, long, default_value = "mandelbrot.ppm")]
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
    let center_x = args.center_x;
    let center_y = args.center_y;
    let half_x = args.half_x;
    let samples = args.ssaa_factor;
    let max_iter = args.max_iter;

    let aspect = width as f64 / height as f64;
    let half_y = half_x / aspect;

    let min_x = center_x - half_x;
    let max_x = center_x + half_x;
    let min_y = center_y - half_y;
    let max_y = center_y + half_y;

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
            min_x,
            max_x,
            min_y,
            max_y,
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

fn write_ppm(
    path: &str,
    width: u32,
    height: u32,
    c_host: &[f32],
    max_iter: u32,
) -> std::io::Result<()> {
    use std::io::Write;

    let mut buf = Vec::with_capacity(c_host.len() * 3 + 32);
    write!(buf, "P6\n{} {}\n255\n", width, height).expect("could not write to buffer.");

    for &v in c_host {
        let (r, g, b) = if v <= 0.0 {
            (0, 0, 0)
        } else {
            let t = (v / max_iter as f32).clamp(0.0, 1.0).powf(0.5);
            let (r, g, b) = viridis(t);
            // Fade to black at the low end so the fast-escaping outer region
            // sinks into black instead of showing viridis's purple floor.
            // brightness ramps 0 -> 1 across the first ~15% of the range.
            let brightness = (t / 0.15).clamp(0.0, 1.0);
            (
                (r as f32 * brightness) as u8,
                (g as f32 * brightness) as u8,
                (b as f32 * brightness) as u8,
            )
        };
        buf.push(r);
        buf.push(g);
        buf.push(b);
    }

    std::fs::write(path, buf)
}

fn viridis(t: f32) -> (u8, u8, u8) {
    // 11 evenly-spaced anchors (t = 0.0, 0.1, ..., 1.0) as normalized RGB.
    const ANCHORS: [(f32, f32, f32); 11] = [
        (0.267, 0.005, 0.329),
        (0.283, 0.141, 0.458),
        (0.254, 0.265, 0.530),
        (0.207, 0.372, 0.553),
        (0.164, 0.471, 0.558),
        (0.128, 0.567, 0.551),
        (0.135, 0.659, 0.518),
        (0.267, 0.749, 0.441),
        (0.478, 0.821, 0.318),
        (0.741, 0.873, 0.150),
        (0.993, 0.906, 0.144),
    ];

    let t = t.clamp(0.0, 1.0);
    let scaled = t * (ANCHORS.len() - 1) as f32;
    let i = scaled.floor() as usize;
    // Guard the top edge so i+1 stays in bounds when t == 1.0.
    let (i, frac) = if i >= ANCHORS.len() - 1 {
        (ANCHORS.len() - 2, 1.0)
    } else {
        (i, scaled - i as f32)
    };

    let (r0, g0, b0) = ANCHORS[i];
    let (r1, g1, b1) = ANCHORS[i + 1];
    let lerp = |a: f32, b: f32| a + (b - a) * frac;
    (
        (lerp(r0, r1) * 255.0) as u8,
        (lerp(g0, g1) * 255.0) as u8,
        (lerp(b0, b1) * 255.0) as u8,
    )
}
