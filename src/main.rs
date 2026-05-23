use clap::Parser;
use core::ops::{Add, Div, Mul, Sub};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel};
use cuda_host::cuda_module;

/// Minimal floating point abstraction for device code
pub trait Float:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
{
    const ZERO: Self;
    const HALF: Self;
    const ONE: Self;
    const TWO: Self;
    const FOUR: Self;
    const LN_2: Self;
    fn from_u32(v: u32) -> Self;
    fn to_f32(self) -> f32;
    fn approximate_ln(self) -> Self;
}

impl Float for f32 {
    const ZERO: Self = 0.0f32;
    const HALF: Self = 0.5f32;
    const ONE: Self = 1.0f32;
    const TWO: Self = 2.0f32;
    const FOUR: Self = 4.0f32;
    const LN_2: Self = core::f32::consts::LN_2;

    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        v as f32
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn approximate_ln(self) -> Self {
        if self <= 0.0 {
            return 0.0;
        }
        let bits = self.to_bits();
        let e = ((bits >> 23) & 0xff) as i32 - 127;
        let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
        let t = m - 1.0;
        let p = t * (1.0 + t * (-0.5 + t * (0.3333 - t * 0.25)));
        p + (e as f32) * core::f32::consts::LN_2
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0f64;
    const HALF: Self = 0.5f64;
    const ONE: Self = 1.0f64;
    const TWO: Self = 2.0f64;
    const FOUR: Self = 4.0f64;
    const LN_2: Self = core::f64::consts::LN_2;

    #[inline(always)]
    fn from_u32(v: u32) -> Self {
        v as f64
    }

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline(always)]
    fn approximate_ln(self) -> Self {
        if self <= 0.0 {
            return 0.0;
        }
        let bits = self.to_bits();
        let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
        let m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
        let t = m - 1.0;
        let p = t * (1.0 + t * (-0.5 + t * (0.33333333 + t * (-0.25 + t * 0.2))));
        p + (e as f64) * core::f64::consts::LN_2
    }
}

struct EscapedPoint<T: Float> {
    re: T,
    im: T,
    iters: u32,
}

impl<T: Float> EscapedPoint<T> {
    fn magnitude_squared(&self) -> T {
        self.re * self.re + self.im * self.im
    }

    fn new(re: T, im: T, iters: u32) -> Self {
        EscapedPoint { re, im, iters }
    }
}

#[inline(always)]
fn escape<T: Float>(cx: T, cy: T, max_iter: u32) -> EscapedPoint<T> {
    let mut zx = T::ZERO;
    let mut zy = T::ZERO;
    let mut iter = 0u32;

    while iter < max_iter {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        if zx2 + zy2 > T::FOUR {
            break;
        }
        zy = T::TWO * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        iter += 1;
    }
    EscapedPoint::new(zx, zy, iter)
}

#[inline(always)]
fn smooth<T: Float>(e: EscapedPoint<T>, max_iter: u32) -> T {
    if e.iters >= max_iter {
        return T::ZERO;
    }

    let log_zn = T::HALF * e.magnitude_squared().approximate_ln();
    let nu = (log_zn / T::LN_2).approximate_ln() / T::LN_2;
    let mu = T::from_u32(e.iters) + T::ONE - nu;
    if mu > T::ZERO { mu } else { T::ZERO }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn mandelbrot<T: Float>(
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
                    let cx = base_x + ox * px_w;
                    let cy = base_y + oy * px_h;
                    // Two phases: escape-iterate, then smooth the raw result.
                    let e = escape::<T>(cx, cy, max_iter);
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
        .mandelbrot(
            &stream,
            LaunchConfig::for_num_elems(n as u32),
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

    write_ppm("mandelbrot.ppm", width, height, &c_host, max_iter).expect("Failed to write ppm");

    println!("Wrote mandelbrot.ppm ({}x{})", width, height);
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
