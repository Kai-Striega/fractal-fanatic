use crate::fractal::Fractal;
use crate::number::{Complex, EscapedPoint, Float};

#[inline(always)]
pub fn smooth<T: Float>(e: EscapedPoint<T>, max_iter: u32) -> T {
    if e.iter >= max_iter {
        return T::ZERO;
    }

    let log_zn = T::HALF * e.z.modulus_sq().approximate_ln();
    let nu = (log_zn / T::LN_2).approximate_ln() / T::LN_2;
    let mu = T::from_u32(e.iter) + T::ONE - nu;
    if mu > T::ZERO { mu } else { T::ZERO }
}

#[inline(always)]
pub fn render_pixel<T: Float, F: Fractal<T>>(
    fractal: F,
    i: u32,
    width: u32,
    height: u32,
    min_x: T,
    max_x: T,
    min_y: T,
    max_y: T,
    max_iter: u32,
    samples: u32,
) -> f32 {
    let total = T::from_u32(samples * samples);

    let px = i % width;
    let py = i / width;

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

    (acc / total).to_f32()
}
