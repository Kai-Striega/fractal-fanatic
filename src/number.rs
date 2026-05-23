use std::ops::{Add, Div, Mul, Sub};

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

#[derive(Copy, Clone)]
pub struct Complex<T: Float> {
    pub re: T,
    pub im: T,
}

impl<T: Float> Add for Complex<T> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Complex {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl<T: Float> Mul for Complex<T> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Complex {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl<T: Float> Complex<T> {
    #[inline(always)]
    pub fn sq(self) -> Self {
        Complex {
            re: self.re * self.re - self.im * self.im,
            im: T::TWO * self.re * self.im,
        }
    }

    #[inline(always)]
    pub fn modulus_sq(self) -> T {
        self.re * self.re + self.im * self.im
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Complex {
            re: T::ZERO,
            im: T::ZERO,
        }
    }
}

pub struct EscapedPoint<T: Float> {
    pub z: Complex<T>,
    pub iter: u32,
}
