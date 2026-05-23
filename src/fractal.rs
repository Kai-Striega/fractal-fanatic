use crate::number::{Complex, EscapedPoint, Float};

pub struct Seed<T: Float> {
    pub z: Complex<T>,
    pub c: Complex<T>,
}

pub trait Fractal<T: Float>: Copy {
    fn step(self, z: Complex<T>, c: Complex<T>) -> Complex<T>;
    fn setup(self, pixel: Complex<T>) -> Seed<T>;

    #[inline(always)]
    fn iterate_until_escape(self, pixel: Complex<T>, max_iter: u32) -> EscapedPoint<T> {
        let Seed { mut z, c } = self.setup(pixel);
        let mut iter = 0u32;
        while iter < max_iter {
            if z.modulus_sq() > T::FOUR {
                break;
            }
            z = self.step(z, c);
            iter += 1;
        }

        EscapedPoint { z, iter }
    }
}

#[derive(Copy, Clone)]
pub struct Mandlebrot;

impl<T: Float> Fractal<T> for Mandlebrot {
    #[inline(always)]
    fn step(self, z: Complex<T>, c: Complex<T>) -> Complex<T> {
        z.sq() + c
    }

    #[inline(always)]
    fn setup(self, pixel: Complex<T>) -> Seed<T> {
        Seed {
            z: Complex::zero(),
            c: pixel,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Julia<F: Float> {
    pub c: Complex<F>,
}

impl<T: Float> Fractal<T> for Julia<T> {
    #[inline(always)]
    fn step(self, z: Complex<T>, c: Complex<T>) -> Complex<T> {
        z.sq() + c
    }

    #[inline(always)]
    fn setup(self, pixel: Complex<T>) -> Seed<T> {
        Seed {
            z: pixel,
            c: self.c,
        }
    }
}
