use crate::number::{Complex, EscapedPoint, Float};

pub trait Fractal<T: Float>: Copy {
    fn step(self, z: Complex<T>, c: Complex<T>) -> Complex<T>;
    fn setup(self, pixel: Complex<T>) -> (Complex<T>, Complex<T>);
    fn iterate_until_escape(self, pixel: Complex<T>, max_iter: u32) -> EscapedPoint<T>;
}

#[derive(Copy, Clone)]
pub struct Mandlebrot;

impl<T: Float> Fractal<T> for Mandlebrot {
    fn step(self, z: Complex<T>, c: Complex<T>) -> Complex<T> {
        z.sq() + c
    }

    fn setup(self, pixel: Complex<T>) -> (Complex<T>, Complex<T>) {
        (Complex::zero(), pixel)
    }

    fn iterate_until_escape(self, pixel: Complex<T>, max_iter: u32) -> EscapedPoint<T> {
        let (mut z, c) = self.setup(pixel);
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
pub struct Julia<F: Float> {
    pub c: Complex<F>,
}

impl<T: Float> Fractal<T> for Julia<T> {
    fn step(self, z: Complex<T>, c: Complex<T>) -> Complex<T> {
        z.sq() + c
    }

    fn setup(self, pixel: Complex<T>) -> (Complex<T>, Complex<T>) {
        (pixel, self.c)
    }

    fn iterate_until_escape(self, pixel: Complex<T>, max_iter: u32) -> EscapedPoint<T> {
        let (mut z, c) = self.setup(pixel);
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
