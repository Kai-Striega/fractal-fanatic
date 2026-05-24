use crate::number::{Complex, Float};

#[derive(Copy, Clone)]
pub struct Orbit<T: Float> {
    pub z: Complex<T>,
    pub c: Complex<T>,
    pub z_prev: Complex<T>,
}

pub enum Outcome<T: Float> {
    Terminated { z: Complex<T>, iter: u32 },
    Exhausted,
}

#[inline(always)]
pub fn smooth<T: Float>(z: Complex<T>, iter: u32) -> T {
    let log_zn = T::HALF * z.modulus_sq().approximate_ln();
    let nu = (log_zn / T::LN_2).approximate_ln() / T::LN_2;
    let mu = T::from_u32(iter) + T::ONE - nu;
    if mu > T::ZERO { mu } else { T::ZERO }
}

pub trait Fractal<T: Float>: Copy {
    fn setup(self, pixel: Complex<T>) -> Orbit<T>;
    fn step(self, orbit: Orbit<T>) -> Orbit<T>;

    #[inline(always)]
    fn terminated(self, orbit: Orbit<T>) -> bool {
        orbit.z.modulus_sq() > T::FOUR
    }

    #[inline(always)]
    fn measure(self, outcome: Outcome<T>) -> T {
        match outcome {
            Outcome::Terminated { z, iter } => smooth(z, iter),
            Outcome::Exhausted => T::ZERO,
        }
    }

    #[inline(always)]
    fn run(self, pixel: Complex<T>, max_iter: u32) -> Outcome<T> {
        let mut orbit = self.setup(pixel);
        let mut iter = 0u32;
        while iter < max_iter {
            if self.terminated(orbit) {
                return Outcome::Terminated { z: orbit.z, iter };
            }
            orbit = self.step(orbit);
            iter += 1;
        }
        Outcome::Exhausted
    }
}

#[derive(Copy, Clone)]
pub struct Mandelbrot;

impl<T: Float> Fractal<T> for Mandelbrot {
    #[inline(always)]
    fn setup(self, pixel: Complex<T>) -> Orbit<T> {
        Orbit {
            z: Complex::zero(),
            c: pixel,
            z_prev: Complex::zero(),
        }
    }

    #[inline(always)]
    fn step(self, orbit: Orbit<T>) -> Orbit<T> {
        Orbit {
            z: orbit.z.sq() + orbit.c,
            c: orbit.c,
            z_prev: orbit.z_prev,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Julia<F: Float> {
    pub c: Complex<F>,
}

impl<T: Float> Fractal<T> for Julia<T> {
    #[inline(always)]
    fn setup(self, pixel: Complex<T>) -> Orbit<T> {
        Orbit {
            z: pixel,
            c: self.c,
            z_prev: Complex::zero(),
        }
    }

    #[inline(always)]
    fn step(self, orbit: Orbit<T>) -> Orbit<T> {
        Orbit {
            z: orbit.z.sq() + orbit.c,
            c: orbit.c,
            z_prev: orbit.z_prev,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Phoenix<F: Float> {
    pub c: Complex<F>,
    pub p: Complex<F>,
}

impl<T: Float> Fractal<T> for Phoenix<T> {
    #[inline(always)]
    fn setup(self, pixel: Complex<T>) -> Orbit<T> {
        Orbit {
            z: pixel,
            c: self.c,
            z_prev: Complex::zero(),
        }
    }

    #[inline(always)]
    fn step(self, orbit: Orbit<T>) -> Orbit<T> {
        Orbit {
            z: orbit.z.sq() + orbit.c + self.p.mul(orbit.z_prev),
            c: orbit.c,
            z_prev: orbit.z,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Newton<F: Float> {
    pub tol: F,
}

impl<T: Float> Fractal<T> for Newton<T> {
    #[inline(always)]
    fn setup(self, pixel: Complex<T>) -> Orbit<T> {
        Orbit {
            z: pixel,
            c: Complex::zero(),
            z_prev: Complex::zero(),
        }
    }

    #[inline(always)]
    fn step(self, orbit: Orbit<T>) -> Orbit<T> {
        // z - f(z)/f'(z), with f(z) = z³ - 1 and f'(z) = 3z².
        let z = orbit.z;
        let f = z.sq().mul(z).sub(Complex::one());
        let fp = z.sq().mul(Complex::from_u32(3));
        Orbit {
            z: z.sub(f.div(fp)),
            c: orbit.c,
            z_prev: orbit.z_prev,
        }
    }

    #[inline(always)]
    fn terminated(self, orbit: Orbit<T>) -> bool {
        let z = orbit.z;
        let f = z.sq().mul(z).sub(Complex::one());
        f.modulus_sq() < self.tol * self.tol
    }

    #[inline(always)]
    fn measure(self, outcome: Outcome<T>) -> T {
        match outcome {
            Outcome::Terminated { iter, .. } => T::from_u32(iter),
            Outcome::Exhausted => T::ZERO,
        }
    }
}
