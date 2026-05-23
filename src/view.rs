use crate::Float;

pub struct View<F: Float> {
    pub center_x: F,
    pub center_y: F,
    pub half_x: F,
}

#[derive(Clone, Copy)]
pub struct Bounds<F: Float> {
    pub min_x: F,
    pub max_x: F,
    pub min_y: F,
    pub max_y: F,
}

impl<F: Float> View<F> {
    pub fn get_bounds(&self, width: u32, height: u32) -> Bounds<F> {
        let aspect = F::from_u32(width) / F::from_u32(height);
        let half_y = self.half_x / aspect;

        Bounds {
            min_x: self.center_x - self.half_x,
            max_x: self.center_x + self.half_x,
            min_y: self.center_y - half_y,
            max_y: self.center_y + half_y,
        }
    }
}

impl<F: Float> Bounds<F> {
    pub fn span_x(&self) -> F { self.max_x - self.min_x }
    pub fn span_y(&self) -> F { self.max_y - self.min_y }
}
