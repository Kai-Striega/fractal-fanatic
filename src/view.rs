pub struct View {
    pub center_x: f64,
    pub center_y: f64,
    pub half_x: f64,
}

pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl View {
    pub fn bounds(&self, width: u32, height: u32) -> Bounds {
        let aspect = width as f64 / height as f64;
        let half_y = self.half_x / aspect;

        Bounds {
            min_x: self.center_x - self.half_x,
            max_x: self.center_x + self.half_x,
            min_y: self.center_y - half_y,
            max_y: self.center_y + half_y,
        }
    }
}
