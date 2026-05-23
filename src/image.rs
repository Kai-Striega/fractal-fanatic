use crate::color::viridis;

pub fn write_ppm(
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
