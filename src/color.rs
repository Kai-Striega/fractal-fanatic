pub fn viridis(t: f32) -> (u8, u8, u8) {
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
