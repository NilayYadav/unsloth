#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    fn right(&self) -> f64 {
        self.x + self.width
    }

    fn bottom(&self) -> f64 {
        self.y + self.height
    }
}

const GAP: f64 = 8.0;

/// All rects are in top-left-origin screen coordinates. Prefers above the
/// selection, flips below when clipped, clamps inside `screen`.
pub fn place_pill(selection: Rect, pill: (f64, f64), screen: Rect) -> (f64, f64) {
    let (pill_w, pill_h) = pill;
    let mut x = selection.x + selection.width / 2.0 - pill_w / 2.0;
    let above_y = selection.y - GAP - pill_h;
    let mut y = if above_y >= screen.y {
        above_y
    } else {
        selection.bottom() + GAP
    };

    x = x.clamp(screen.x, (screen.right() - pill_w).max(screen.x));
    y = y.clamp(screen.y, (screen.bottom() - pill_h).max(screen.y));
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: Rect = Rect { x: 0.0, y: 0.0, width: 1440.0, height: 900.0 };
    const PILL: (f64, f64) = (320.0, 44.0);

    #[test]
    fn places_above_and_centered_when_space_allows() {
        let selection = Rect::new(600.0, 400.0, 200.0, 20.0);
        let (x, y) = place_pill(selection, PILL, SCREEN);
        assert_eq!(x, 600.0 + 100.0 - 160.0);
        assert_eq!(y, 400.0 - 8.0 - 44.0);
    }

    #[test]
    fn flips_below_when_clipped_at_top() {
        let selection = Rect::new(600.0, 20.0, 200.0, 20.0);
        let (_, y) = place_pill(selection, PILL, SCREEN);
        assert_eq!(y, 40.0 + 8.0);
    }

    #[test]
    fn clamps_to_screen_edges() {
        let selection = Rect::new(-50.0, 400.0, 60.0, 20.0);
        let (x, _) = place_pill(selection, PILL, SCREEN);
        assert_eq!(x, 0.0);

        let selection = Rect::new(1400.0, 400.0, 100.0, 20.0);
        let (x, _) = place_pill(selection, PILL, SCREEN);
        assert_eq!(x, 1440.0 - 320.0);
    }

    #[test]
    fn respects_screen_origin_offset_for_secondary_displays() {
        let screen = Rect::new(1440.0, 100.0, 1920.0, 1080.0);
        let selection = Rect::new(1500.0, 105.0, 100.0, 20.0);
        let (x, y) = place_pill(selection, PILL, screen);
        // No room above inside this screen → below
        assert_eq!(y, 125.0 + 8.0);
        assert!(x >= screen.x);
    }

    #[test]
    fn point_anchor_stays_on_screen() {
        let (x, y) = place_pill(Rect::new(10.0, 10.0, 1.0, 1.0), PILL, SCREEN);
        assert!(x >= 0.0 && y >= 0.0);
    }
}
