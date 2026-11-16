//! Opt-in generation policy: consume real think tokens, then evaluate a real closing marker.
use tokenizer::Token;

#[derive(Clone, Debug)]
pub struct Markers {
    pub start: Token,
    pub end: Token,
    pub opened: bool,
}

pub(crate) struct Budget {
    markers: Markers,
    remaining: usize,
    active: bool,
    finished: bool,
    output_left: usize,
}
impl Budget {
    pub(crate) fn new(markers: Markers, requested: usize, output_room: usize) -> Self {
        let active = markers.opened;
        Self {
            markers,
            remaining: requested.min(output_room.saturating_sub(2)),
            active,
            finished: false,
            output_left: output_room,
        }
    }
    pub(crate) fn forced(&self) -> Option<Token> {
        (self.active && (self.remaining == 0 || self.output_left <= 2)).then_some(self.markers.end)
    }
    pub(crate) fn accepted(&mut self, token: Token) {
        self.output_left = self.output_left.saturating_sub(1);
        if token == self.markers.end {
            self.active = false;
            self.finished = true;
        } else if token == self.markers.start && !self.finished {
            self.active = true;
        } else if self.active {
            self.remaining = self.remaining.saturating_sub(1);
        }
    }
}
