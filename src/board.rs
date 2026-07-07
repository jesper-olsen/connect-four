//! Bitboard core for Connect Four, ported from Game.c (Fhourstones 3.2, J. Tromp).
//!
//! Board encoding (7x6 case), bit index = H1*col + row, row 0 = bottom:
//!   .  .  .  .  .  .  .   TOP (guard row)
//!   5 12 19 26 33 40 47
//!   4 11 18 25 32 39 46
//!   3 10 17 24 31 38 45
//!   2  9 16 23 30 37 44
//!   1  8 15 22 29 36 43
//!   0  7 14 21 28 35 42   BOTTOM
//!
//! `color[nplies & 1]` is always the side to move.

use std::fmt;

pub const WIDTH: usize = 7;
pub const HEIGHT: usize = 6;
pub const H1: usize = HEIGHT + 1; // column stride, including guard row
pub const H2: usize = HEIGHT + 2;
pub const SIZE: usize = HEIGHT * WIDTH; // playable squares
pub const SIZE1: usize = H1 * WIDTH; // bits used (with guard row)

// COL1 has bits 0..HEIGHT set (one column's worth of playable rows).
const COL1: u64 = (1u64 << H1) - 1;
// ALL1 has all SIZE1 bits set.
const ALL1: u64 = (1u64 << SIZE1) - 1;
// BOTTOM has bit i*H1 set for each column i (the bottom row).
const BOTTOM: u64 = ALL1 / COL1;
// TOP has the guard-row bit set for each column (one above the playable rows).
const TOP: u64 = BOTTOM << HEIGHT;

#[derive(Clone, Copy)]
pub struct Board {
    /// color[0] = player-to-move-first's stones, color[1] = other's,
    /// indexed as color[side], side = nplies & 1 for "side to move".
    pub color: [u64; 2],
    /// height[col] = bit index of the next free square in that column.
    pub height: [u8; WIDTH],
    /// move history, column index per ply (0-based), used to unmake moves.
    pub moves: [u8; SIZE],
    pub nplies: usize,
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f, "Moves: ")?;
        for m in &self.moves[0..self.nplies] {
            write!(f, "{m}")?;
        }
        Ok(())
    }
}

impl Board {
    pub fn new() -> Self {
        Board {
            color: [0, 0],
            height: std::array::from_fn(|i| (H1 * i) as u8),
            moves: [0; SIZE],
            nplies: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Board::new();
    }

    /// Side to move: 0 or 1.
    #[inline]
    pub fn side(&self) -> usize {
        self.nplies & 1
    }

    /// True iff `newboard` has no stone in the guard row (i.e. no column overflow).
    #[inline]
    pub fn is_legal(newboard: u64) -> bool {
        newboard & TOP == 0
    }

    /// True iff column `col` has room for another stone.
    #[inline]
    pub fn is_playable(&self, col: usize) -> bool {
        let candidate = self.color[self.side()] | (1u64 << self.height[col]);
        Board::is_legal(candidate)
    }

    /// True iff `newboard` contains four in a row (horizontal, vertical, or either diagonal).
    #[inline]
    pub fn has_won(newboard: u64) -> bool {
        // each of the four directions is handled the same way:
        // 1) detect adjacent coins 
        // 2) shifts 'two positons' to detect 4 stones next to each other
        let diag1 = newboard & (newboard >> HEIGHT);
        let hori = newboard & (newboard >> H1);   // 1=> pair of coins in a row
        let diag2 = newboard & (newboard >> H2);
        let vert = newboard & (newboard >> 1);
        let win = (diag1 & (diag1 >> (2 * HEIGHT)))
            | (hori & (hori >> (2 * H1)))         // 1 => 2 pairs of coins next to wach other
            | (diag2 & (diag2 >> (2 * H2)))
            | (vert & (vert >> 2));
        win != 0
    }

    /// True iff `newboard` is both a legal position and a win.
    #[inline]
    pub fn is_legal_has_won(newboard: u64) -> bool {
        Board::is_legal(newboard) && Board::has_won(newboard)
    }

    /// Play a stone in `col` for the side to move. Caller must ensure legality.
    /// Order matters: the old height value is used for the shift *before* incrementing,
    /// matching C's `height[n]++` post-increment inside the shift expression.
    pub fn make_move(&mut self, col: usize) {
        let side = self.side();
        self.color[side] ^= 1u64 << self.height[col];
        self.height[col] += 1;
        self.moves[self.nplies] = col as u8;
        self.nplies += 1;
    }

    /// Undo the last move played. Mirror image of `make_move`'s bit ops.
    pub fn unmake_move(&mut self) {
        self.nplies -= 1;
        let col = self.moves[self.nplies] as usize;
        self.height[col] -= 1;
        let side = self.side();
        self.color[side] ^= 1u64 << self.height[col];
    }

    /// Complete encoding of the current position (whose turn + both players' stones),
    /// suitable for hashing. Matches Game.c's `positioncode`.
    pub fn position_code(&self) -> u64 {
        self.color[self.side()] + self.color[0] + self.color[1] + BOTTOM
    }
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_heights_match_column_starts() {
        let b = Board::new();
        for i in 0..WIDTH {
            assert_eq!(b.height[i], (H1 * i) as u8);
        }
        assert_eq!(b.nplies, 0);
        assert_eq!(b.color, [0, 0]);
    }

    #[test]
    fn make_unmake_round_trip_restores_state() {
        let mut b = Board::new();
        let before = (b.color, b.height, b.nplies);
        for &col in &[3usize, 3, 2, 4, 4, 4] {
            b.make_move(col);
        }
        for _ in 0..6 {
            b.unmake_move();
        }
        assert_eq!((b.color, b.height, b.nplies), before);
    }

    #[test]
    fn no_win_on_empty_board() {
        assert!(!Board::has_won(0));
    }

    #[test]
    fn vertical_win_detected() {
        let mut b = Board::new();
        // Four stacked stones for side 0 in column 0: plies 0,2,4,6 are side 0's moves
        // (side alternates each ply), so play col 0 six times alternated with col 1
        // to keep side-0's stones stacking in column 0 without needing col 1 to win.
        for _ in 0..4 {
            b.make_move(0); // side 0
            if b.nplies < 7 {
                b.make_move(1); // side 1, elsewhere, avoid interfering
            }
        }
        assert!(Board::has_won(b.color[0]));
    }

    #[test]
    fn horizontal_win_detected() {
        let mut b = Board::new();
        // side 0 plays columns 0,1,2,3 on the bottom row (moves 0,2,4,6); side 1 fills
        // columns 0,1,2 on top of those in between (moves 1,3,5) to keep alternation.
        b.make_move(0); // side0 col0
        b.make_move(0); // side1 col0
        b.make_move(1); // side0 col1
        b.make_move(1); // side1 col1
        b.make_move(2); // side0 col2
        b.make_move(2); // side1 col2
        b.make_move(3); // side0 col3
        assert!(Board::has_won(b.color[0]));
    }

    #[test]
    fn overflow_column_is_illegal() {
        let mut b = Board::new();
        // Fill column 0 to the top (HEIGHT stones), alternating opponent moves elsewhere.
        for _ in 0..HEIGHT {
            b.make_move(0);
            b.make_move(6); // keep alternation away from column 0
        }
        // One more stone in column 0 would overflow into the guard row.
        let overflow = b.color[b.side()] | (1u64 << b.height[0]);
        assert!(!Board::is_legal(overflow));
    }

    #[test]
    fn position_code_changes_with_moves() {
        let mut b = Board::new();
        let code0 = b.position_code();
        b.make_move(3);
        let code1 = b.position_code();
        assert_ne!(code0, code1);
    }
}
