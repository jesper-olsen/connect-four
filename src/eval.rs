//! A standard, "textbook" Connect Four evaluation heuristic: for every
//! possible four-in-a-row window on the board (horizontal, vertical, and
//! both diagonals -- 69 of them on a 7x6 board), score how favorable that
//! window currently looks for each side, plus a small bonus for center-
//! column control (center squares participate in more winning lines than
//! edge squares). This has no connection to `search.rs`'s exact solver --
//! it's an approximation meant to be paired with depth-limited minimax.

use crate::board::{Board, H1, HEIGHT, WIDTH};
use std::sync::OnceLock;

/// Score awarded for a window containing exactly N of one side's pieces
/// and otherwise empty (index = piece count, 0..=3; four-in-a-row is a
/// terminal condition handled separately, never scored via this table).
const WINDOW_SCORE: [i32; 4] = [0, 1, 10, 50];

/// Bonus per own piece in the center column, and penalty per opponent piece
/// there -- center squares appear in more of the 69 windows than any other
/// column, so this reinforces (rather than duplicates) that structural bias.
const CENTER_BONUS: i32 = 3;

fn bit(col: usize, row: usize) -> u64 {
    1u64 << (H1 * col + row)
}

/// All 69 four-in-a-row window bitmasks for a 7x6 board, computed once.
fn winning_windows() -> &'static [u64] {
    static WINDOWS: OnceLock<Vec<u64>> = OnceLock::new();
    WINDOWS.get_or_init(|| {
        let mut windows = Vec::with_capacity(69);

        // Horizontal.
        for row in 0..HEIGHT {
            for col in 0..=WIDTH - 4 {
                windows.push((0..4).fold(0u64, |m, k| m | bit(col + k, row)));
            }
        }
        // Vertical.
        for col in 0..WIDTH {
            for row in 0..=HEIGHT - 4 {
                windows.push((0..4).fold(0u64, |m, k| m | bit(col, row + k)));
            }
        }
        // Diagonal, rising left-to-right ("/").
        for col in 0..=WIDTH - 4 {
            for row in 0..=HEIGHT - 4 {
                windows.push((0..4).fold(0u64, |m, k| m | bit(col + k, row + k)));
            }
        }
        // Diagonal, falling left-to-right ("\").
        for col in 0..=WIDTH - 4 {
            for row in 3..HEIGHT {
                windows.push((0..4).fold(0u64, |m, k| m | bit(col + k, row - k)));
            }
        }
        debug_assert_eq!(windows.len(), 69);
        windows
    })
}

/// Static evaluation of `board` from `side`'s point of view: positive favors
/// `side`, negative favors the opponent. Not meaningful for terminal
/// (already-won or full) positions -- callers should check those first.
pub fn evaluate(board: &Board, side: usize) -> i32 {
    let mine = board.color[side];
    let theirs = board.color[side ^ 1];
    let mut score = 0i32;

    for &window in winning_windows() {
        let mine_bits = (mine & window).count_ones() as usize;
        let their_bits = (theirs & window).count_ones() as usize;
        // A window contested by both sides can never become a four-in-a-row
        // for either, so it only contributes score while one side "owns" it
        // exclusively (the other side has zero pieces in that window).
        if their_bits == 0 && mine_bits > 0 {
            score += WINDOW_SCORE[mine_bits];
        } else if mine_bits == 0 && their_bits > 0 {
            score -= WINDOW_SCORE[their_bits];
        }
    }

    let center_col = WIDTH / 2;
    let center_mask: u64 = (0..HEIGHT).fold(0, |m, row| m | bit(center_col, row));
    score += CENTER_BONUS * (mine & center_mask).count_ones() as i32;
    score -= CENTER_BONUS * (theirs & center_mask).count_ones() as i32;

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_count_is_69() {
        assert_eq!(winning_windows().len(), 69);
    }

    #[test]
    fn empty_board_is_neutral() {
        let b = Board::new();
        assert_eq!(evaluate(&b, 0), 0);
        assert_eq!(evaluate(&b, 1), 0);
    }

    #[test]
    fn center_column_piece_favors_its_owner() {
        let mut b = Board::new();
        b.make_move(WIDTH / 2); // side 0 plays center
        assert!(evaluate(&b, 0) > 0);
        assert!(evaluate(&b, 1) < 0);
        // Symmetric from either side's viewpoint.
        assert_eq!(evaluate(&b, 0), -evaluate(&b, 1));
    }

    #[test]
    fn three_in_a_row_scores_higher_than_two() {
        let mut two = Board::new();
        two.make_move(0);
        two.make_move(4); // opponent plays elsewhere, spread out so it
        two.make_move(1); // never builds a threat of its own that would
        two.make_move(5); // confound this comparison

        let mut three = Board::new();
        three.make_move(0);
        three.make_move(4);
        three.make_move(1);
        three.make_move(5);
        three.make_move(2);
        three.make_move(6);

        assert!(evaluate(&three, 0) > evaluate(&two, 0));
    }
}
