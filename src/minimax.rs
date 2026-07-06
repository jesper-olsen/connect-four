//! Depth-limited minimax with alpha-beta pruning and a heuristic evaluation
//! function (see `eval.rs`) at the horizon. Unlike `search::Solver`, this
//! never proves an exact game-theoretic result -- it trades that guarantee
//! for bounded, predictable search cost, and doesn't share any code with
//! the exact solver (different score representation, no transposition
//! table, no history heuristic).

use crate::board::{Board, SIZE, WIDTH};
use crate::eval::evaluate;

/// Score magnitude used for a proven win/loss, comfortably larger than any
/// heuristic evaluation `eval::evaluate` can produce.
const WIN_SCORE: i32 = 1_000_000;

/// Try the center column first, then outward -- purely a move-ordering
/// heuristic to help alpha-beta cut earlier; doesn't affect correctness.
const COLUMN_ORDER: [usize; WIDTH] = center_out_order();

const fn center_out_order() -> [usize; WIDTH] {
    // Hand-written for WIDTH=7: 3,2,4,1,5,0,6. Written explicitly (rather
    // than computed) since Board's dimensions are fixed constants anyway.
    [3, 2, 4, 1, 5, 0, 6]
}

pub struct MinimaxAi {
    pub depth: u32,
}

impl MinimaxAi {
    pub fn new(depth: u32) -> Self {
        MinimaxAi { depth }
    }

    /// Choose a column by searching `self.depth` plies ahead and scoring
    /// the horizon with `eval::evaluate`. Returns `None` only if the board
    /// is completely full.
    pub fn best_move(&self, board: &Board) -> Option<usize> {
        let side = board.side();

        // Fast path: take an outright win immediately, no search needed.
        for &col in COLUMN_ORDER.iter() {
            if board.is_playable(col) {
                let candidate = board.color[side] | (1u64 << board.height[col]);
                if Board::has_won(candidate) {
                    return Some(col);
                }
            }
        }

        let mut best_col = None;
        let mut alpha = -WIN_SCORE - 1;
        let beta = WIN_SCORE + 1;
        let mut b = *board;

        for &col in COLUMN_ORDER.iter() {
            if !b.is_playable(col) {
                continue;
            }
            b.make_move(col);
            let score = -self.negamax(&mut b, self.depth.saturating_sub(1), -beta, -alpha);
            b.unmake_move();
            if best_col.is_none() || score > alpha {
                best_col = Some(col);
                alpha = score;
            }
        }
        best_col
    }

    fn negamax(&self, board: &mut Board, depth: u32, mut alpha: i32, beta: i32) -> i32 {
        let side = board.side();
        let other = side ^ 1;

        // The side to move here is already lost if their opponent's last
        // move completed a connect-four; prefer slower losses (more
        // remaining depth at detection = a shallower, sooner loss).
        if Board::has_won(board.color[other]) {
            return -(WIN_SCORE + depth as i32);
        }
        if board.nplies == SIZE {
            return 0; // draw
        }
        if depth == 0 {
            return evaluate(board, side);
        }

        let mut best = -WIN_SCORE - 1;
        let mut any_move = false;
        for &col in COLUMN_ORDER.iter() {
            if !board.is_playable(col) {
                continue;
            }
            any_move = true;
            board.make_move(col);
            let score = -self.negamax(board, depth - 1, -beta, -alpha);
            board.unmake_move();
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }
        if !any_move {
            return 0; // board full, draw (SIZE check above should already catch this)
        }
        best
    }
}

impl Default for MinimaxAi {
    fn default() -> Self {
        MinimaxAi::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board_from_moves(moves: &str) -> Board {
        let mut b = Board::new();
        for c in moves.chars() {
            let col = c.to_digit(10).unwrap() as usize - 1;
            b.make_move(col);
        }
        b
    }

    #[test]
    fn takes_immediate_win_when_available() {
        // Side to move has three in a row horizontally at row 0, cols 0-2,
        // with column 3 open -- col4 (1-indexed) completes it.
        let mut b = board_from_moves("");
        for &(col, opponent_col) in &[(0, 4), (1, 4), (2, 4)] {
            b.make_move(col);
            b.make_move(opponent_col);
        }
        let ai = MinimaxAi::new(4);
        assert_eq!(ai.best_move(&b), Some(3));
    }

    #[test]
    fn always_returns_a_legal_column() {
        let b = board_from_moves("445544554455"); // arbitrary midgame position
        let ai = MinimaxAi::new(3);
        let mv = ai.best_move(&b).expect("board isn't full");
        assert!(b.is_playable(mv));
    }
}
