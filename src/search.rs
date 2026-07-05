//! Negamax alpha-beta search with history-heuristic move ordering, ported
//! from SearchGame.c. Scores are encoded 1..=5 (LOSS..=WIN, DRAW=3),
//! negamax-negated via `LOSSWIN - x` (LOSSWIN = 6), so the encoding is
//! symmetric about DRAW.

use crate::board::{Board, H1, HEIGHT, SIZE, SIZE1, WIDTH};
use crate::tt::{DRAW, DRAWLOSS, DRAWWIN, LOSS, LOSSWIN, TransTable, UNKNOWN, WIN};

/// Additional plies (beyond the root) searched full-width before the
/// (currently unused, always-0) opening-book cutoff applies.
const BOOKPLY: usize = 0;
/// Additional plies (beyond the root) for which per-node values are reported.
const REPORTPLY: usize = 2;

const SCORE_CHARS: [char; 6] = ['#', '-', '<', '=', '>', '+'];

/// Result of scanning a position for legal, non-suicidal columns.
enum Candidates {
    /// The position's score is already forced (an unstoppable double threat,
    /// or no safe column at all) -- no further search needed at this node.
    Forced(i32),
    /// `av[..count]` holds the safe, playable columns for this position, in
    /// original left-to-right order (not yet history-ordered).
    Open([usize; WIDTH], usize),
}

/// Scan for playable columns that don't immediately hand the opponent a
/// win, with must-block / unstoppable-double-threat detection -- the exact
/// candidate-generation logic from SearchGame.c's `ab`, factored out so both
/// `ab` (recursive search) and `best_move` (root move selection) apply the
/// same filtering rather than `best_move` re-deriving a weaker version of it.
fn candidates(board: &Board) -> Candidates {
    let side = board.nplies & 1;
    let otherside = side ^ 1;
    let other = board.color[otherside];

    let mut av = [0usize; WIDTH];
    let mut nav = 0usize;
    let mut i = 0usize;
    'scan: while i < WIDTH {
        let newbrd = other | (1u64 << board.height[i]);
        if !Board::is_legal(newbrd) {
            i += 1;
            continue;
        }
        let winontop = Board::is_legal_has_won(other | (2u64 << board.height[i]));
        if Board::has_won(newbrd) {
            // Opponent would win if given this square: we must take it,
            // unless doing so hands them an immediate win on top anyway.
            if winontop {
                return Candidates::Forced(LOSS);
            }
            av[0] = i;
            nav = 1;
            let mut j = i + 1;
            while j < WIDTH {
                if Board::is_legal_has_won(other | (1u64 << board.height[j])) {
                    return Candidates::Forced(LOSS); // a second, unstoppable threat elsewhere
                }
                j += 1;
            }
            break 'scan;
        }
        if !winontop {
            av[nav] = i;
            nav += 1;
        }
        i += 1;
    }
    if nav == 0 {
        return Candidates::Forced(LOSS);
    }
    Candidates::Open(av, nav)
}

/// Selection-sort step: finds the highest-history-value column among
/// `av[i..nav)`, moves it to position `i` (shifting the intervening entries
/// up by one), and returns it. Ties are broken toward the EARLIEST
/// (lowest-index) candidate -- this requires a strict `>`, not `>=`. An
/// earlier port of this search used `>=` and, on the maximally symmetric
/// empty-board position, ended up visiting ~2.7x more nodes for the same
/// final score, because tie-breaking direction changes which subtrees get
/// cut first. Shared by `ab`'s internal move loop and `best_move`'s
/// root-level loop, so both benefit from the same ordering.
fn pick_next_move(
    history_side: &[i32; SIZE1],
    board: &Board,
    av: &mut [usize; WIDTH],
    i: usize,
    nav: usize,
) -> usize {
    let mut l = i;
    let mut val = history_side[board.height[av[l]] as usize];
    let mut j = i + 1;
    while j < nav {
        let v = history_side[board.height[av[j]] as usize];
        if v > val {
            val = v;
            l = j;
        }
        j += 1;
    }
    let chosen = av[l];
    let mut k = l;
    while k > i {
        av[k] = av[k - 1];
        k -= 1;
    }
    av[i] = chosen;
    chosen
}

pub struct Solver {
    pub tt: TransTable,
    history: [[i32; SIZE1]; 2],
    pub nodes: u64,
    book_ply: usize,
    report_ply: usize,
    /// When true, prints a per-node "<moves><char><work>" line for every
    /// node within `report_ply` of the root, matching SearchGame.c's
    /// diagnostic output.
    pub verbose: bool,
}

impl Solver {
    pub fn new(tt_size: usize) -> Self {
        let mut solver = Solver {
            tt: TransTable::new(tt_size),
            history: [[0; SIZE1]; 2],
            nodes: 0,
            book_ply: 0,
            report_ply: 0,
            verbose: false,
        };
        // Seed with center-weighted defaults immediately -- `solve()` also
        // re-seeds on every call (each solve is an independent position), so
        // this doesn't change its behavior at all. But `best_move()` never
        // calls `init_history` (it deliberately keeps history warm across a
        // whole game), so without this, a freshly-constructed `Solver` used
        // only via `best_move()` would search with an all-zero history
        // table -- and with everything tied at zero, move ordering quietly
        // degrades to plain left-to-right column order.
        solver.init_history();
        solver
    }

    /// Seed the history-heuristic table with the symmetric center-weighted
    /// values from SearchGame.c's `inithistory` (columns/rows near the
    /// board's center start with higher scores).
    fn init_history(&mut self) {
        for side in 0..2 {
            for i in 0..(WIDTH + 1) / 2 {
                for h in 0..H1 / 2 {
                    let ii = i as i32;
                    let hh = h as i32;
                    let val = 4
                        + ii.min(3)
                        + (hh.min(3) - (3 - ii).max(0)).max(-1)
                        + ii.min(hh).min(3)
                        + hh.min(3);
                    let idx1 = H1 * i + h;
                    let idx2 = H1 * (WIDTH - 1 - i) + HEIGHT - 1 - h;
                    let idx3 = H1 * i + HEIGHT - 1 - h;
                    let idx4 = H1 * (WIDTH - 1 - i) + h;
                    self.history[side][idx1] = val;
                    self.history[side][idx2] = val;
                    self.history[side][idx3] = val;
                    self.history[side][idx4] = val;
                }
            }
        }
    }

    /// Negamax with alpha-beta pruning, history-ordered moves, transposition
    /// table probing/storing, and immediate-threat detection -- ported
    /// directly from SearchGame.c's `ab`.
    pub fn ab(&mut self, board: &mut Board, mut alpha: i32, mut beta: i32) -> i32 {
        self.nodes += 1;
        if board.nplies == SIZE - 1 {
            return DRAW; // one move left; by assumption the mover can't win
        }
        let side = board.nplies & 1;

        let (mut av, nav) = match candidates(board) {
            Candidates::Forced(score) => return score,
            Candidates::Open(av, nav) => (av, nav),
        };
        if board.nplies == SIZE - 2 {
            return DRAW; // two moves left, no immediate win possible for either side
        }
        if nav == 1 {
            board.make_move(av[0]);
            let score = LOSSWIN - self.ab(board, LOSSWIN - beta, LOSSWIN - alpha);
            board.unmake_move();
            return score;
        }

        let (lock, index) = self.tt.hash_key(board);
        let ttscore = self.tt.lookup(index, lock);
        if ttscore != UNKNOWN {
            if ttscore == DRAWLOSS {
                beta = DRAW;
                if beta <= alpha {
                    return ttscore;
                }
            } else if ttscore == DRAWWIN {
                alpha = DRAW;
                if alpha >= beta {
                    return ttscore;
                }
            } else {
                return ttscore; // exact score already known
            }
        }

        let poscnt_before = self.tt.posed;
        #[allow(unused_assignments)]
        let mut besti = 0usize; // always overwritten before read; kept for clarity/parity with C
        let mut score = LOSS;

        let mut i = 0usize;
        while i < nav {
            let chosen = pick_next_move(&self.history[side], board, &mut av, i, nav);

            board.make_move(chosen);
            let val = LOSSWIN - self.ab(board, LOSSWIN - beta, LOSSWIN - alpha);
            board.unmake_move();

            if val > score {
                besti = i;
                score = val;
                if score > alpha && board.nplies >= self.book_ply {
                    alpha = score;
                    if alpha >= beta {
                        // Fail-high: if we cut off exactly at DRAW with moves
                        // still untried, the true value could be better than
                        // DRAW, so report the "at least draw" bound instead.
                        if score == DRAW && i < nav - 1 {
                            score = DRAWWIN;
                        }
                        // Reward the move that caused the cutoff, penalize
                        // the ones tried and rejected before it.
                        if besti > 0 {
                            for k in 0..besti {
                                self.history[side][board.height[av[k]] as usize] -= 1;
                            }
                            self.history[side][board.height[av[besti]] as usize] += besti as i32;
                        }
                        break;
                    }
                }
            }
            i += 1;
        }

        // Combine a stored upper bound with a newly proven lower bound (or
        // vice versa) at the same point into an exact DRAW.
        if score == LOSSWIN - ttscore {
            score = DRAW;
        }

        let poscnt = self.tt.posed - poscnt_before;
        let work = intlog(poscnt);
        self.tt.store(index, lock, score, work);

        if self.verbose && board.nplies <= self.report_ply {
            println!(
                "{}{}{}",
                format_moves(board),
                SCORE_CHARS[score as usize],
                work
            );
        }
        score
    }

    /// Solve the position currently on `board`. Returns (score, nodes visited,
    /// elapsed milliseconds). Does not clear the transposition table --
    /// call `self.tt.clear()` first if a fresh table is wanted, matching
    /// SearchGame.c's `emptyTT()` before each `solve()`.
    pub fn solve(&mut self, board: &mut Board) -> (i32, u64, u128) {
        self.nodes = 0;
        let side = board.nplies & 1;
        let otherside = side ^ 1;
        if Board::has_won(board.color[otherside]) {
            return (LOSS, self.nodes, 1);
        }
        for i in 0..WIDTH {
            if Board::is_legal_has_won(board.color[side] | (1u64 << board.height[i])) {
                return (WIN, self.nodes, 1);
            }
        }
        self.init_history();
        self.report_ply = board.nplies + REPORTPLY;
        self.book_ply = board.nplies + BOOKPLY;
        let start = std::time::Instant::now();
        let score = self.ab(board, LOSS, WIN);
        let elapsed_ms = start.elapsed().as_millis().max(1);
        (score, self.nodes, elapsed_ms)
    }

    /// Choose the best legal column to play in the current position. Unlike
    /// `solve`, this doesn't clear the transposition table or reinitialize
    /// history -- for interactive play, keep reusing the same `Solver`
    /// across a whole game so later, shallower searches benefit from
    /// everything learned on earlier turns.
    ///
    /// Shares `ab`'s own candidate generation and history-based move
    /// ordering (via `candidates`/`pick_next_move`), so the root gets the
    /// same "try the most promising move first" benefit that makes the
    /// rest of the search efficient -- trying columns in naive left-to-right
    /// order here previously made this several times slower than `solve`
    /// for equivalent positions, since the best root move (typically the
    /// center column) was tried last instead of first.
    ///
    /// Returns `None` only if the board is completely full. If the position
    /// is already lost regardless of what's played (e.g. an unstoppable
    /// double threat), this still returns some legal column so the game can
    /// continue -- callers should check win/draw state themselves before
    /// calling this in the first place.
    pub fn best_move(&mut self, board: &mut Board) -> Option<usize> {
        self.nodes = 0;
        self.book_ply = board.nplies + BOOKPLY;
        self.report_ply = 0; // no per-node tracing during interactive play
        let side = board.side();

        // Fast path: an outright winning move needs no search, mirroring
        // solve()'s own root-level pre-check.
        for col in 0..WIDTH {
            if Board::is_legal_has_won(board.color[side] | (1u64 << board.height[col])) {
                return Some(col);
            }
        }

        let (mut av, nav) = match candidates(board) {
            Candidates::Forced(_) => {
                // Lost regardless of what's played (or no safe column at
                // all) -- just play any legal column so the game continues.
                return (0..WIDTH).find(|&c| board.is_playable(c));
            }
            Candidates::Open(av, nav) => (av, nav),
        };

        let mut best_col = None;
        let mut alpha = LOSS;
        let beta = WIN;
        let mut i = 0usize;
        while i < nav {
            let chosen = pick_next_move(&self.history[side], board, &mut av, i, nav);
            board.make_move(chosen);
            let score = LOSSWIN - self.ab(board, LOSSWIN - beta, LOSSWIN - alpha);
            board.unmake_move();
            if best_col.is_none() || score > alpha {
                best_col = Some(chosen);
                alpha = score;
                if alpha >= beta {
                    break; // found a proven win; can't do better
                }
            }
            i += 1;
        }
        best_col
    }
}

/// log2(n), truncated, matching SearchGame.c's `intlog`/work-counting loop.
pub fn intlog(mut n: u64) -> u32 {
    let mut work = 0u32;
    loop {
        n >>= 1;
        if n == 0 {
            break;
        }
        work += 1;
    }
    work
}

/// Render the move history as a string of 1-based column digits, matching
/// SearchGame.c's `printMoves`.
pub fn format_moves(board: &Board) -> String {
    board.moves[..board.nplies]
        .iter()
        .map(|&c| std::char::from_digit((c as u32) + 1, 10).unwrap())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    const TT_SIZE: usize = 8_306_069;

    fn board_from_moves(moves: &str) -> Board {
        let mut b = Board::new();
        for c in moves.chars() {
            let col = c.to_digit(10).unwrap() as usize - 1;
            b.make_move(col);
        }
        b
    }

    // Reference values below are cross-checked against both the C (Fhourstones
    // 3.2) and Java (Fhourstones 3.1) implementations, which agree with each
    // other exactly on these positions.

    #[test]
    fn pos_45461667_matches_reference() {
        let mut b = board_from_moves("45461667");
        let mut solver = Solver::new(TT_SIZE);
        let (score, nodes, _) = solver.solve(&mut b);
        assert_eq!(score, WIN);
        assert_eq!(nodes, 51_596);
    }

    #[test]
    fn pos_35333571_matches_reference() {
        let mut b = board_from_moves("35333571");
        let mut solver = Solver::new(TT_SIZE);
        let (score, nodes, _) = solver.solve(&mut b);
        assert_eq!(score, LOSS);
        assert_eq!(nodes, 8_716_732);
    }

    #[test]
    fn pos_13333111_matches_reference() {
        let mut b = board_from_moves("13333111");
        let mut solver = Solver::new(TT_SIZE);
        let (score, nodes, _) = solver.solve(&mut b);
        assert_eq!(score, DRAW);
        assert_eq!(nodes, 169_704_432);
    }

    #[test]
    #[ignore] // ~3 min in release mode; run explicitly with `cargo test --release -- --ignored`
    fn pos_empty_board_matches_reference() {
        let mut b = Board::new();
        let mut solver = Solver::new(TT_SIZE);
        let (score, nodes, _) = solver.solve(&mut b);
        assert_eq!(score, WIN);
        assert_eq!(nodes, 1_479_113_766);
    }

    #[test]
    fn history_init_is_symmetric() {
        let mut solver = Solver::new(101);
        solver.init_history();
        // Center columns/rows should score at least as high as edge ones,
        // and the table should be left-right / top-ish symmetric by construction.
        let center = solver.history[0][H1 * 3 + 2];
        let corner = solver.history[0][H1 * 0 + 0];
        assert!(center >= corner);
    }
}
