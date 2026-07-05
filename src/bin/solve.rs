//! CLI driver, ported from SearchGame.c's `main`. Reads one position per
//! line of stdin (a string of column digits 1..=WIDTH; other characters are
//! ignored), solves it, and prints a report matching the original's format.
//!
//! Note: unlike the C version, a final input line with no trailing newline
//! is still processed (Rust's line iterator yields it normally); the C
//! version discards a trailing partial line if it hits EOF before a
//! newline. This is a deliberate, minor divergence -- it only affects
//! malformed/truncated input, never a normal run.

use connect_four::board::{Board, HEIGHT, SIZE1, WIDTH};
use connect_four::search::{Solver, format_moves, intlog};
use connect_four::tt::LOCKSIZE;
use std::io::{self, BufRead, Write};

const TT_SIZE: usize = 8_306_069;
//const TT_SIZE: usize = 15_999_961;
// should be a prime no less than about 2^{SIZE1-LOCKSIZE}, e.g.
// 4194301,8306069,8388593,15999961,33554393,67108859,134217689,268435399

const SCORE_CHARS: [char; 6] = ['#', '-', '<', '=', '>', '+'];

fn main() {
    debug_assert!(
        SIZE1 <= 64,
        "bitboard must fit in a u64 for this board size"
    );
    debug_assert!(
        TT_SIZE >= ((1usize << (SIZE1 - LOCKSIZE as usize)) * 31 / 32),
        "transposition table size is significantly smaller than recommended for this board size"
    );

    let stdout = io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "Fhourstones 3.2 (Rust)").unwrap();
    writeln!(out, "Boardsize = {}x{}", WIDTH, HEIGHT).unwrap();
    writeln!(
        out,
        "Using {} transposition table entries of size {} bytes.",
        TT_SIZE,
        std::mem::size_of::<u64>()
    )
    .unwrap();

    let mut solver = Solver::new(TT_SIZE);
    let stdin = io::stdin();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let mut board = Board::new();
        for c in line.chars() {
            if let Some(d) = c.to_digit(10) {
                if d >= 1 && d as usize <= WIDTH {
                    board.make_move(d as usize - 1);
                }
            }
        }

        write!(out, "\nSolving {}-ply position after ", board.nplies).unwrap();
        write!(out, "{}", format_moves(&board)).unwrap();
        writeln!(out, " . . .").unwrap();

        solver.tt.clear();
        let (score, nodes, ms) = solver.solve(&mut board);
        let work = intlog(solver.tt.posed);

        writeln!(
            out,
            "score = {} ({})  work = {}",
            score, SCORE_CHARS[score as usize], work
        )
        .unwrap();

        let kpos_per_sec = nodes as f64 / ms as f64;
        writeln!(
            out,
            "{} pos / {} msec = {:.1} Kpos/sec",
            nodes, ms, kpos_per_sec
        )
        .unwrap();

        if let Some(stats) = solver.tt.stats_line() {
            writeln!(out, "{}", stats).unwrap();
        }
    }
}
