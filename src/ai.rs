//! A single type `play.rs` can hold regardless of which algorithm is
//! backing a given AI-controlled seat. An enum rather than `Box<dyn Trait>`
//! since there are exactly three known implementations and this sidesteps
//! any `Send`/object-safety questions around handing the engine off to a
//! background search thread.

use crate::board::Board;
use crate::mcts::MctsAi;
use crate::minimax::MinimaxAi;
use crate::search::Solver;
use std::time::Duration;

/// A cheap, `Copy`-able tag for "which algorithm" -- distinct from
/// `AiEngine` itself, which owns the (potentially large) runtime state for
/// whichever algorithm is chosen. Callers like `play.rs` want to store and
/// compare "what kind of opponent is seat N" independently of whether an
/// actual engine instance has been allocated yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AiKind {
    /// Exact game-theoretic solver (`search::Solver`).
    Perfect,
    /// Depth-limited minimax with a heuristic evaluation function.
    Minimax,
    /// Monte Carlo Tree Search with a wall-clock time budget.
    Mcts,
}

/// Construction parameters for each possible `AiKind`. Fields not relevant
/// to a given kind are simply ignored when building that kind.
pub struct AiConfig {
    pub tt_size: usize,
    pub minimax_depth: u32,
    pub mcts_budget: Duration,
}

pub enum AiEngine {
    /// Exact game-theoretic solver (see `search.rs`) -- never wrong, but
    /// the first move or two from a near-empty board can take tens of
    /// seconds. Keeps its transposition table/history heuristic warm across
    /// the whole game (see `Solver::best_move`'s own documentation).
    Perfect(Solver),
    /// Depth-limited minimax with a heuristic evaluation function --
    /// bounded, predictable cost; not always optimal.
    Minimax(MinimaxAi),
    /// Monte Carlo Tree Search with a wall-clock time budget.
    Mcts(MctsAi),
}

impl AiEngine {
    /// Build the engine for a given kind, using whichever fields of
    /// `config` that kind actually needs.
    pub fn new(kind: AiKind, config: &AiConfig) -> Self {
        match kind {
            AiKind::Perfect => AiEngine::Perfect(Solver::new(config.tt_size)),
            AiKind::Minimax => AiEngine::Minimax(MinimaxAi::new(config.minimax_depth)),
            AiKind::Mcts => AiEngine::Mcts(MctsAi::new(config.mcts_budget)),
        }
    }

    /// Choose a column to play. Returns `None` only if the board is
    /// completely full.
    pub fn best_move(&mut self, board: &Board) -> Option<usize> {
        match self {
            AiEngine::Perfect(solver) => {
                let mut b = *board;
                solver.best_move(&mut b)
            }
            AiEngine::Minimax(ai) => ai.best_move(board),
            AiEngine::Mcts(ai) => ai.best_move(board),
        }
    }
}
