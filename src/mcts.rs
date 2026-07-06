//! Monte Carlo Tree Search: builds a tree via repeated
//! selection/expansion/simulation/backpropagation cycles until a wall-clock
//! time budget runs out, then plays the root child with the most visits.
//! Doesn't share anything with `search.rs` or `eval.rs` -- no heuristic
//! evaluation function is used at all; leaf values come purely from random
//! rollouts to a terminal (win/loss/draw) position.
//!
//! The tree is stored as a flat arena (`Vec<Node>`, indexed by position)
//! rather than `Rc<RefCell<..>>` links, which keeps mutation
//! straightforward and avoids interior-mutability bookkeeping.

use crate::board::{Board, SIZE, WIDTH};
use rand::Rng;
use rand::RngExt;
use rand::prelude::IndexedRandom;
use std::time::{Duration, Instant};

struct Node {
    parent: Option<usize>,
    /// The column played to reach this node from its parent; `None` only
    /// for the root, which represents the current actual position.
    move_played: Option<usize>,
    /// The side who played `move_played` to reach this node -- i.e. whose
    /// perspective `wins` is measured from. Meaningless for the root
    /// (`move_played` is `None` there, so it's never read).
    player_just_moved: usize,
    children: Vec<usize>,
    /// Columns not yet expanded into a child.
    untried: Vec<usize>,
    visits: u32,
    /// Accumulated reward for `player_just_moved` (1.0 per win, 0.5 per
    /// draw, 0.0 per loss, summed across all rollouts through this node).
    wins: f64,
}

pub struct MctsAi {
    pub time_budget: Duration,
    pub exploration: f64,
}

impl MctsAi {
    pub fn new(time_budget: Duration) -> Self {
        MctsAi {
            time_budget,
            exploration: std::f64::consts::SQRT_2,
        }
    }

    /// Search for `self.time_budget`, then return the most-visited root
    /// move (the standard "robust child" choice -- more stable than
    /// highest win-rate, since a move visited only once can have a
    /// misleadingly extreme rate). Returns `None` only if the board is
    /// completely full.
    pub fn best_move(&self, board: &Board) -> Option<usize> {
        let legal_at_root: Vec<usize> = (0..WIDTH).filter(|&c| board.is_playable(c)).collect();
        if legal_at_root.is_empty() {
            return None;
        }
        if legal_at_root.len() == 1 {
            return Some(legal_at_root[0]);
        }

        let mut arena = vec![Node {
            parent: None,
            move_played: None,
            player_just_moved: board.side() ^ 1, // arbitrary; never read for the root
            children: Vec::new(),
            untried: legal_at_root,
            visits: 0,
            wins: 0.0,
        }];

        let mut rng = rand::rng();
        let start = Instant::now();

        while start.elapsed() < self.time_budget {
            self.playout(&mut arena, board, &mut rng);
        }

        arena[0]
            .children
            .iter()
            .copied()
            .max_by_key(|&c| arena[c].visits)
            .and_then(|c| arena[c].move_played)
    }

    /// One selection/expansion/simulation/backpropagation cycle.
    fn playout(&self, arena: &mut Vec<Node>, root_board: &Board, rng: &mut impl Rng) {
        let mut node_idx = 0usize;
        let mut b = *root_board;

        // 1. Selection: descend via UCB1 while every child has been tried
        // and there's at least one child to descend into.
        while arena[node_idx].untried.is_empty() && !arena[node_idx].children.is_empty() {
            node_idx = self.select_child(arena, node_idx);
            b.make_move(arena[node_idx].move_played.unwrap());
        }

        // If the move that reached this node already ended the game, no
        // expansion or rollout is needed -- the outcome is already known.
        let already_decided = if arena[node_idx].move_played.is_some() {
            let pjm = arena[node_idx].player_just_moved;
            if Board::has_won(b.color[pjm]) {
                Some(Some(pjm))
            } else if b.nplies == SIZE {
                Some(None)
            } else {
                None
            }
        } else {
            None
        };

        let result = match already_decided {
            Some(known) => known,
            None => {
                // 2. Expansion (guaranteed possible: a non-terminal node
                // always has at least one untried move here, since a node
                // with no children and no untried moves would have been
                // caught by `already_decided` above when it was created).
                let i = rng.random_range(0..arena[node_idx].untried.len());
                let mv = arena[node_idx].untried.swap_remove(i);
                let mover = b.side();
                b.make_move(mv);
                let child_legal: Vec<usize> = (0..WIDTH).filter(|&c| b.is_playable(c)).collect();
                let child_idx = arena.len();
                arena.push(Node {
                    parent: Some(node_idx),
                    move_played: Some(mv),
                    player_just_moved: mover,
                    children: Vec::new(),
                    untried: child_legal,
                    visits: 0,
                    wins: 0.0,
                });
                arena[node_idx].children.push(child_idx);
                node_idx = child_idx;

                // 3. Simulation: if the new node itself is already terminal,
                // use that; otherwise roll out with random play.
                if Board::has_won(b.color[mover]) {
                    Some(mover)
                } else if b.nplies == SIZE {
                    None
                } else {
                    Self::rollout(b, rng)
                }
            }
        };

        // 4. Backpropagation.
        let mut cur = Some(node_idx);
        while let Some(idx) = cur {
            let node = &mut arena[idx];
            node.visits += 1;
            if node.move_played.is_some() {
                node.wins += Self::reward(result, node.player_just_moved);
            }
            cur = node.parent;
        }
    }

    fn select_child(&self, arena: &[Node], idx: usize) -> usize {
        let parent_visits = arena[idx].visits as f64;
        arena[idx]
            .children
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let ucb = |c: usize| -> f64 {
                    let n = &arena[c];
                    if n.visits == 0 {
                        return f64::INFINITY;
                    }
                    let exploitation = n.wins / n.visits as f64;
                    let exploration =
                        self.exploration * (parent_visits.ln() / n.visits as f64).sqrt();
                    exploitation + exploration
                };
                ucb(a).partial_cmp(&ucb(b)).unwrap()
            })
            .expect("select_child is only called when children is non-empty")
    }

    /// Play uniformly random legal moves from `b` until the game ends.
    /// Returns the winning side, or `None` for a draw.
    fn rollout(mut b: Board, rng: &mut impl Rng) -> Option<usize> {
        loop {
            let mover = b.side();
            let legal: Vec<usize> = (0..WIDTH).filter(|&c| b.is_playable(c)).collect();
            let Some(&mv) = legal.choose(rng) else {
                return None; // no legal moves left: draw
            };
            b.make_move(mv);
            if Board::has_won(b.color[mover]) {
                return Some(mover);
            }
            if b.nplies == SIZE {
                return None;
            }
        }
    }

    fn reward(winner: Option<usize>, side: usize) -> f64 {
        match winner {
            Some(w) if w == side => 1.0,
            Some(_) => 0.0,
            None => 0.5,
        }
    }
}

impl Default for MctsAi {
    fn default() -> Self {
        MctsAi::new(Duration::from_secs(2))
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
    fn returns_a_legal_move_from_empty_board() {
        let b = Board::new();
        let ai = MctsAi::new(Duration::from_millis(200));
        let mv = ai.best_move(&b).expect("empty board has legal moves");
        assert!(b.is_playable(mv));
    }

    #[test]
    fn single_legal_column_is_returned_without_searching() {
        // Fill every column but one.
        let mut b = Board::new();
        for col in [0, 1, 2, 4, 5, 6] {
            for _ in 0..6 {
                b.make_move(col);
            }
        }
        let ai = MctsAi::new(Duration::from_millis(50));
        assert_eq!(ai.best_move(&b), Some(3));
    }

    #[test]
    fn finds_immediate_win_given_enough_budget() {
        let mut b = board_from_moves("");
        for &(col, opponent_col) in &[(0, 4), (1, 4), (2, 4)] {
            b.make_move(col);
            b.make_move(opponent_col);
        }
        let ai = MctsAi::new(Duration::from_millis(500));
        assert_eq!(ai.best_move(&b), Some(3));
    }
}
