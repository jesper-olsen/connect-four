//! Transposition table, ported from TransGame.c.
//!
//! Each 64-bit slot holds two candidate entries for the same bucket:
//! a "big" entry (kept when it represents at least as much search work as
//! whatever's already stored, or is an exact match for the same position)
//! and a "new" entry (always overwritten on a miss, cheap fallback storage).
//!
//! Bit layout per slot (64 bits total) -- chosen to match the layout already
//! verified correct in the Haskell port's `GameTreeSearch.hs`, not C's raw
//! struct memory layout (which is compiler/endianness-dependent and not
//! behaviorally significant -- only the replacement policy and the scores
//! it produces need to match C, not the physical bit order):
//!
//!   bits  0..26  new_lock   (26 bits)
//!   bits 26..29  new_score  (3 bits)
//!   bits 29..32  big_score  (3 bits)
//!   bits 32..58  big_lock   (26 bits)
//!   bits 58..64  big_work   (6 bits)

use crate::board::{Board, H1, SIZE1};

pub const LOCKSIZE: u32 = 26;
const LOCK_MASK: u64 = (1u64 << LOCKSIZE) - 1;
const NEWSIZE: u32 = LOCKSIZE + 3; // width of the new_lock+new_score region
const SCORE_MASK: u64 = 0x7;
const NEW_MASK: u64 = (1u64 << NEWSIZE) - 1;
const BIG_MASK: u64 = !NEW_MASK;

/// Number of opening plies for which column-mirror symmetry is checked
/// when hashing (matches TransGame.c's SYMMREC).
pub const SYMMREC: usize = 10;

// Score values, matching TransGame.c exactly.
pub const UNKNOWN: i32 = 0;
pub const LOSS: i32 = 1;
pub const DRAWLOSS: i32 = 2;
pub const DRAW: i32 = 3;
pub const DRAWWIN: i32 = 4;
pub const WIN: i32 = 5;
pub const LOSSWIN: i32 = 6;

#[inline]
fn big_work(x: u64) -> u32 {
    (x >> (2 * NEWSIZE)) as u32
}
#[inline]
fn big_lock(x: u64) -> u64 {
    (x >> (3 + NEWSIZE)) & LOCK_MASK
}
#[inline]
fn big_score(x: u64) -> i32 {
    ((x >> NEWSIZE) & SCORE_MASK) as i32
}
#[inline]
fn new_lock(x: u64) -> u64 {
    x & LOCK_MASK
}
#[inline]
fn new_score(x: u64) -> i32 {
    ((x >> LOCKSIZE) & SCORE_MASK) as i32
}

pub struct TransTable {
    entries: Vec<u64>,
    pub posed: u64, // count of store() calls, used for "work" reporting
}

impl TransTable {
    pub fn new(size: usize) -> Self {
        TransTable {
            entries: vec![0u64; size],
            posed: 0,
        }
    }

    pub fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            *e = 0;
        }
        self.posed = 0;
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up `lock` at bucket `index`. Returns UNKNOWN if neither the
    /// "big" nor "new" slot at this bucket matches the lock.
    pub fn lookup(&self, index: usize, lock: u64) -> i32 {
        let he = self.entries[index];
        if lock == big_lock(he) {
            big_score(he)
        } else if lock == new_lock(he) {
            new_score(he)
        } else {
            UNKNOWN
        }
    }

    /// Store `score` for `lock` at bucket `index`, having invested `work`
    /// (log2 of positions searched below this node) to obtain it.
    ///
    /// Replacement policy (matches `transtore` in TransGame.c): the "big"
    /// slot is overwritten when it already holds this same lock, or when
    /// the new work is at least as deep as what's currently recorded there;
    /// otherwise the position is written to the cheaper "new" slot instead,
    /// leaving "big" untouched.
    pub fn store(&mut self, index: usize, lock: u64, score: i32, work: u32) {
        self.posed += 1;
        let he = self.entries[index];
        let new_he = if lock == big_lock(he) || work >= big_work(he) {
            let packed_work_lock = ((work as u64) << LOCKSIZE) | lock;
            let packed_with_score = (packed_work_lock << 3) | (score as u64);
            (packed_with_score << NEWSIZE) | (he & NEW_MASK)
        } else {
            (he & BIG_MASK) | ((score as u64) << LOCKSIZE) | lock
        };
        self.entries[index] = new_he;
    }

    /// Compute (lock, bucket index) for the current board position, matching
    /// TransGame.c's `hash()`. For the first SYMMREC plies, the position is
    /// normalized against its column-mirror image (whichever encodes smaller)
    /// so mirror-symmetric positions share a table entry.
    pub fn hash_key(&self, board: &Board) -> (u64, usize) {
        let mut htemp = board.position_code();
        if board.nplies < SYMMREC {
            let mirrored = mirror_columns(htemp);
            if mirrored < htemp {
                htemp = mirrored;
            }
        }
        debug_assert!(SIZE1 > LOCKSIZE as usize, "assumes SIZE1 > LOCKSIZE");
        let lock = htemp >> (SIZE1 - LOCKSIZE as usize);
        let index = (htemp % self.entries.len() as u64) as usize;
        (lock, index)
    }

    /// Fraction of stored entries at each score value (LOSS..=WIN), matching
    /// TransGame.c's `htstat`. Returns None if the table has no entries yet.
    pub fn stats(&self) -> Option<[f64; 6]> {
        let mut counts = [0u64; 6]; // indices 0..=5, only 1..=5 (LOSS..=WIN) used
        for &he in &self.entries {
            let bl = big_lock(he);
            if bl != 0 {
                counts[big_score(he) as usize] += 1;
            }
            let nl = new_lock(he);
            if nl != 0 {
                counts[new_score(he) as usize] += 1;
            }
        }
        let total: u64 = counts[LOSS as usize..=WIN as usize].iter().sum();
        if total == 0 {
            return None;
        }
        let mut fractions = [0.0; 6];
        for (i, &c) in counts.iter().enumerate() {
            fractions[i] = c as f64 / total as f64;
        }
        Some(fractions)
    }

    /// Formatted like TransGame.c's `htstat` printf line, e.g.
    /// "- 0.281  < 0.000  = 0.001  > 0.001  + 0.716".
    pub fn stats_line(&self) -> Option<String> {
        self.stats().map(|f| {
            format!(
                "- {:.3}  < {:.3}  = {:.3}  > {:.3}  + {:.3}",
                f[LOSS as usize],
                f[DRAWLOSS as usize],
                f[DRAW as usize],
                f[DRAWWIN as usize],
                f[WIN as usize]
            )
        })
    }
}

/// Reverse the WIDTH columns of a position code (each column is H1 bits wide).
fn mirror_columns(mut htemp: u64) -> u64 {
    let mut mirrored = 0u64;
    let col1 = (1u64 << H1) - 1;
    while htemp != 0 {
        mirrored = (mirrored << H1) | (htemp & col1);
        htemp >>= H1;
    }
    mirrored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn empty_table_lookup_is_unknown() {
        let tt = TransTable::new(101);
        assert_eq!(tt.lookup(0, 12345), UNKNOWN);
    }

    #[test]
    fn store_then_lookup_round_trip() {
        let mut tt = TransTable::new(101);
        tt.store(5, 999, WIN, 3);
        assert_eq!(tt.lookup(5, 999), WIN);
        // Different lock at the same bucket must miss.
        assert_eq!(tt.lookup(5, 1000), UNKNOWN);
    }

    #[test]
    fn low_work_store_goes_to_new_slot_not_big() {
        let mut tt = TransTable::new(101);
        tt.store(0, 111, WIN, 10); // occupies "big" (nothing there yet, work>=0)
        tt.store(0, 222, LOSS, 2); // lower work, different lock -> "new" slot only
        assert_eq!(tt.lookup(0, 111), WIN); // big slot survives
        assert_eq!(tt.lookup(0, 222), LOSS); // new slot holds the second entry
    }

    #[test]
    fn higher_work_store_replaces_big_slot() {
        let mut tt = TransTable::new(101);
        tt.store(0, 111, WIN, 2);
        tt.store(0, 222, LOSS, 9); // higher work -> takes over "big"
        assert_eq!(tt.lookup(0, 222), LOSS);
        assert_eq!(tt.lookup(0, 111), UNKNOWN); // evicted from big, wasn't in new
    }

    #[test]
    fn same_lock_update_replaces_big_regardless_of_work() {
        let mut tt = TransTable::new(101);
        tt.store(0, 111, DRAW, 20);
        tt.store(0, 111, WIN, 0); // same lock, low work -> still updates big
        assert_eq!(tt.lookup(0, 111), WIN);
    }

    #[test]
    fn clear_resets_table_and_posed_count() {
        let mut tt = TransTable::new(50);
        tt.store(0, 111, WIN, 5);
        assert_eq!(tt.posed, 1);
        tt.clear();
        assert_eq!(tt.posed, 0);
        assert_eq!(tt.lookup(0, 111), UNKNOWN);
    }

    #[test]
    fn hash_key_stable_for_same_position() {
        let tt = TransTable::new(8306069);
        let b = Board::new();
        let (lock1, idx1) = tt.hash_key(&b);
        let (lock2, idx2) = tt.hash_key(&b);
        assert_eq!((lock1, idx1), (lock2, idx2));
    }

    #[test]
    fn hash_key_index_in_bounds() {
        let tt = TransTable::new(8306069);
        let mut b = Board::new();
        for &col in &[3usize, 2, 4, 3, 5] {
            b.make_move(col);
            let (_, idx) = tt.hash_key(&b);
            assert!(idx < tt.len());
        }
    }

    #[test]
    fn stats_none_when_empty() {
        let tt = TransTable::new(50);
        assert!(tt.stats().is_none());
    }

    #[test]
    fn stats_reflect_stored_scores() {
        let mut tt = TransTable::new(50);
        tt.store(0, 1, WIN, 1);
        tt.store(1, 2, LOSS, 1);
        let f = tt.stats().unwrap();
        assert!((f[WIN as usize] - 0.5).abs() < 1e-9);
        assert!((f[LOSS as usize] - 0.5).abs() < 1e-9);
    }
}
