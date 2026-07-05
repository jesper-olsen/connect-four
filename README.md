# Connect Four

A Rust port of John Tromp's Fhourstones Connect Four solver +  
a small interactive terminal UI built on top of it, using the solver as a
perfect-play opponent.

The solver is a negamax/alpha-beta engine with a transposition table and history-heuristic move
ordering. It does not use a heuristic evaluation function - it searches the game tree from 
root to terminals and finds exact scores. For a standard 7x6 connect four board, the game tree has4.5 trillion nodes (4,531,985,219,092).

![Gif Image interactive play](demo.gif)

## Verification

Checked against the original C (v3.2) and Java (v3.1) reference implementations, which agree
with each other exactly:

Position (moves played)          | Score | Nodes searched
:----------                      | :---- | -------------: 
45461667                         | Win   |        51,596  
35333571                         | Loss  |     8,716,732
13333111                         | Draw  |   169,704,432
(empty board, full 42-ply solve) | Win   | 1,479,113,766

Run the exact-node-count regression tests with:
```
cargo test --release --lib
```

## Usage: solve (benchmark / analysis)

Reads one position per line from stdin, as a string of column digits
(1–7, left to right); other characters are ignored, and an empty line
solves the starting position. This matches the original Fhourstones CLI
input format, so existing test/benchmark input files work unchanged.

```
echo "4453" | cargo run --release --bin solve
```

For each position it reports the game-theoretic score, log2 of positions
stored in the transposition table ("work"), total nodes searched, search
speed, and a histogram of stored transposition-table entries by score.

Usage: play (interactive)

```
cargo run --release --bin play                                     # Human (red) vs AI (yellow)
cargo run --release --bin play -- --player1 human --player2 human  # two humans
cargo run --release --bin play -- --player1 ai --player2 ai        # watch the AI play itself
cargo run --release --bin play -- --moves 4453 --player2 ai        # Player 1 coins in rows 4 & 5, Player 2 coins in rows 4 & 3
```

The AI plays perfectly, which means genuinely slow the first time it has to
search deep into an empty or near-empty board — expect the very first move
from a fresh position to take tens of seconds.

## References 

1. [John Trump's Connect Four page](https://tromp.github.io/c4/c4.html)
2. [Fhourstones Benchmark](https://tromp.github.io/c4/fhour.html)
3. ["The Complete Book of Connect Four",  James Dow Allen](https://fabpedigree.com/james/C4/c4_book.htm)
4. [Wikipedia's Conect Four](https://en.wikipedia.org/wiki/Connect_Four)

## License

The engine is a derivative of John Tromp's original Fhourstones C
implementation, which carries this notice:

> This software is copyright (c) 1996-2005 by John Tromp.
This notice must not be removed.
This software must not be sold for profit.
You may redistribute if your distributees have the same rights and
restrictions.

Those terms apply to this port as well.

