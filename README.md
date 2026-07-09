# Connect Four

A Rust port of John Tromp's Fhourstones Connect Four solver +  
a small interactive terminal UI built on top of it, using the solver as a perfect-play opponent.

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

## Usage: play (interactive)

The play terminal app allows two players to play interactively.
The flags `--player1` and `--player2` configures who plays each side.
Choices are `human`, `perfect` (the solver), `minimax` (alpha-beta negamax [5]) and `mcts` (Monte Carlo Tree Search [6]). 

```
cargo run --release --bin play                                       # Human (red) vs perfect-play AI (yellow)
cargo run --release --bin play -- --player1 human --player2 human    # two humans
cargo run --release --bin play -- --player1 perfect --player2 perfect  # watch the perfect solver play itself
cargo run --release --bin play -- --player1 minimax --depth 6 --player2 mcts --mcts-millis 1000
cargo run --release --bin play -- --moves 4453 --player2 perfect     # start from a given position
```


## References 

1. [John Tromp's Connect Four page](https://tromp.github.io/c4/c4.html)
2. [Fhourstones Benchmark](https://tromp.github.io/c4/fhour.html)
3. ["The Complete Book of Connect Four",  James Dow Allen](https://fabpedigree.com/james/C4/c4_book.htm)
4. [Wikipedia's Conect Four](https://en.wikipedia.org/wiki/Connect_Four)
5. [Minimax](https://en.wikipedia.org/wiki/Minimax)
6. [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

