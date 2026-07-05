//! Interactive Connect Four TUI, built on the same board/search/tt engine
//! as `solve.rs`. Unlike that benchmark driver, this keeps a single
//! `Solver` (transposition table + history heuristic) alive for the whole
//! game rather than clearing it before each move -- most of the tree
//! explored on one turn remains valid on the next, so later moves get
//! dramatically faster as the game goes on. The AI runs on a background
//! thread so the UI stays responsive; expect the very first AI move from
//! an empty board to take roughly as long as a full from-scratch solve
//! (tens of seconds), with subsequent moves much quicker.

use c4::board::{Board, H1, HEIGHT, SIZE, WIDTH};
use c4::search::Solver;
use ratatui::crossterm::event::{self, Event, KeyCode};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Style, Stylize},
    widgets::Paragraph,
};
use std::sync::mpsc;
use std::{error::Error, io, time::Duration};

const TT_SIZE: usize = 8_306_069;
//const TT_SIZE: usize = 15_999_961;
// should be a prime no less than about 2^{SIZE1-LOCKSIZE}, e.g.
// 4194301,8306069,8388593,15999961,33554393,67108859,134217689,268435399


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    fn from_side(side: usize) -> Self {
        if side == 0 {
            Player::Player1
        } else {
            Player::Player2
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Controller {
    Human,
    Ai,
}

#[derive(Clone, Copy)]
struct FallingCoin {
    col: usize,
    animation_row: f32,
    target_row: usize,
}

pub struct App {
    pub board: Board,
    pub controllers: [Controller; 2],
    pub wins: [u32; 2],
    pub winner: Option<Player>,
    pub is_draw: bool,
    pub selected_column: usize,
    falling_coin: Option<FallingCoin>,
}

impl App {
    pub fn new(controllers: [Controller; 2]) -> Self {
        App {
            board: Board::new(),
            controllers,
            wins: [0, 0],
            winner: None,
            is_draw: false,
            selected_column: WIDTH / 2,
            falling_coin: None,
        }
    }

    pub fn current_turn(&self) -> Player {
        Player::from_side(self.board.side())
    }

    /// Piece at (row, col); row 0 is the bottom row, matching the engine's
    /// own bit layout (bit index = H1*col + row).
    pub fn cell(&self, row: usize, col: usize) -> Option<Player> {
        let bit = 1u64 << (H1 * col + row);
        if self.board.color[0] & bit != 0 {
            Some(Player::Player1)
        } else if self.board.color[1] & bit != 0 {
            Some(Player::Player2)
        } else {
            None
        }
    }

    pub fn move_left(&mut self) {
        if self.selected_column > 0 {
            self.selected_column -= 1;
        }
    }

    pub fn move_right(&mut self) {
        if self.selected_column + 1 < WIDTH {
            self.selected_column += 1;
        }
    }

    /// Start dropping a coin in the selected column, if legal and nothing's
    /// already falling. The move isn't committed to the board until the
    /// drop animation finishes (see `update`).
    pub fn try_drop_coin(&mut self) {
        if self.falling_coin.is_some() || self.winner.is_some() || self.is_draw {
            return;
        }
        let col = self.selected_column;
        if !self.board.is_playable(col) {
            return;
        }
        // height[col] is a bit index (H1*col + row); mod H1 recovers the row
        // since H1*col is an exact multiple of H1.
        let target_row = (self.board.height[col] as usize) % H1;
        self.falling_coin = Some(FallingCoin {
            col,
            animation_row: (HEIGHT - 1) as f32,
            target_row,
        });
    }

    pub fn try_restart(&mut self) {
        self.board.reset();
        self.winner = None;
        self.is_draw = false;
        self.falling_coin = None;
        self.selected_column = WIDTH / 2;
    }

    /// Advance the falling-coin animation by one tick. Commits the move to
    /// the board and checks for a win/draw once it lands.
    pub fn update(&mut self) {
        let Some(coin) = &mut self.falling_coin else {
            return;
        };
        if coin.animation_row > coin.target_row as f32 {
            coin.animation_row = (coin.animation_row - 0.9).max(coin.target_row as f32);
            return;
        }

        let col = coin.col;
        let mover = self.current_turn();
        self.board.make_move(col);
        self.falling_coin = None;

        let moved_color = self.board.color[mover as usize];
        if Board::has_won(moved_color) {
            self.winner = Some(mover);
            self.wins[mover as usize] += 1;
        } else if self.board.nplies == SIZE {
            self.is_draw = true;
        }
    }
}

pub fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<(), Box<dyn Error>> {
    let tick_rate = Duration::from_millis(33);

    // While Some, an AI search is running on a background thread. The
    // Solver is moved into the thread for the duration of the search (see
    // below) and handed back over the channel alongside the chosen move,
    // so its transposition table and history heuristic persist for the
    // rest of the game.
    let mut solver: Option<Solver> = Some(Solver::new(TT_SIZE));
    let mut ai_move_rx: Option<mpsc::Receiver<(Option<usize>, Solver)>> = None;

    loop {
        terminal.draw(|f| ui(f, app, ai_move_rx.is_some()))?;

        let is_human_turn = app.controllers[app.current_turn() as usize] == Controller::Human;

        if event::poll(tick_rate)?
            && let Event::Key(key) = event::read()?
        {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Left | KeyCode::Char('h') if is_human_turn => app.move_left(),
                KeyCode::Right | KeyCode::Char('l') if is_human_turn => app.move_right(),
                KeyCode::Enter | KeyCode::Char(' ') if is_human_turn => app.try_drop_coin(),
                KeyCode::Char('r') => app.try_restart(),
                _ => {}
            }
        }

        app.update();

        if app.falling_coin.is_none() && app.winner.is_none() && !app.is_draw {
            let active_controller = app.controllers[app.current_turn() as usize];
            if active_controller == Controller::Ai && ai_move_rx.is_none() {
                let board = app.board; // Board is Copy
                let mut s = solver
                    .take()
                    .expect("solver is only absent while a search is in flight");
                let (tx, rx) = mpsc::channel();
                std::thread::spawn(move || {
                    let mut b = board;
                    let col = s.best_move(&mut b);
                    let _ = tx.send((col, s));
                });
                ai_move_rx = Some(rx);
            } else if let Some(rx) = &ai_move_rx {
                if let Ok((chosen, returned_solver)) = rx.try_recv() {
                    solver = Some(returned_solver);
                    if let Some(col) = chosen {
                        app.selected_column = col;
                        app.try_drop_coin();
                    }
                    ai_move_rx = None;
                }
            }
        }
    }
    Ok(())
}

fn ui(f: &mut ratatui::Frame, app: &App, ai_thinking: bool) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Min(14),
            Constraint::Length(2),
        ])
        .split(f.area());

    let width = chunks[0].width as usize;
    let title_str = "  Connect 4 - Perfect-Play Engine";
    let wins_str = format!("[Wins: P1: {} | P2: {}]  ", app.wins[0], app.wins[1]);
    let mid_padding = width.saturating_sub(title_str.len() + wins_str.len());
    let header_text = format!(
        "{}{}{}\n{}",
        title_str,
        " ".repeat(mid_padding),
        wins_str,
        "=".repeat(width)
    );
    f.render_widget(Paragraph::new(header_text).cyan(), chunks[0]);

    let status_widget = if let Some(winner) = app.winner {
        let (win_msg, color) = match winner {
            Player::Player1 => ("🏆 PLAYER 1 WINS! Press [r] to restart", Color::Red),
            Player::Player2 => ("🏆 PLAYER 2 WINS! Press [r] to restart", Color::Yellow),
        };
        Paragraph::new(format!("\n{}", win_msg))
            .style(Style::default().fg(color).bold().on_dark_gray())
            .alignment(Alignment::Center)
    } else if app.is_draw {
        Paragraph::new("\n🤝 IT'S A DRAW! Press [r] to restart")
            .style(Style::default().fg(Color::White).bold())
            .alignment(Alignment::Center)
    } else {
        let turn_text = match app.current_turn() {
            Player::Player1 => "Player 1's Turn (🔴)",
            Player::Player2 => "Player 2's Turn (🟡)",
        };
        let suffix = if ai_thinking {
            "  🤔 thinking..."
        } else {
            ""
        };
        Paragraph::new(format!("\n{}{}", turn_text, suffix))
            .style(Style::default())
            .alignment(Alignment::Center)
    };
    f.render_widget(status_widget, chunks[1]);

    let arrow_padding = " ".repeat((app.selected_column * 5) + 3);
    let arrow_text = if app.falling_coin.is_none() && app.winner.is_none() && !app.is_draw {
        "▼"
    } else {
        " "
    };
    let turn_color = match app.current_turn() {
        Player::Player1 => Color::Red,
        Player::Player2 => Color::Yellow,
    };
    let selector = Paragraph::new(format!("{}{}", arrow_padding, arrow_text))
        .style(Style::default().fg(turn_color));
    f.render_widget(selector, chunks[2]);

    let mut board_text = String::new();
    let grid_border = "+----+----+----+----+----+----+----+\n";
    board_text.push_str(grid_border);

    // Display rows top-to-bottom; the engine's row 0 is the bottom row, so
    // walk display_row 0..HEIGHT and map to board row (HEIGHT-1-display_row).
    for display_row in 0..HEIGHT {
        let row = HEIGHT - 1 - display_row;
        for col in 0..WIDTH {
            let mut token = "    ";
            if let Some(player) = app.cell(row, col) {
                token = match player {
                    Player::Player1 => " 🔴 ",
                    Player::Player2 => " 🟡 ",
                };
            } else if let Some(falling) = app.falling_coin {
                if falling.col == col && falling.animation_row.round() as usize == row {
                    token = match app.current_turn() {
                        Player::Player1 => " 🔴 ",
                        Player::Player2 => " 🟡 ",
                    };
                }
            }
            board_text.push_str(&format!("|{}", token));
        }
        board_text.push_str("|\n");
        board_text.push_str(grid_border);
    }
    f.render_widget(Paragraph::new(board_text), chunks[3]);

    let hl = "-".repeat(width);
    let controls_text = format!(
        "{hl}\nControls: [←/h] Left | [→/l] Right | [Enter/Space] Drop | [r] Restart | [q] Quit"
    );
    let controls = Paragraph::new(controls_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);
    f.render_widget(controls, chunks[4]);
}

fn main() -> Result<(), Box<dyn Error>> {
    use ratatui::crossterm::{
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Human (Player 1) vs. AI (Player 2) by default.
    let mut app = App::new([Controller::Human, Controller::Ai]);
    let result = run_app(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}
