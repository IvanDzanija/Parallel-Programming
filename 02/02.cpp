#include <mpi.h>

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

using i32 = int32_t;
using i8 = int8_t;
using u8 = uint8_t;

// Constants
static constexpr double DOUBLE_INF = std::numeric_limits<double>::infinity();
static constexpr size_t ROWS = 6;
static constexpr size_t COLS = 7;
static constexpr size_t MIN_DEPTH = 10;
static constexpr size_t TASK_SPLIT_DEPTH = 3;  // 7^3 = 343

// Game state enums
enum class Turn : u8 { HUMAN = 1, COMPUTER = 2 };
Turn swap(Turn t) { return (t == Turn::HUMAN) ? Turn::COMPUTER : Turn::HUMAN; }
enum class BoardCell : u8 { EMPTY = 0, HUMAN = 1, COMPUTER = 2 };
BoardCell place_turn(Turn t) {
  return (t == Turn::HUMAN) ? BoardCell::HUMAN : BoardCell::COMPUTER;
}

struct Board {
  std::array<std::array<BoardCell, COLS>, ROWS> grid;

  void display() const {
    for (const auto &row : grid) {
      for (const auto &coll : row) {
        // Player = X, Computer = O, Empty = .
        char c = (coll == BoardCell::HUMAN) ? 'X'
                                            : (coll == BoardCell::COMPUTER ? 'O' : '.');
        std::cout << c << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "0 1 2 3 4 5 6" << std::endl;
    std::cout << std::endl;
  }

  BoardCell find_winner(i32 r, i32 c) const {
    static std::array<std::pair<i8, i8>, 4> DIRECTIONS = {
        {{0, 1}, {1, 0}, {1, 1}, {1, -1}}};
    BoardCell player = grid[r][c];
    for (auto [dr, dc] : DIRECTIONS) {
      i32 count = 1;
      for (i32 i = 1; i < 4; ++i) {
        i32 rr = r + dr * i;
        i32 cc = c + dc * i;
        if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && grid[rr][cc] == player) {
          ++count;
        } else {
          break;
        }
      }
      for (i32 i = 1; i < 4 && std::make_pair(dr, dc) != std::make_pair(0, 1); ++i) {
        i32 rr = r - dr * i;
        i32 cc = c - dc * i;
        if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && grid[rr][cc] == player) {
          ++count;
        } else {
          break;
        }
      }
      if (count >= 4) {
        return player;
      }
    }
    return BoardCell::EMPTY;
  }

  double evaluate(Turn turn, i32 prev_r, i32 prev_c, i32 depth) {
    BoardCell winner = find_winner(prev_r, prev_c);
    if (winner == BoardCell::HUMAN) {
      return -1.F;
    }
    if (winner == BoardCell::COMPUTER) {
      return 1.F;
    }
    if (depth == 0) {
      return 0.F;
    }

    double sum = 0;
    i32 moves = 0;
    for (i32 c = 0; c < COLS; ++c) {
      i32 r = ROWS - 1;
      // Find the lowest empty cell in column c
      for (; r >= 0 && grid[r][c] != BoardCell::EMPTY; --r);

      if (r != -1) {
        ++moves;
        grid[r][c] = (turn == Turn::COMPUTER) ? BoardCell::COMPUTER : BoardCell::HUMAN;
        double res = evaluate(swap(turn), r, c, depth - 1);
        grid[r][c] = BoardCell::EMPTY;  // Undo move
        if (turn == Turn::COMPUTER && res == 1.F) {
          return 1.F;
        } else if (turn == Turn::HUMAN && res == -1.F) {
          return -1.F;
        }
        sum += res;
      }
    }
    return moves == 0 ? 0 : sum / moves;
  }
};

std::optional<double> execute_task_by_id(i32 t_id, Board temp_board) {
  i32 temp_tid = t_id;
  Turn current_turn = Turn::COMPUTER;
  i32 last_r = -1, last_c = -1;

  // Decode task ID into a sequence of moves
  std::vector<i32> moves(TASK_SPLIT_DEPTH);
  for (size_t d = 0; d < TASK_SPLIT_DEPTH; ++d) {
    moves[d] = temp_tid % COLS;
    temp_tid /= COLS;
  }

  for (size_t d = 0; d < TASK_SPLIT_DEPTH; ++d) {
    i32 c = moves[d];
    i32 r = ROWS - 1;
    for (; r >= 0 && temp_board.grid[r][c] != BoardCell::EMPTY; --r);

    if (r == -1) {
      return std::nullopt;  // Invalid move, column is full
    }

    temp_board.grid[r][c] = place_turn(current_turn);
    last_r = r;
    last_c = c;

    BoardCell winner = temp_board.find_winner(r, c);
    if (winner == BoardCell::COMPUTER) {
      return 1.F;
    }
    if (winner == BoardCell::HUMAN) {
      return -1.F;
    }
    current_turn = swap(current_turn);
  }

  return temp_board.evaluate(current_turn, last_r, last_c,
                             MIN_DEPTH - TASK_SPLIT_DEPTH);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  i32 rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    std::cout << "Number of processes: " << size << std::endl;
  }

  Board board{};
  Turn turn = Turn::HUMAN;
  bool finished = false;
  i32 task_count = std::pow(COLS, TASK_SPLIT_DEPTH);

  double total_time = 0;
  while (!finished) {
    // Human move
    if (rank == 0 && turn == Turn::HUMAN) {
      board.display();
      i32 move;
      std::cout << "Choose you move (0-6): " << std::endl;
      std::cin >> move;
      for (i32 i = ROWS - 1; i >= 0; --i) {
        if (board.grid[i][move] == BoardCell::EMPTY) {
          board.grid[i][move] = BoardCell::HUMAN;
          board.display();
          if (board.find_winner(i, move) == BoardCell::HUMAN) {
            std::cout << "Victory!" << std::endl;
            finished = true;
          }
          break;
        }
      }
      turn = Turn::COMPUTER;
    }

    // Sync, this makes sure all the processes wait here for the human to make a move
    MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (finished) {
      break;
    }
    MPI_Bcast(board.grid.data(), ROWS * COLS, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Computer move
    double start_time = MPI_Wtime();

    std::vector<double> local_sums(COLS, 0.0);
    std::vector<i32> local_counts(COLS, 0);

    for (i32 t_id = rank; t_id < task_count; t_id += size) {
      auto result = execute_task_by_id(t_id, board);
      if (result.has_value()) {
        i32 first_col = t_id % COLS;  // Computers move is always the unit digit
        local_sums[first_col] += result.value();
        local_counts[first_col] += 1;
      }
    }

    std::vector<double> global_sums(COLS);
    std::vector<i32> global_counts(COLS);
    MPI_Reduce(local_sums.data(), global_sums.data(), COLS, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(local_counts.data(), global_counts.data(), COLS, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
      i32 best_move = -1;
      double best_score = -DOUBLE_INF;

      for (i32 c = 0; c < COLS; ++c) {
        if (global_counts[c] == 0) {
          continue;  // column was fully blocked
        }
        double avg = global_sums[c] / global_counts[c];
        if (avg > best_score) {
          best_score = avg;
          best_move = c;
        }
      }

      // Execute best move
      for (i32 i = ROWS - 1; i >= 0; --i) {
        if (board.grid[i][best_move] == BoardCell::EMPTY) {
          board.grid[i][best_move] = BoardCell::COMPUTER;
          board.display();
          if (board.find_winner(i, best_move) == BoardCell::COMPUTER) {
            std::cout << "Defeat!" << std::endl;
            finished = true;
          }
          break;
        }
      }
      std::cout << "Computer played: " << best_move
                << " in: " << MPI_Wtime() - start_time << "s" << std::endl;
      turn = Turn::HUMAN;
      total_time += MPI_Wtime() - start_time;
    }
    MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (finished) {
      break;
    }
  }
  MPI_Finalize();
  std::cout << "Total time taken by computer moves: " << total_time << "s" << std::endl;
  return 0;
}
