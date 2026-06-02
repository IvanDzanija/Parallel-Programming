#include <mpi.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

using i32 = int32_t;
using i8 = int8_t;
using u8 = uint8_t;

// Constants
static constexpr double DOUBLE_INF = std::numeric_limits<double>::infinity();
static constexpr size_t ROWS = 6;
static constexpr size_t COLS = 7;
static constexpr size_t MIN_DEPTH = 11;
static constexpr size_t TASK_SPLIT_DEPTH = 3;  // 7^3 = 343 tasks

// MPI message tags
static constexpr i32 TAG_TASK = 1;
static constexpr i32 TAG_RESULT = 2;
static constexpr i32 STOP_TASK = -1;

// Game state enums
enum class Turn : u8 { HUMAN = 1, COMPUTER = 2 };
Turn swap(Turn t) { return t == Turn::HUMAN ? Turn::COMPUTER : Turn::HUMAN; }
enum class BoardCell : u8 { EMPTY = 0, HUMAN = 1, COMPUTER = 2 };
BoardCell place_turn(Turn t) {
  return t == Turn::HUMAN ? BoardCell::HUMAN : BoardCell::COMPUTER;
}

struct Board {
  std::array<std::array<BoardCell, COLS>, ROWS> grid;

  void display() const {
    for (const auto &row : grid) {
      for (const auto &cell : row) {
        char c = (cell == BoardCell::HUMAN)      ? 'X'
                 : (cell == BoardCell::COMPUTER) ? 'O'
                                                 : '.';
        std::cout << c << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "0 1 2 3 4 5 6" << std::endl;
    std::cout << std::endl;
  }

  BoardCell find_winner(i32 r, i32 c) const {
    static constexpr std::array<std::pair<i8, i8>, 4> DIRECTIONS = {
        {{0, 1}, {1, 0}, {1, 1}, {1, -1}}};
    BoardCell player = grid[r][c];
    if (player == BoardCell::EMPTY) {
      return BoardCell::EMPTY;
    }

    for (auto [dr, dc] : DIRECTIONS) {
      i32 count = 1;
      for (i32 i = 1; i < 4; ++i) {
        i32 rr = r + dr * i, cc = c + dc * i;
        if (rr >= 0 && rr < ROWS && cc >= 0 && cc < COLS && grid[rr][cc] == player) {
          ++count;
        } else {
          break;
        }
      }
      for (i32 i = 1; i < 4; ++i) {
        i32 rr = r - dr * i, cc = c - dc * i;
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
      return -1.0;
    }
    if (winner == BoardCell::COMPUTER) {
      return 1.0;
    }
    if (depth == 0) {
      return 0.0;
    }

    double sum = 0.0;
    i32 moves = 0;
    for (i32 c = 0; c < COLS; ++c) {
      i32 r = ROWS - 1;
      for (; r >= 0 && grid[r][c] != BoardCell::EMPTY; --r);
      if (r == -1) {
        continue;
      }

      ++moves;
      grid[r][c] = place_turn(turn);
      double res = evaluate(swap(turn), r, c, depth - 1);
      grid[r][c] = BoardCell::EMPTY;

      if (turn == Turn::COMPUTER && res == 1.0) {
        return 1.0;
      }
      if (turn == Turn::HUMAN && res == -1.0) {
        return -1.0;
      }
      sum += res;
    }
    return moves == 0 ? 0.0 : sum / moves;
  }
};

std::optional<double> execute_task_by_id(i32 t_id, Board board) {
  i32 temp = t_id;
  Turn curr = Turn::COMPUTER;
  i32 last_r = -1, last_c = -1;

  std::array<i32, TASK_SPLIT_DEPTH> moves{};
  for (size_t d = 0; d < TASK_SPLIT_DEPTH; ++d) {
    moves[d] = temp % COLS;
    temp /= COLS;
  }

  for (size_t d = 0; d < TASK_SPLIT_DEPTH; ++d) {
    i32 c = moves[d];
    i32 r = ROWS - 1;
    for (; r >= 0 && board.grid[r][c] != BoardCell::EMPTY; --r);
    if (r == -1) {
      return std::nullopt;
    }

    board.grid[r][c] = place_turn(curr);
    last_r = r;
    last_c = c;

    BoardCell winner = board.find_winner(r, c);
    if (winner == BoardCell::COMPUTER) {
      return 1.0;
    }
    if (winner == BoardCell::HUMAN) {
      return -1.0;
    }
    curr = swap(curr);
  }

  return board.evaluate(curr, last_r, last_c, MIN_DEPTH - TASK_SPLIT_DEPTH);
}

struct ResultMsg {
  std::array<double, 3> buff;
  static ResultMsg pack(i32 first_column, std::optional<double> value) {
    return {(double)first_column, value.has_value() ? 1.0 : 0.0,
            value.has_value() ? value.value() : 0.0};
  }

  [[nodiscard]] i32 first_column() const { return (i32)buff[0]; }
  bool has_value() const { return buff[1] != 0.0; }
  double value() const { return buff[2]; }
};

// Master work
void run_master(i32 num_workers, Board &board) {
  Turn turn = Turn::HUMAN;
  bool finished = false;
  i32 task_count = (i32)std::pow((double)COLS, (double)TASK_SPLIT_DEPTH);
  double total_time = 0.0;

  while (!finished) {
    // Human move
    if (turn == Turn::HUMAN) {
      board.display();
      i32 move;
      std::cout << "Choose your move (0-6):\n";
      std::cin >> move;

      for (i32 i = ROWS - 1; i >= 0; --i) {
        if (board.grid[i][move] == BoardCell::EMPTY) {
          board.grid[i][move] = BoardCell::HUMAN;
          board.display();
          if (board.find_winner(i, move) == BoardCell::HUMAN) {
            std::cout << "Victory!\n";
            finished = true;
          }
          break;
        }
      }
      turn = Turn::COMPUTER;
    }

    MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (finished) {
      break;
    }
    MPI_Bcast(board.grid.data(), ROWS * COLS, MPI_BYTE, 0, MPI_COMM_WORLD);

    // Computer move
    double start_time = MPI_Wtime();

    std::vector<double> global_sums(COLS, 0.0);
    std::vector<i32> global_counts(COLS, 0);

    i32 next_task = 0;
    i32 active = 0;

    // Distribute the initial batch of tasks
    for (i32 w = 1; w <= num_workers; ++w) {
      if (next_task < task_count) {
        MPI_Send(&next_task, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
        ++next_task;
        ++active;
      } else {
        MPI_Send(&STOP_TASK, 1, MPI_INT, w, TAG_TASK, MPI_COMM_WORLD);
      }
    }

    // Collect results and give next available task
    while (active > 0) {
      ResultMsg msg;
      MPI_Status status;
      MPI_Recv(msg.buff.data(), 3, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_RESULT,
               MPI_COMM_WORLD, &status);

      if (msg.has_value()) {
        global_sums[msg.first_column()] += msg.value();
        global_counts[msg.first_column()] += 1;
      }

      if (next_task < task_count) {
        MPI_Send(&next_task, 1, MPI_INT, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
        ++next_task;
      } else {
        MPI_Send(&STOP_TASK, 1, MPI_INT, status.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
        --active;
      }
    }

    // Pick best column
    i32 best_move = -1;
    double best_score = -DOUBLE_INF;

    for (i32 c = 0; c < (i32)COLS; ++c) {
      if (global_counts[c] == 0) {
        continue;  // column was fully blocked
      }
      double avg = global_sums[c] / global_counts[c];
      if (avg > best_score) {
        best_score = avg;
        best_move = c;
      }
    }

    // Execute the best move
    for (i32 i = ROWS - 1; i >= 0; --i) {
      if (board.grid[i][best_move] == BoardCell::EMPTY) {
        board.grid[i][best_move] = BoardCell::COMPUTER;
        board.display();
        if (board.find_winner(i, best_move) == BoardCell::COMPUTER) {
          std::cout << "Defeat!\n";
          finished = true;
        }
        break;
      }
    }
    double elapsed = MPI_Wtime() - start_time;
    total_time += elapsed;
    std::cout << "Computer played: " << best_move << " in: " << elapsed << "s\n";
    turn = Turn::HUMAN;

    MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    if (finished) {
      break;
    }
  }

  std::cout << "Total computer time: " << total_time << "s\n";
}

// Worker work
void run_worker(Board &board) {
  bool finished = false;

  while (true) {
    MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);  // Human won here
    if (finished) {
      break;
    }
    MPI_Bcast(board.grid.data(), ROWS * COLS, MPI_BYTE, 0, MPI_COMM_WORLD);

    while (true) {
      i32 t_id;
      MPI_Recv(&t_id, 1, MPI_INT, 0, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (t_id == STOP_TASK) {
        break;
      }

      auto result = execute_task_by_id(t_id, board);
      ResultMsg msg = ResultMsg::pack(t_id % COLS, result);
      MPI_Send(msg.buff.data(), 3, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }

    MPI_Bcast(&finished, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);  // Computer won here
    if (finished) {
      break;
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  i32 rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    if (rank == 0) {
      std::cerr << "ERROR: need at least 2 processes." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    std::cout << "Running with 1 master and " << (size - 1) << " workers." << std::endl;
  }

  Board board{};

  if (rank == 0) {
    run_master(size - 1, board);
  } else {
    run_worker(board);
  }

  MPI_Finalize();
  return 0;
}
