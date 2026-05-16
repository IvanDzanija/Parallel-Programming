#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "mpi.h"

using i32 = int32_t;
using u8 = uint8_t;

i32 total;

enum class State : u8 { THINKING, HUNGRY, EATING };
enum Side : u8 { LEFT = 0, RIGHT = 1 };
enum class Message : u8 { REQUEST = 0, FORK = 1 };  // We use this as tags in MPI
static constexpr u8 REQUEST = 0;
static constexpr u8 FORK = 1;

class Philosopher {
 private:
  i32 _id;
  State _state;
  std::array<i32, 2> _neighbors;  // [LEFT, RIGHT]
  std::array<bool, 2> _has_fork;
  std::array<bool, 2> _dirty;
  std::array<bool, 2> _request_pending;
  std::array<bool, 2> _request_sent;
  std::mt19937 _rng;

  i32 _rand_seconds(i32 lo = 1, i32 hi = 3) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(_rng);
  }

  std::string _indent() const { return std::string(_id, '\t'); }

  void _log(const std::string &msg) const {
    std::cout << _indent() << "[P" << _id << "]: " << msg << std::endl;
  }

  i32 _get_fork_id(Side dir) const {
    if (dir == LEFT) {
      return _id;
    } else {
      return (_id - 1 + total) % total;
    }
  }

  i32 _get_tag(Side dir, u8 msg_type) const {
    return (_get_fork_id(dir) * 2) + msg_type;
  }

  void _request_fork(Side dir) {
    i32 tag = _get_tag(dir, REQUEST);
    _log("Requesting " + std::string(dir == LEFT ? "LEFT" : "RIGHT") + " (Fork " +
         std::to_string(_get_fork_id(dir)) + ") from P" +
         std::to_string(_neighbors[dir]));

    MPI_Send(nullptr, 0, MPI_INT, _neighbors[dir], tag, MPI_COMM_WORLD);
    _request_sent[dir] = true;
  }

  void _check_for_forks() {
    for (Side dir : {LEFT, RIGHT}) {
      if (_request_sent[dir]) {
        i32 tag = _get_tag(dir, FORK);
        i32 flag;
        MPI_Iprobe(_neighbors[dir], tag, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        if (flag) {
          MPI_Recv(nullptr, 0, MPI_INT, _neighbors[dir], tag, MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
          _has_fork[dir] = true;
          _dirty[dir] = false;
          _request_sent[dir] = false;
          _log("Received " + std::string(dir == LEFT ? "LEFT" : "RIGHT") + " fork.");
        }
      }
    }
  }

  void _give_fork(Side dir) {
    _has_fork[dir] = false;
    _dirty[dir] = false;
    i32 tag = _get_tag(dir, FORK);
    MPI_Send(nullptr, 0, MPI_INT, _neighbors[dir], tag, MPI_COMM_WORLD);
  }

  void _serve_requests() {
    for (Side dir : {LEFT, RIGHT}) {
      if (_request_pending[dir] && _has_fork[dir] && _dirty[dir]) {
        _give_fork(dir);
        _request_pending[dir] = false;
      }
    }
  }

  void _check_for_requests() {
    for (Side dir : {LEFT, RIGHT}) {
      i32 tag = _get_tag(dir, REQUEST);
      i32 flag;
      MPI_Iprobe(_neighbors[dir], tag, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        MPI_Recv(nullptr, 0, MPI_INT, _neighbors[dir], tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        _request_pending[dir] = true;
      }
    }
    _serve_requests();
  }

 public:
  explicit Philosopher(i32 id) : _id(id), _state(State::THINKING) {
    _neighbors[RIGHT] = (id - 1 + total) % total;
    _neighbors[LEFT] = (id + 1) % total;
    _has_fork[LEFT] = _id < _neighbors[LEFT];
    _has_fork[RIGHT] = _id < _neighbors[RIGHT];
    // Both dirty at the start
    _dirty[RIGHT] = _has_fork[RIGHT];
    _dirty[LEFT] = _has_fork[LEFT];
    _request_pending.fill(false);
    _request_sent.fill(false);

    auto seed =
        static_cast<unsigned>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
        id;
    _rng.seed(seed);
  }

  Side opposite(Side dir) { return dir == LEFT ? RIGHT : LEFT; }

  void eat() {
    assert(_state == State::HUNGRY);
    assert(_has_fork[LEFT] && _has_fork[RIGHT]);  // Must have both forks to eat
    _state = State::EATING;
    i32 eat_time = _rand_seconds(2, 5);
    _log("Eating for " + std::to_string(eat_time) + " seconds.");
    std::this_thread::sleep_for(std::chrono::seconds(eat_time));
    _state = State::THINKING;
    for (Side dir : {LEFT, RIGHT}) {
      if (_has_fork[dir]) {
        _dirty[dir] = true;  // Mark fork as dirty after eating
      }
    }
  }

  void think() {
    _state = State::THINKING;
    i32 think_time = _rand_seconds(1, 3);
    _log("Thinking for " + std::to_string(think_time) + " seconds.");
    for (i32 time = 0; time < think_time * 1000; ++time) {
      _check_for_requests();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    _state = State::HUNGRY;
  }

  void aquire_forks() {
    assert(_state == State::HUNGRY);
    while (!_has_fork[LEFT] || !_has_fork[RIGHT]) {
      for (Side dir : {LEFT, RIGHT}) {
        if (!_has_fork[dir] && !_request_sent[dir]) {
          _request_fork(dir);
        }
      }
      _check_for_forks();
      _check_for_requests();
    }
  }
};

int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> d(10000, 500);

  i32 rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &total);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  Philosopher philosopher(rank);

  while (true) {
    philosopher.think();
    philosopher.aquire_forks();
    philosopher.eat();
  }

  MPI_Finalize();
  return 0;
}
