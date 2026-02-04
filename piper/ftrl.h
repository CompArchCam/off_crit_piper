#pragma once

#include "../tage/tage_prediction.h"
#include "index.h"
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <vector>

struct ActiveWeight {
  Index index;
  float weight;
  size_t feature_idx;
  uint64_t cycle;
};

// Base FTRL Class
class FTRL {
protected:
  struct TableState {
    std::vector<float> z_table;
    std::vector<float> n_table;

    TableState(size_t size) : z_table(size, 0.0f), n_table(size, 0.0f) {}
  };

  std::vector<TableState> tables;
  std::deque<ActiveWeight> nonzero_weights;

  static constexpr size_t TABLE_SIZE{1 << 14};

  float alpha;
  float beta;
  float l1;
  float l2;

  float compute_weight_internal(float z, float n) const {
    if (std::abs(z) <= l1)
      return 0.0f;
    return -(z - (z > 0 ? 1.0f : -1.0f) * l1) /
           (((beta + std::sqrt(n)) / alpha) + l2);
  }

  // Helper methods
  uint64_t hash_index(uint64_t pc) const { return pc & (TABLE_SIZE - 1); }

public:
  FTRL(const std::vector<std::optional<uint64_t>> &feature_sizes, float _alpha,
       float _beta, float _l1, float _l2)
      : alpha(_alpha), beta(_beta), l1(_l1), l2(_l2) {
    if (alpha <= 0.0f)
      throw std::runtime_error("FTRL alpha must be > 0");

    tables.reserve(feature_sizes.size());

    for (const auto &size : feature_sizes) {
      tables.emplace_back(size.has_value() ? size.value() : TABLE_SIZE);
    }
  }

  virtual ~FTRL() = default;

  virtual void update(const std::vector<Index> &indices, float pred,
                      bool actual, uint64_t cycle, uint64_t pc,
                      const TagePrediction &tage_pred, float tage_weight) = 0;

  const std::deque<ActiveWeight> &get_nonzero_weights() const {
    return nonzero_weights;
  }

  // Pop one weight from the queue (for FSC sync)
  bool pop_active_weight(ActiveWeight &out) {
    if (nonzero_weights.empty())
      return false;
    out = nonzero_weights.front();
    nonzero_weights.pop_front();
    return true;
  }

  void pop_active_weights_older_than(uint64_t cycle_threshold,
                                     std::vector<ActiveWeight> &out) {
    while (!nonzero_weights.empty()) {
      const auto &front = nonzero_weights.front();
      if (front.cycle <= cycle_threshold) {
        out.push_back(front);
        nonzero_weights.pop_front();
      } else {
        // Since weights are pushed in chronological order, we can stop early
        break;
      }
    }
  }

  void clear_nonzero_queue() { nonzero_weights.clear(); }

  virtual size_t size_bytes() const {
    size_t total = 0;
    for (const auto &t : tables) {
      total += t.z_table.size() * sizeof(float);
      total += t.n_table.size() * sizeof(float);
    }
    return total;
  }
};
