#pragma once

#include "ftrl.h"

// High Branch Pressure FTRL (Always updates)
class HighBranchFTRL : public FTRL {
public:
  HighBranchFTRL(size_t num_features, float _alpha, float _beta, float _l1,
                 float _l2)
      : FTRL(num_features, _alpha, _beta, _l1, _l2) {}

  void update(const std::vector<uint64_t> &indices, float pred, bool actual,
              uint64_t cycle, uint64_t pc,
              const TagePrediction &tage_pred) override {
    (void)pc;
    (void)tage_pred;

    // Always train
    float g = pred - (actual ? 1.0f : 0.0f);
    float g2 = g * g;

    size_t count = std::min(indices.size(), tables.size());

    for (size_t i = 0; i < count; ++i) {
      uint64_t raw_idx = indices[i];
      auto &table = tables[i];

      size_t idx = raw_idx & 0x3FFF;

      float zi = table.z_table[idx];
      float ni = table.n_table[idx];

      float weight = compute_weight_internal(zi, ni);

      float si = (std::sqrt(ni + g2) - std::sqrt(ni)) / alpha;
      table.z_table[idx] += g - si * weight;
      table.n_table[idx] += g2;
    }

    // Always send weights
    for (size_t i = 0; i < count; ++i) {
      uint64_t raw_idx = indices[i];
      const auto &table = tables[i];
      size_t idx = raw_idx & 0x3FFF;
      float weight_to_send =
          compute_weight_internal(table.z_table[idx], table.n_table[idx]);

      // Push to queue
      nonzero_weights.push_back({raw_idx, weight_to_send, i, cycle});
    }
  }
};
