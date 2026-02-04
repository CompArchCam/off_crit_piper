#pragma once

#include "ftrl.h"
#include <vector>

// FTRL with Hard Branch Table (HBT) gating
// FTRL with Hard Branch Table (HBT) gating
class HbtFTRL : public FTRL {
private:
  static constexpr size_t HBT_SIZE = 1024;
  static constexpr uint64_t HBT_DECAY_PERIOD = 8192;

  std::vector<uint8_t> hbt_table;
  uint64_t retired_branches;

  size_t get_hbt_index(uint64_t pc) const { return pc % HBT_SIZE; }

  bool is_hbt_active(uint64_t pc) const {
    return hbt_table[get_hbt_index(pc)] > 0;
  }

  void update_hbt(uint64_t pc, bool mispredicted) {
    size_t idx = get_hbt_index(pc);
    if (mispredicted) {
      if (hbt_table[idx] < 3) {
        hbt_table[idx]++;
      }
    }
  }

  void apply_decay() {
    for (auto &val : hbt_table) {
      if (val > 0)
        val--;
    }
  }

public:
  HbtFTRL(const std::vector<std::optional<uint64_t>> &feature_sizes,
          float _alpha, float _beta, float _l1, float _l2,
          uint64_t _hbt_decay_period_unused, uint8_t _hbt_decay_amount_unused)
      : FTRL(feature_sizes, _alpha, _beta, _l1, _l2), hbt_table(HBT_SIZE, 0),
        retired_branches(0) {}

  void update(const std::vector<Index> &indices, float pred, bool actual,
              uint64_t cycle, uint64_t pc, const TagePrediction &tage_pred,
              float tage_weight) override {

    retired_branches++;

    // HBT decay
    if (retired_branches % HBT_DECAY_PERIOD == 0) {
      apply_decay();
    }

    // 1. Update Hard Branch Table
    bool mispredicted = (tage_pred.prediction != actual);
    update_hbt(pc, mispredicted);

    // 2. FTRL Training Gate
    bool is_hbt = is_hbt_active(pc);
    bool hbt_conf = is_hbt;

    if (is_hbt) {
      float ftrl_pred = 0.0;
      for (size_t i = 0; i < indices.size(); ++i) {
        uint64_t raw_idx = indices[i].value;
        auto &table = tables[i];
        size_t idx = raw_idx % TABLE_SIZE;
        ftrl_pred +=
            compute_weight_internal(table.z_table[idx], table.n_table[idx]);
      }

      float g = ftrl_pred - (actual ? 1.0f : 0.0f);
      float g2 = g * g;

      size_t count = std::min(indices.size(), tables.size());

      for (size_t i = 0; i < count; ++i) {
        uint64_t raw_idx = indices[i].value;
        auto &table = tables[i];

        size_t idx = raw_idx % TABLE_SIZE;

        float zi = table.z_table[idx];
        float ni = table.n_table[idx];

        float weight = compute_weight_internal(zi, ni);

        float si = (std::sqrt(ni + g2) - std::sqrt(ni)) / alpha;
        table.z_table[idx] += g - si * weight;
        table.n_table[idx] += g2;
      }
    }

    // 3. Weight Synchronization to FSC via HBT check
    size_t count = std::min(indices.size(), tables.size());
    for (size_t i = 0; i < count; ++i) {
      float weight_to_send = 0.0f;

      if (hbt_conf) {
        const auto &table = tables[i];
        size_t idx = indices[i].value % TABLE_SIZE;
        weight_to_send =
            compute_weight_internal(table.z_table[idx], table.n_table[idx]);
      }

      nonzero_weights.push_back({indices[i], weight_to_send, i, cycle});
    }
  }

  size_t size_bits() const { return FTRL::size_bytes() * 8 + HBT_SIZE * 2; }
};
