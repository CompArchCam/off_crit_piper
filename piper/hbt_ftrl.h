#pragma once

#include "ftrl.h"
#include <vector>

// FTRL with Hard Branch Table (HBT) gating
class HbtFTRL : public FTRL {
private:
  struct HBTEntry {
    uint8_t misp_counter; // 5-bit saturating counter (0-31)
    uint64_t tag;         // Partial PC tag for identification
    bool valid;           // Entry validity flag

    HBTEntry() : misp_counter(0), tag(0), valid(false) {}
  };

  std::vector<HBTEntry> hbt;
  uint64_t hbt_decay_period;
  uint8_t hbt_decay_amount;
  uint64_t retired_branches;

  uint64_t compute_tag(uint64_t pc) const {
    // Use upper bits of PC as tag
    return (pc >> 14) & 0xFFFF;
  }

  uint8_t get_hbt_misp_counter(uint64_t pc) const {
    size_t idx = hash_index(pc);
    const auto &entry = hbt[idx];
    return entry.misp_counter;
  }

  bool is_hbt_active(uint64_t pc) const { return get_hbt_misp_counter(pc) > 0; }

  void update_hbt(uint64_t pc, bool mispredicted, bool hit_in_last) {
    size_t idx = hash_index(pc);
    auto &entry = hbt[idx];
    uint64_t tag = compute_tag(pc);

    if (mispredicted) {
      // Allocate new entry or update existing
      if (entry.tag != tag) {
        // Only allocate if counter is 0 (allows overwriting old entries)
        // Only allocate/update if branch hit in last history table
        if (entry.misp_counter == 0 && hit_in_last) {
          entry.valid = true;
          entry.tag = tag;
          entry.misp_counter = 1;
        }
      } else {
        // Saturate at 31
        if (entry.misp_counter < 31)
          entry.misp_counter++;
      }
    }
  }

  void apply_leaky_bucket_hbt() {
    for (auto &entry : hbt) {
      if (entry.valid && entry.misp_counter > 0) {
        uint8_t decrement = std::min(hbt_decay_amount, entry.misp_counter);
        entry.misp_counter -= decrement;
        // Invalidate entry if counter reaches 0
        if (entry.misp_counter == 0) {
          entry.valid = false;
        }
      }
    }
  }

public:
  HbtFTRL(size_t num_features, float _alpha, float _beta, float _l1, float _l2,
          uint64_t _hbt_decay_period, uint8_t _hbt_decay_amount)
      : FTRL(num_features, _alpha, _beta, _l1, _l2),
        hbt_decay_period(_hbt_decay_period),
        hbt_decay_amount(_hbt_decay_amount), hbt(TABLE_SIZE),
        retired_branches(0) {}

  void update(const std::vector<uint64_t> &indices, float pred, bool actual,
              uint64_t cycle, uint64_t pc,
              const TagePrediction &tage_pred) override {

    retired_branches++;

    // HBT decay
    if (retired_branches % hbt_decay_period == 0) {
      apply_leaky_bucket_hbt();
    }

    // 1. Update Hard Branch Table
    bool mispredicted = (tage_pred.prediction != actual);
    update_hbt(pc, mispredicted, tage_pred.hit_in_last_history_table);

    // 2. FTRL Training Gate - now purely based on HBT being active
    bool is_hbt = is_hbt_active(pc);
    bool hbt_conf = is_hbt;

    if (is_hbt) {
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
    }

    // 3. Weight Synchronization to FSC via HBT check
    size_t count = std::min(indices.size(), tables.size());
    for (size_t i = 0; i < count; ++i) {
      uint64_t raw_idx = indices[i];
      float weight_to_send = 0.0f;

      if (hbt_conf) {
        const auto &table = tables[i];
        size_t idx = raw_idx & 0x3FFF;
        weight_to_send =
            compute_weight_internal(table.z_table[idx], table.n_table[idx]);
      }

      // Push to queue
      nonzero_weights.push_back({raw_idx, weight_to_send, i, cycle});
    }
  }

  size_t size_bytes() const override {
    return FTRL::size_bytes() + hbt.size() * sizeof(HBTEntry);
  }
};
