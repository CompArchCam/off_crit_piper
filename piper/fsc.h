#pragma once

#include "utils/hash.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

class FSC {
public:
  static constexpr int WEIGHT_BITS = 8;
  static constexpr float WEIGHT_SCALE = static_cast<float>(1 << WEIGHT_BITS);

private:
  struct FscEntry {
    uint16_t tag = 0;
    int16_t weight = 0;
    // Could add valid bit, but weight 0.0f or initial tag could serve?
    // Better to have explicit empty state or initialization.
    // Let's use weight 0.0f as implicit empty or just overwrite.
    // User said "If there are no non zero weights in the way", implying nonzero
    // weight means occupied.
  };

  struct SkewEntry {
    uint8_t tage_win = 0;
    uint8_t ftrl_win = 0;
  };

  static constexpr size_t NUM_WAYS = 4;
  static constexpr size_t NUM_SETS = 128; // 128 entries per way = 512 total

  // Each feature has its own table
  std::vector<std::vector<FscEntry>> tables;
  std::vector<SkewEntry> skew_table;
  int tag_size;

public:
  enum class AllocationPolicy { ALWAYS_REPLACE, CHECK_MAGNITUDE };

private:
  AllocationPolicy alloc_policy;

  static float fixed_to_float(int16_t w) {
    return static_cast<float>(w) / WEIGHT_SCALE;
  }

  static int16_t float_to_fixed(float f) {
    return static_cast<int16_t>(std::round(f * WEIGHT_SCALE));
  }

  // Helper: Compute tag from raw index
  uint16_t compute_tag(uint64_t index) const {
    // Simple tag derivation: mix index or just shift
    // Since set index uses low bits (index % 128), tag should use higher bits
    // Let's just shift right by 7 (since 128 = 2^7)
    uint64_t shifted = index >> 7;
    uint64_t mask = (1ULL << tag_size) - 1;
    return static_cast<uint16_t>(shifted & mask);
  }

  // Helper: Get set index
  size_t get_set(uint64_t index) const { return index % NUM_SETS; }

public:
  FSC(size_t num_features, int _tag_size, AllocationPolicy policy)
      : tag_size(_tag_size), alloc_policy(policy) {
    tables.resize(num_features);
    for (auto &table : tables) {
      table.resize(NUM_SETS * NUM_WAYS);
    }
    // 4096 entries for skew table
    skew_table.resize(4096);
  }

  // Get skew confidence
  // Returns true if ftrl_win is saturated (3) and tage_win is <= 1
  bool get_conf(uint64_t pc) const {
    uint64_t idx = hash_64(pc) & 4095; // Simple hash
    // const auto &entry = skew_table[idx]; // Unused variable
    return true;
  }

  void update_skew(uint64_t pc, bool tage_correct, bool fsc_correct) {
    uint64_t idx = hash_64(pc) & 4095;
    auto &entry = skew_table[idx];

    // Saturating counters, 2-bit (max 3)
    if (tage_correct && !fsc_correct) {
      if (entry.tage_win < 3) {
        entry.tage_win++;
      } else {
        if (entry.ftrl_win > 0)
          entry.ftrl_win--;
      }
    } else if (!tage_correct && fsc_correct) {
      if (entry.ftrl_win < 3) {
        entry.ftrl_win++;
      } else {
        if (entry.tage_win > 0)
          entry.tage_win--;
      }
    }
  }

  // Get prediction probability by summing weights of hits
  float get_prediction(const std::vector<uint64_t> &indices) const {
    float sum = 0.0f;
    size_t count = std::min(indices.size(), tables.size());

    for (size_t i = 0; i < count; ++i) {
      uint64_t index = indices[i];
      size_t set = get_set(index);
      uint16_t tag = compute_tag(index);

      const auto &table = tables[i];
      size_t base = set * NUM_WAYS;

      for (size_t w = 0; w < NUM_WAYS; ++w) {
        const auto &entry = table[base + w];
        if (entry.weight != 0 && entry.tag == tag) {
          sum += fixed_to_float(entry.weight);
          break; // Assumes only one match per set (enforced by allocation)
        }
      }
    }

    return sum;
  }

  // Allocate/Update entry
  void allocate(uint64_t index, size_t feature_idx, float weight) {
    size_t set = get_set(index);
    uint16_t tag = compute_tag(index);

    auto &table = tables[feature_idx];
    size_t base = set * NUM_WAYS;

    // 1. Check for existing match to update
    for (size_t w = 0; w < NUM_WAYS; ++w) {
      if (table[base + w].tag == tag) {
        table[base + w].weight = float_to_fixed(weight);
        return;
      }
    }

    if (weight == 0.0f)
      return; // Don't allocate zero weights

    // 2. Find empty slot
    int victim_way = -1;
    for (size_t w = 0; w < NUM_WAYS; ++w) {
      if (table[base + w].weight == 0) {
        victim_way = w;
        break;
      }
    }

    // 3. If no empty slot, find smallest absolute weight
    if (victim_way == -1) {
      int16_t min_abs_weight = std::numeric_limits<int16_t>::max();
      for (size_t w = 0; w < NUM_WAYS; ++w) {
        int16_t abs_w = std::abs(table[base + w].weight);
        if (abs_w < min_abs_weight) {
          min_abs_weight = abs_w;
          victim_way = w;
        }
      }

      // Check policy if we are replacing an occupied slot
      if (alloc_policy == AllocationPolicy::CHECK_MAGNITUDE) {
        int16_t incoming_mag = std::abs(float_to_fixed(weight));
        if (incoming_mag <= min_abs_weight) {
          return; // Don't allocate if new weight is not bigger than smallest
                  // existing
        }
      }
    }

    // 4. Update victim
    if (victim_way != -1) {
      table[base + victim_way].tag = tag;
      table[base + victim_way].weight = float_to_fixed(weight);
    }
  }
  // Zero weights that disagreed with the actual outcome
  void update_weights_fast(const std::vector<uint64_t> &indices, bool taken) {
    size_t count = std::min(indices.size(), tables.size());

    for (size_t i = 0; i < count; ++i) {
      uint64_t index = indices[i];
      size_t set = get_set(index);
      uint16_t tag = compute_tag(index);

      auto &table = tables[i];
      size_t base = set * NUM_WAYS;

      for (size_t w = 0; w < NUM_WAYS; ++w) {
        if (table[base + w].tag == tag) {
          int16_t weight = table[base + w].weight;
          // If taken (true), we want positive weights. If weight < 0, zero it.
          // If not taken (false), we want negative weights. If weight > 0, zero
          // it.
          if ((taken && weight < 0) || (!taken && weight > 0)) {
            table[base + w].weight = 0;
          }
          break;
        }
      }
    }
  }

  // Get total size in bits
  size_t get_size() const {
    size_t total_bits = 0;
    size_t entries_per_table = NUM_SETS * NUM_WAYS;
    for (size_t i = 0; i < tables.size(); ++i) {
      // Each entry: tag_size + 16 bits for weight
      total_bits += entries_per_table * (tag_size + 16);
    }
    return total_bits;
  }
};
