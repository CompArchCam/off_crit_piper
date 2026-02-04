#include "fixed.h"
#include "index.h"
#include "utils/hash.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

class FSC {
public:
  static constexpr int TOTAL_WEIGHT_BITS = 10;
  static constexpr int WEIGHT_DECIMAL_BITS = 5;
  static constexpr int FEATURES_PER_TABLE = 2;

private:
  struct FscEntry {
    uint16_t tag = 0;
    FixedFloat<true> weight{TOTAL_WEIGHT_BITS, WEIGHT_DECIMAL_BITS};

    FscEntry() = default;
  };

  static constexpr size_t NUM_WAYS = 4;
  static constexpr size_t NUM_SETS = 128;

  std::vector<std::vector<FscEntry>> tagged_tables;
  std::vector<std::vector<FixedFloat<true>>> fixed_tables;
  std::vector<uint64_t> fixed_table_sizes;
  size_t num_fixed_features;
  size_t num_variable_features;

  int tag_size;
  size_t num_logical_features;

public:
  enum class AllocationPolicy { ALWAYS_REPLACE, CHECK_MAGNITUDE };

private:
  AllocationPolicy alloc_policy;

  uint16_t compute_tag(uint64_t index) const {
    uint64_t shifted = index >> 7;
    uint64_t mask = (1ULL << tag_size) - 1;
    return static_cast<uint16_t>(shifted & mask);
  }

  size_t get_set(uint64_t index) const { return index % NUM_SETS; }

public:
  FSC(const std::vector<std::optional<uint64_t>> &feature_sizes, int _tag_size,
      AllocationPolicy policy)
      : tag_size(_tag_size), alloc_policy(policy) {

    for (const auto &size : feature_sizes) {
      if (size.has_value()) {
        fixed_table_sizes.push_back(size.value());
        fixed_tables.push_back(std::vector<FixedFloat<true>>(
            size.value(),
            FixedFloat<true>{TOTAL_WEIGHT_BITS, WEIGHT_DECIMAL_BITS}));
      }
    }

    num_fixed_features = fixed_table_sizes.size();
    num_variable_features = feature_sizes.size() - num_fixed_features;
    num_logical_features = feature_sizes.size();

    if (num_variable_features > 0) {
      size_t num_tables =
          (num_variable_features + FEATURES_PER_TABLE - 1) / FEATURES_PER_TABLE;
      tagged_tables.resize(num_tables);
      for (auto &table : tagged_tables) {
        table.resize(NUM_SETS * NUM_WAYS);
      }
    }
  }

  bool get_conf(uint64_t pc) const { return true; }

  float get_prediction(const std::vector<Index> &indices) const {
    assert(indices.size() == num_logical_features);

    size_t fixed_count = 0;
    size_t variable_count = 0;

    for (const auto &idx : indices) {
      if (idx.is_fixed_size)
        fixed_count++;
      else
        variable_count++;
    }

    assert(fixed_count == num_fixed_features);
    assert(variable_count == num_variable_features);

    float sum = 0.0f;
    size_t fixed_idx = 0;
    size_t variable_idx = 0;

    for (size_t i = 0; i < indices.size(); ++i) {
      const auto &index = indices[i];

      if (index.is_fixed_size) {
        assert(fixed_idx < num_fixed_features);
        size_t table_idx = index.value % fixed_table_sizes[fixed_idx];
        sum += fixed_tables[fixed_idx][table_idx].to_float();
        fixed_idx++;
      } else {
        uint64_t val = index.value;
        size_t set = get_set(val);
        uint16_t tag = compute_tag(val);

        const auto &table = tagged_tables[variable_idx / FEATURES_PER_TABLE];
        size_t base = set * NUM_WAYS;

        for (size_t w = 0; w < NUM_WAYS; ++w) {
          const auto &entry = table[base + w];
          if (entry.weight.raw != 0 && entry.tag == tag) {
            sum += entry.weight.to_float();
            break;
          }
        }
        variable_idx++;
      }
    }

    return sum;
  }

  void allocate(const Index &index, size_t feature_idx, float weight) {
    if (feature_idx < num_fixed_features) {
      size_t table_idx = index.value % fixed_table_sizes[feature_idx];
      fixed_tables[feature_idx][table_idx] = FixedFloat<true>::from_float(
          weight, TOTAL_WEIGHT_BITS, WEIGHT_DECIMAL_BITS);
    } else {
      size_t var_idx = feature_idx - num_fixed_features;
      uint64_t val = index.value;
      size_t set = get_set(val);
      uint16_t tag = compute_tag(val);

      auto &table = tagged_tables[var_idx / FEATURES_PER_TABLE];
      size_t base = set * NUM_WAYS;

      for (size_t w = 0; w < NUM_WAYS; ++w) {
        if (table[base + w].tag == tag) {
          table[base + w].weight = FixedFloat<true>::from_float(
              weight, TOTAL_WEIGHT_BITS, WEIGHT_DECIMAL_BITS);
          return;
        }
      }

      if (weight == 0.0f)
        return;

      int victim_way = -1;
      for (size_t w = 0; w < NUM_WAYS; ++w) {
        if (table[base + w].weight.raw == 0) {
          victim_way = w;
          break;
        }
      }

      if (victim_way == -1) {
        float min_abs_weight = std::numeric_limits<float>::max();
        for (size_t w = 0; w < NUM_WAYS; ++w) {
          float abs_w = std::abs(table[base + w].weight.to_float());
          if (abs_w < min_abs_weight) {
            min_abs_weight = abs_w;
            victim_way = w;
          }
        }

        if (alloc_policy == AllocationPolicy::CHECK_MAGNITUDE) {
          float incoming_mag = std::abs(weight);
          if (incoming_mag <= min_abs_weight) {
            return;
          }
        }
      }

      if (victim_way != -1) {
        table[base + victim_way].tag = tag;
        table[base + victim_way].weight = FixedFloat<true>::from_float(
            weight, TOTAL_WEIGHT_BITS, WEIGHT_DECIMAL_BITS);
      }
    }
  }

  void update_weights_fast(const std::vector<Index> &indices, bool taken) {
    assert(indices.size() == num_logical_features);

    size_t fixed_idx = 0;
    size_t variable_idx = 0;

    for (size_t i = 0; i < indices.size(); ++i) {
      const auto &index = indices[i];

      if (index.is_fixed_size) {
        assert(fixed_idx < num_fixed_features);
        size_t table_idx = index.value % fixed_table_sizes[fixed_idx];
        float weight_val = fixed_tables[fixed_idx][table_idx].to_float();

        if ((taken && weight_val < 0) || (!taken && weight_val > 0)) {
          fixed_tables[fixed_idx][table_idx].reset();
        }
        fixed_idx++;
      } else {
        uint64_t val = index.value;
        size_t set = get_set(val);
        uint16_t tag = compute_tag(val);

        auto &table = tagged_tables[variable_idx / FEATURES_PER_TABLE];
        size_t base = set * NUM_WAYS;

        for (size_t w = 0; w < NUM_WAYS; ++w) {
          if (table[base + w].tag == tag) {
            float weight_val = table[base + w].weight.to_float();
            if ((taken && weight_val < 0) || (!taken && weight_val > 0)) {
              table[base + w].weight.reset();
            }
            break;
          }
        }
        variable_idx++;
      }
    }
  }

  size_t get_size() const {
    size_t total_bits = 0;

    for (size_t i = 0; i < fixed_tables.size(); ++i) {
      total_bits += fixed_tables[i].size() * TOTAL_WEIGHT_BITS;
    }

    size_t entries_per_table = NUM_SETS * NUM_WAYS;
    for (size_t i = 0; i < tagged_tables.size(); ++i) {
      total_bits += entries_per_table * (tag_size + TOTAL_WEIGHT_BITS);
    }

    return total_bits;
  }
};
