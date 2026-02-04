#pragma once

#include "features.h"
#include "index.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class IndexManager {
private:
  std::vector<std::unique_ptr<Feature>> features;

public:
  IndexManager() = default;

  // Add a feature (table size fixed in FTRL)
  void add_feature(std::unique_ptr<Feature> feature) {
    features.push_back(std::move(feature));
  }

  // Get total number of features/tables
  size_t get_num_features() const { return features.size(); }

  // Update all features with branch outcome
  void update_features(InstClass inst_class, bool actual_outcome,
                       const branch_info &bi) {
    for (auto &f : features) {
      f->update(inst_class, actual_outcome, bi);
    }
  }

  void get_indices(const branch_info &bi,
                   std::vector<Index> &out_indices) const {
    out_indices.clear();
    out_indices.reserve(features.size());
    for (const auto &f : features) {
      auto size = f->is_fixed_size();
      if (size.has_value()) {
        out_indices.emplace_back(f->get_index(bi), true);
      } else {
        out_indices.emplace_back(hash_64(f->get_index(bi)), false);
      }
    }
  }

  std::vector<std::optional<uint64_t>> get_sizes() const {
    std::vector<std::optional<uint64_t>> sizes;
    for (const auto &f : features) {
      sizes.push_back(f->is_fixed_size());
    }
    return sizes;
  }
};
