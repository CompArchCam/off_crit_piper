#pragma once

#include "features.h"
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
  void update_features(InstClass inst_class, uint64_t pc, bool actual_outcome,
                       uint64_t target) {
    for (auto &f : features) {
      f->update(inst_class, pc, actual_outcome, target);
    }
  }

  // Generate indices for all features for a given PC
  void get_indices(uint64_t pc, std::vector<uint64_t> &out_indices) const {
    out_indices.clear();
    out_indices.reserve(features.size());
    for (const auto &f : features) {
      out_indices.push_back(hash_64(f->get_index(pc)));
    }
  }
};
