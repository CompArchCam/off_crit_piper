#pragma once

#include "../lib/sim_common_structs.h"
#include <cstdint>
#include <string>
#include <vector>

/**
 * Abstract base class for FTRP_P features.
 *
 * Features are components that can be trained online to predict branch
 * outcomes. Each feature maintains its own state and provides indexing into
 * prediction tables.
 */
class Feature {
protected:
  std::string name;
  bool use_pc;

public:
  virtual ~Feature() = default;

  explicit Feature(const std::vector<std::string> &args, bool _use_pc = true)
      : use_pc(_use_pc) {
    if (args.empty()) {
      // throw std::runtime_error("Feature requires at least a name.");
      // Allowed: name might be passed separately or inferred
      name = "Unknown";
    } else {
      name = args[0];
    }
  }

  /**
   * Update the feature state based on branch outcome.
   * Called when a branch is resolved to train the feature.
   *
   * @param inst_class  Instruction class of the branch
   * @param pc          Program counter of the branch
   * @param actual_outcome Actual outcome (true for taken, false for not-taken)
   */
  virtual void update(InstClass inst_class, uint64_t pc, bool actual_outcome,
                      uint64_t target) = 0;

  /**
   * Get the index into the prediction table for a given program counter.
   * This index is used to look up the prediction weight/value for this feature.
   *
   * @param pc Program counter of the branch to predict
   * @return uint64_t Index into the feature's prediction table
   */
  virtual uint64_t get_index(uint64_t pc) const = 0;

protected:
  // Derived classes can add their own state here
};
