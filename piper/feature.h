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
public:
  /**
   * Constructor that initializes the feature with command-line style arguments.
   * This allows for flexible configuration of features from parsed input files.
   *
   * @param args Vector of string arguments for feature initialization
   *             (similar to main method args)
   */
  explicit Feature(const std::vector<std::string> &args) {
    // Base class can store args or leave implementation to derived classes
    (void)args; // Suppress unused parameter warning in base class
  }

  /**
   * Virtual destructor to ensure proper cleanup of derived classes
   */
  virtual ~Feature() = default;

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
