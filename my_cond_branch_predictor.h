#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>

#include "piper/feature.h"
#include "piper/features.h"
#include "piper/fsc.h"
#include "piper/ftrl.h"
#include "piper/hb_ftrl.h"
#include "piper/hbt_ftrl.h"
#include "piper/index_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct PredState {
  std::vector<uint64_t> hbt_indices;
  std::vector<uint64_t> hb_indices;
  float sum;
};

class SampleCondPredictor {
  // Components
  IndexManager hbt_index_manager;
  IndexManager hb_index_manager;

  std::unique_ptr<HbtFTRL> hbt_ftrl;
  std::unique_ptr<HighBranchFTRL> hb_ftrl;

  std::unique_ptr<FSC> hbt_fsc;
  std::unique_ptr<FSC> hb_fsc;

  // State tracking
  struct HistoryEntry {
    PredState state;
    TagePrediction tage_pred;
    bool combined_pred;
    float hbt_fsc_sum;
    float hb_fsc_sum;
  };
  std::unordered_map<uint64_t, HistoryEntry> pred_time_histories;

  // --- periodic mispred reporting (main tage vs mini tage) ---
  uint64_t br_count = 0;
  uint64_t main_tage_misp = 0;

  // Helper to create features based on type and args
  std::unique_ptr<Feature> create_feature(const std::string &type,
                                          const std::vector<std::string> &args,
                                          bool use_pc) {
    if (type == "PC") {
      return std::make_unique<PCFeature>(args, use_pc);
    } else if (type == "GHist") {
      return std::make_unique<GHistFeature>(args, use_pc);
    } else if (type == "GPath") {
      return std::make_unique<GPathFeature>(args, use_pc);
    } else if (type == "LHist") {
      return std::make_unique<LHistFeature>(args, use_pc);
    } else if (type == "Recency") {
      return std::make_unique<RecencyFeature>(args, use_pc);
    } else if (type == "RecencyPos") {
      return std::make_unique<RecencyPosFeature>(args, use_pc);
    } else if (type == "IMLI") {
      return std::make_unique<IMLIFeature>(args, use_pc);
    } else if (type == "BlurryPath") {
      return std::make_unique<BlurryPathFeature>(args, use_pc);
    } else if (type == "ReturnStackHist") {
      return std::make_unique<ReturnStackHistFeature>(args, use_pc);
    } else if (type == "BrIMLI") {
      return std::unique_ptr<BrIMLIFeature>(new BrIMLIFeature(args, use_pc));
    } else if (type == "TaIMLI") {
      return std::unique_ptr<TaIMLIFeature>(new TaIMLIFeature(args, use_pc));
    } else if (type == "PHist") {
      return std::make_unique<PHistFeature>(args, use_pc);
    } else if (type == "FHist") {
      return std::make_unique<FHistFeature>(args, use_pc);
    } else if (type == "BHist") {
      return std::make_unique<BHistFeature>(args, use_pc);
    } else if (type == "SLHist") {
      return std::make_unique<SLHistFeature>(args, use_pc);
    } else if (type == "TLHist") {
      return std::make_unique<TLHistFeature>(args, use_pc);
    } else if (type == "QLHist") {
      return std::make_unique<QLHistFeature>(args, use_pc);
    } else {
      std::cerr << "Unknown feature type: " << type << std::endl;
      exit(1);
    }
  }

public:
  SampleCondPredictor(void) {}

  void setup(const std::string &config_path) {
    float ftrl_alpha = 0.4f;
    float ftrl_beta = 0.8f;
    float ftrl_l1 = 0.2f;
    float ftrl_l2 = 0.6f;

    uint64_t hbt_decay_period = 10000;
    uint8_t hbt_decay_amount = 1;

    // Config for High Branch FTRL (default same as HBT or independent?)
    // User didn't specify separate params, so reuse for now or look for
    // separate Use same for now.

    bool config_loaded = false;

    enum class ParseState { NONE, HBT, MANY_BRANCHES };
    ParseState state = ParseState::NONE;

    if (!config_path.empty()) {
      std::ifstream infile(config_path);
      if (infile.is_open()) {
        std::string line;
        while (std::getline(infile, line)) {
          if (line.empty() || line[0] == '#')
            continue;
          std::istringstream iss(line);
          std::string type;
          iss >> type;

          if (type == "ftrl" || type == "hbt_ftrl") {
            iss >> ftrl_alpha >> ftrl_beta >> ftrl_l1 >> ftrl_l2;
          } else if (type == "hbt") {
            iss >> hbt_decay_period >> hbt_decay_amount;
          } else if (type == "features_hbt") {
            state = ParseState::HBT;
          } else if (type == "features_many_branches") {
            state = ParseState::MANY_BRANCHES;
          } else if (type == "features") {
            // Legacy support: default to HBT
            state = ParseState::HBT;
          } else {
            // Feature declaration: Name arg1 arg2 ...
            std::vector<std::string> args;
            std::string arg;
            while (iss >> arg) {
              args.push_back(arg);
            }

            if (state == ParseState::HBT) {
              // HBT features use PC by default
              hbt_index_manager.add_feature(create_feature(type, args, true));
            } else if (state == ParseState::MANY_BRANCHES) {
              // Many branches features do NOT use PC by default
              hb_index_manager.add_feature(create_feature(type, args, false));
            } else {
              // Ignore or error if no section defined?
              // Legacy support: if features seen before section, discard or
              // adding to default HBT? Let's assume HBT if unspecified to avoid
              // breakage
              hbt_index_manager.add_feature(create_feature(type, args, true));
            }
          }
        }
        config_loaded = true;
      } else {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        exit(1);
      }
    }

    assert(config_loaded);

    // 2. Initialize FTRLs
    hbt_ftrl = std::make_unique<HbtFTRL>(
        hbt_index_manager.get_num_features(), ftrl_alpha, ftrl_beta, ftrl_l1,
        ftrl_l2, hbt_decay_period, hbt_decay_amount);

    hb_ftrl = std::make_unique<HighBranchFTRL>(
        hb_index_manager.get_num_features(), ftrl_alpha, ftrl_beta, ftrl_l1,
        ftrl_l2);

    // 3. Initialize FSCs
    int tag_size = 12;
    // HBT FSC: Always Replace
    hbt_fsc = std::unique_ptr<FSC>(
        new FSC(hbt_index_manager.get_num_features(), tag_size,
                FSC::AllocationPolicy::ALWAYS_REPLACE));
    // High Branch FSC: Check Magnitude
    /* hb_fsc = std::unique_ptr<FSC>(
        new FSC(hb_index_manager.get_num_features(), tag_size,
                FSC::AllocationPolicy::CHECK_MAGNITUDE)); */

    printf("HBT FSC Size: %zu bits\n", hbt_fsc->get_size());
    /* printf("HB FSC Size: %zu bits\n", hb_fsc->get_size()); */
  }

  void terminate() {}

  // sample function to get unique instruction id
  uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    assert(piece < 16);
    return (seq_no << 4) | (piece & 0x000F);
  }

  bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC,
               const TagePrediction &tage_pred) {

    // 1. Get Indices
    std::vector<uint64_t> hbt_indices;
    hbt_index_manager.get_indices(PC, hbt_indices);

    std::vector<uint64_t> hb_indices;
    hb_index_manager.get_indices(PC, hb_indices);

    // 2. Get FSC Predictions
    float hbt_fsc_sum = hbt_fsc->get_prediction(hbt_indices);
    // float hb_fsc_sum = hb_fsc->get_prediction(hb_indices);

    // 3. Get Tage Signed Weight
    float tage_weight = 0.0f;
    float sign = tage_pred.prediction ? 1.0f : -1.0f;
    int confidence = tage_pred.confidence;

    if (confidence == 2)
      tage_weight = 0.9f; // High
    else if (confidence == 1)
      tage_weight = 0.4f; // Med
    else
      tage_weight = 0.2f; // Low

    tage_weight *= sign;

    // 4. Combine
    float total_sum = tage_weight;
    // Add HBT FSC
    // Note: get_conf was always true in previous version
    if (hbt_fsc->get_conf(PC)) {
      total_sum += hbt_fsc_sum;
    }
    // Add HB FSC (no skew gating/conf check mentioned, just add?)
    // User: "add them both together etc etc etc"
    // total_sum += hb_fsc_sum;

    bool final_pred = (total_sum >= 0);

    // 5. Sto§re State for Update
    PredState state;
    state.hbt_indices = std::move(hbt_indices);
    state.hb_indices = std::move(hb_indices);
    state.sum = total_sum;

    HistoryEntry entry;
    entry.state = std::move(state);
    entry.tage_pred = tage_pred;
    entry.combined_pred = final_pred;
    entry.hbt_fsc_sum = hbt_fsc_sum;
    // entry.hb_fsc_sum = hb_fsc_sum;

    pred_time_histories.emplace(get_unique_inst_id(seq_no, piece),
                                std::move(entry));

    return final_pred;
  }

  void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken,
                      uint64_t nextPC) {
    // Update all features
    hbt_index_manager.update_features(InstClass::condBranchInstClass, PC, taken,
                                      nextPC);
    hb_index_manager.update_features(InstClass::condBranchInstClass, PC, taken,
                                     nextPC);
  }

  void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir,
              bool predDir, uint64_t nextPC, uint64_t cycle) {
    auto it = pred_time_histories.find(get_unique_inst_id(seq_no, piece));
    if (it == pred_time_histories.end())
      return;

    const auto &entry = it->second;
    const auto &state = entry.state;

    // --- mispred tracking: main tage vs mini tage ---
    br_count++;
    if (entry.tage_pred.prediction != resolveDir)
      main_tage_misp++;

    // Train FTRL (Update weights)
    float prob = 1.0f / (1.0f + std::exp(-state.sum));

    // Train HBT FTRL
    hbt_ftrl->update(state.hbt_indices, prob, resolveDir, cycle, PC,
                     entry.tage_pred);

    // Train HB FTRL
    /* hb_ftrl->update(state.hb_indices, prob, resolveDir, cycle, PC,
                    entry.tage_pred); */

    // Update Skew (using HBT FSC? Only HBT FSC had get_conf usage in predict)
    // Assuming skew logic applies to the aggregate or just HBT?
    // Usually FSC is about cached weights. Both have weights.
    // Let's update skew using the aggregate prediction for now or just skip
    // exact skew logic if undefined. Original: fsc->update_skew(PC,
    // tage_correct, fsc_correct); Here we have TWO FSCs.

    bool tage_correct = (entry.tage_pred.prediction == resolveDir);
    // Which FSC correctness?
    bool hbt_fsc_correct = ((entry.hbt_fsc_sum >= 0) == resolveDir);
    hbt_fsc->update_skew(PC, tage_correct, hbt_fsc_correct);

    // hb_fsc doesn't have skew gating in predict currently, so maybe no need to
    // update skew? Or maybe it should have skew. Leaving as is (only HBT FSC
    // has skew).

    pred_time_histories.erase(it);
  }

  // Called at commit time to move 1 weight from FTRL to FSC
  void commit(uint64_t seq_no, uint8_t piece, uint64_t pc, bool pred_dir,
              bool resolve_dir) {
    // Unused in provided snippet
  }

  void timestep(uint64_t cycle) {
    if (cycle < 20)
      return;
    uint64_t threshold = cycle - 20;

    // Transfer HBT weights
    std::vector<ActiveWeight> old_weights;
    hbt_ftrl->pop_active_weights_older_than(threshold, old_weights);
    for (const auto &w : old_weights) {
      hbt_fsc->allocate(w.hash, w.feature_idx, w.weight);
    }

    // Transfer HB weights
    /* old_weights.clear();
    hb_ftrl->pop_active_weights_older_than(threshold, old_weights);
    for (const auto &w : old_weights) {
      hb_fsc->allocate(w.hash, w.feature_idx, w.weight);
    } */
  }

  void print_performance() const {}
};

#endif
static SampleCondPredictor cond_predictor_impl;
