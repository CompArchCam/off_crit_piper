#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>

#include "piper/feature.h"
#include "piper/features.h"
#include "piper/fsc.h"
#include "piper/ftrl.h"
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
  std::vector<Index> hbt_indices;
  float sum;
};

class SampleCondPredictor {
  // Components
  IndexManager hbt_index_manager;

  std::unique_ptr<HbtFTRL> hbt_ftrl;

  std::unique_ptr<FSC> hbt_fsc;

  // State tracking
  struct HistoryEntry {
    PredState state;
    branch_info bi;
    TagePrediction tage_pred;
    bool combined_pred;
    float hbt_fsc_sum;
    float tage_weight;
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

    bool config_loaded = false;

    // Default to HBT mode
    enum class ParseState { NONE, HBT };
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
          } else if (type == "features_hbt" || type == "features") {
            state = ParseState::HBT;
          } else {
            // Feature declaration: Name arg1 arg2 ...
            std::vector<std::string> args;
            std::string arg;
            while (iss >> arg) {
              args.push_back(arg);
            }

            // HBT features use PC by default
            hbt_index_manager.add_feature(create_feature(type, args, true));
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
    std::vector<std::optional<uint64_t>> feature_sizes =
        hbt_index_manager.get_sizes();

    hbt_ftrl =
        std::make_unique<HbtFTRL>(feature_sizes, ftrl_alpha, ftrl_beta, ftrl_l1,
                                  ftrl_l2, hbt_decay_period, hbt_decay_amount);

    // 3. Initialize FSCs
    int tag_size = 12;
    hbt_fsc = std::unique_ptr<FSC>(new FSC(
        feature_sizes, tag_size, FSC::AllocationPolicy::CHECK_MAGNITUDE));

    printf("HBT FSC Size: %zu bits %zu KB\n", hbt_fsc->get_size(),
           hbt_fsc->get_size() / 1024 / 8);
    printf("HBT FTRL Size: %zu bits %zu KB\n", hbt_ftrl->size_bits(),
           hbt_ftrl->size_bits() / 8 / 1024);
  }

  void terminate() {}

  // sample function to get unique instruction id
  uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    assert(piece < 16);
    return (seq_no << 4) | (piece & 0x000F);
  }

  bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC,
               const TagePrediction &tage_pred) {

    // 0. Construct Branch Info
    branch_info bi;
    bi.pc = PC;
    bi.target = 0; // Target unknown at prediction
    // Assuming high confidence if confidence counter is maxed (e.g. 3 for
    // 2-bit? TAGE usually has counters) TAGE usually has 3-bit counters (0..7)
    // or similar. The existing code Checks 2 -> High, 1 -> Med. Max seems to be
    // 3 in existing code print? "if (confidence == 3)" is used.
    bi.tage_conf = (tage_pred.confidence >= 3);
    bi.hcpred = tage_pred.hcpred;
    bi.longestmatchpred = tage_pred.longestmatchpred;

    // 1. Get Indices
    std::vector<Index> hbt_indices;
    hbt_index_manager.get_indices(bi, hbt_indices);

    // 2. Get FSC Predictions
    float hbt_fsc_sum = hbt_fsc->get_prediction(hbt_indices);

    // 3. Get Tage Signed Weight
    float tage_weight = 0.0f;
    float sign = tage_pred.prediction ? 1.0f : -1.0f;
    int confidence = tage_pred.confidence;

    if (confidence == 3) {
      std::cout << "confidence: " << confidence << std::endl;
    }
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

    bool final_pred = (total_sum >= 0);

    // 5. Sto§re State for Update
    PredState state;
    state.hbt_indices = std::move(hbt_indices);
    state.sum = total_sum;

    HistoryEntry entry;
    entry.state = std::move(state);
    entry.bi = bi;
    entry.tage_pred = tage_pred;
    entry.combined_pred = final_pred;
    entry.hbt_fsc_sum = hbt_fsc_sum;
    entry.tage_weight = tage_weight;

    pred_time_histories.emplace(get_unique_inst_id(seq_no, piece),
                                std::move(entry));

    return final_pred;
  }

  void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken,
                      uint64_t nextPC) {

    branch_info bi;
    bi.pc = PC;
    bi.target = nextPC;
    bi.tage_conf = false; // Not available/needed for history update usually

    // Update all features
    hbt_index_manager.update_features(InstClass::condBranchInstClass, taken,
                                      bi);
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
                     entry.tage_pred, entry.tage_weight);

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
      hbt_fsc->allocate(w.index, w.feature_idx, w.weight);
    }
  }

  void print_performance() const {}
};

#endif
static SampleCondPredictor cond_predictor_impl;
