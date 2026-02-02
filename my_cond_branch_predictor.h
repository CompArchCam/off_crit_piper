#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>

#include "piper/feature.h"
#include "piper/features.h"
#include "piper/fsc.h"
#include "piper/ftrl.h"
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
  std::vector<uint64_t> indices;
  float sum;
};

class SampleCondPredictor {
  // Components
  IndexManager index_manager;
  std::unique_ptr<FTRL> ftrl;
  std::unique_ptr<FSC> fsc;

  // State tracking
  struct HistoryEntry {
    PredState state;
    TagePrediction tage_pred;
    bool combined_pred;
    float fsc_sum;
  };
  std::unordered_map<uint64_t, HistoryEntry> pred_time_histories;

  // --- periodic mispred reporting (main tage vs mini tage) ---
  uint64_t br_count = 0;
  uint64_t main_tage_misp = 0;

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

          if (type == "ftrl") {
            iss >> ftrl_alpha >> ftrl_beta >> ftrl_l1 >> ftrl_l2;
          } else if (type == "hbt") {
            iss >> hbt_decay_period >> hbt_decay_amount;
          } else if (type == "features") {
            // section header, ignore
          } else {
            // Feature declaration: Name arg1 arg2 ...
            std::vector<std::string> args;
            std::string arg;
            while (iss >> arg) {
              args.push_back(arg);
            }

            if (type == "PC") {
              index_manager.add_feature(std::make_unique<PCFeature>(args));
            } else if (type == "GHist") {
              index_manager.add_feature(std::make_unique<GHistFeature>(args));
            } else if (type == "GPath") {
              index_manager.add_feature(std::make_unique<GPathFeature>(args));
            } else if (type == "LHist") {
              index_manager.add_feature(std::make_unique<LHistFeature>(args));
            } else if (type == "Recency") {
              index_manager.add_feature(std::make_unique<RecencyFeature>(args));
            } else if (type == "RecencyPos") {
              index_manager.add_feature(
                  std::make_unique<RecencyPosFeature>(args));
            } else if (type == "IMLI") {
              index_manager.add_feature(std::make_unique<IMLIFeature>(args));
            } else if (type == "BlurryPath") {
              index_manager.add_feature(
                  std::make_unique<BlurryPathFeature>(args));
            } else if (type == "ReturnStackHist") {
              index_manager.add_feature(
                  std::make_unique<ReturnStackHistFeature>(args));
            } else if (type == "BrIMLI") {
              index_manager.add_feature(
                  std::unique_ptr<BrIMLIFeature>(new BrIMLIFeature(args)));
            } else if (type == "TaIMLI") {
              index_manager.add_feature(
                  std::unique_ptr<TaIMLIFeature>(new TaIMLIFeature(args)));
            } else if (type == "PHist") {
              index_manager.add_feature(std::make_unique<PHistFeature>(args));
            } else if (type == "FHist") {
              index_manager.add_feature(std::make_unique<FHistFeature>(args));
            } else if (type == "BHist") {
              index_manager.add_feature(std::make_unique<BHistFeature>(args));
            } else if (type == "SLHist") {
              index_manager.add_feature(std::make_unique<SLHistFeature>(args));
            } else if (type == "TLHist") {
              index_manager.add_feature(std::make_unique<TLHistFeature>(args));
            } else if (type == "QLHist") {
              index_manager.add_feature(std::make_unique<QLHistFeature>(args));
            } else {
              std::cerr << "Unknown feature type: " << type << std::endl;
              exit(1);
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

    // 2. Initialize FTRL
    ftrl = std::unique_ptr<FTRL>(
        new FTRL(index_manager.get_num_features(), ftrl_alpha, ftrl_beta,
                 ftrl_l1, ftrl_l2, hbt_decay_period, hbt_decay_amount));

    // 3. Initialize FSC (fast path cache)
    // Use tag bits: 8 bits for all
    // 3. Initialize FSC (fast path cache)
    // Use tag bits: 8 bits for all
    int tag_size = 12;
    fsc = std::unique_ptr<FSC>(
        new FSC(index_manager.get_num_features(), tag_size));

    printf("FSC Size: %zu bits\n", fsc->get_size());
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
    std::vector<uint64_t> indices;
    index_manager.get_indices(PC, indices);

    // 2. Get FSC Prediction (Sum of weights)
    float fsc_sum = fsc->get_prediction(indices);

    // 3. Get Tage Signed Weight
    float tage_weight = 0.0f;
    float sign = tage_pred.prediction ? 1.0f : -1.0f;
    int confidence = tage_pred.confidence;

    if (confidence == 2)
      tage_weight = 0.9f; // High
    else if (confidence == 1)
      tage_weight = 0.4f; // Med (was 0.75)
    else
      tage_weight = 0.2f; // Low (was 0.5)

    tage_weight *= sign;

    // 4. Combine
    float total_sum = tage_weight;
    if (fsc->get_conf(PC)) {
      total_sum += fsc_sum;
    }
    bool final_pred = (total_sum >= 0);

    // 5. Sto§re State for Update
    PredState state;
    state.indices = std::move(indices);
    state.sum = total_sum;

    HistoryEntry entry;
    entry.state = std::move(state);
    entry.tage_pred = tage_pred;
    entry.combined_pred = final_pred;
    entry.fsc_sum = fsc_sum;

    pred_time_histories.emplace(get_unique_inst_id(seq_no, piece),
                                std::move(entry));

    return final_pred;
  }

  void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken,
                      uint64_t nextPC) {
    // Update all features
    // Note: InstClass not passed here in sample interface?
    // spec_update in interface passes inst_class but history_update signature
    // here doesn't have it. We can assume condBranch for now or update
    // signature. The interface calls this only for cond branches.
    index_manager.update_features(InstClass::condBranchInstClass, PC, taken,
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

    // Train FTRL (Update weights in full model)
    // Use the indices we captured at prediction time
    // User request: "return the prob as 1.0f / (1.0f + std::exp(-dot)); as the
    // pred float instead of this predDir ? 1.0f : 0.0f"
    float prob = 1.0f / (1.0f + std::exp(-state.sum));
    ftrl->update(state.indices, prob, resolveDir, cycle, PC, entry.tage_pred);

    // If FSC was incorrect, zero out its weights so FTRL can refresh them
    bool fsc_pred_dir = (entry.fsc_sum >= 0.0f);
    bool fsc_correct = (fsc_pred_dir == resolveDir);
    bool tage_correct = (entry.tage_pred.prediction == resolveDir);

    fsc->update_skew(PC, tage_correct, fsc_correct);
    // if (!fsc_correct) {
    // fsc->update_weights_fast(state.indices, resolveDir);
    //}

    pred_time_histories.erase(it);
  }

  // Called at commit time to move 1 weight from FTRL to FSC
  void commit(uint64_t seq_no, uint8_t piece, uint64_t pc, bool pred_dir,
              bool resolve_dir) {
    // if (!ftrl) return;

    // ActiveWeight entry;
    // ftrl->pop_active_weight(entry);
    // fsc->allocate(entry.hash, entry.feature_idx, entry.weight);
  }

  void timestep(uint64_t cycle) {
    if (cycle < 20)
      return;
    uint64_t threshold = cycle - 20;
    std::vector<ActiveWeight> old_weights;
    ftrl->pop_active_weights_older_than(threshold, old_weights);
    for (const auto &w : old_weights) {
      fsc->allocate(w.hash, w.feature_idx, w.weight);
    }
  }

  void print_performance() const {}
};

#endif
static SampleCondPredictor cond_predictor_impl;
