#ifndef _PREDICTOR_H_
#define _PREDICTOR_H_

#include <stdlib.h>



#include "piper/index_manager.h"
#include "piper/ftrl.h"
#include "piper/fsc.h"
#include "piper/features.h"
#include "piper/feature.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <deque>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>

struct PredState {
    std::vector<uint64_t> indices;
    float sum;
};

// Debugger Class
class Debugger {
    uint64_t total_branches = 0;
    uint64_t tage_correct = 0;
    uint64_t combined_correct = 0;
    uint64_t fixed_by_fsc = 0;
    uint64_t worsened_by_fsc = 0;
    
public:
    void update(bool tage_pred, bool combined_pred, bool actual) {
        total_branches++;
        bool t_corr = (tage_pred == actual);
        bool c_corr = (combined_pred == actual);
        
        if (t_corr) tage_correct++;
        if (c_corr) combined_correct++;
        
        if (!t_corr && c_corr) fixed_by_fsc++;
        if (t_corr && !c_corr) worsened_by_fsc++;
    }
    
    void print_stats() const {
        printf("\n=== MyPredictor Debugger Stats ===\n");
        printf("Total Branches: %llu\n", total_branches);
        printf("Tage Correct:   %llu (%.4f%%)\n", tage_correct, 100.0 * tage_correct / (double)total_branches);
        printf("Hybrid Correct: %llu (%.4f%%)\n", combined_correct, 100.0 * combined_correct / (double)total_branches);
        printf("Fixed by FSC:   %llu\n", fixed_by_fsc);
        printf("Worsened by FSC: %llu\n", worsened_by_fsc);
        int64_t net = (int64_t)fixed_by_fsc - (int64_t)worsened_by_fsc;
        printf("Net Benefit:    %lld\n", net);
        printf("==================================\n");
    }

    void print_performance() const {
        uint64_t tage_misp = total_branches - tage_correct;
        uint64_t combined_misp = total_branches - combined_correct;
        printf("ReferenceMispred/Mispred: %llu/%llu\n", combined_misp, tage_misp);
    }
};

class SampleCondPredictor
{
        // Components
        IndexManager index_manager;
        std::unique_ptr<FTRL> ftrl;
        std::unique_ptr<FSC> fsc;
        
        // Debugger
        Debugger debugger;
        
        // State tracking
        struct HistoryEntry {
            PredState state;
            bool tage_pred;
            bool combined_pred;
            float fsc_sum;
        };
        std::unordered_map<uint64_t, HistoryEntry> pred_time_histories;

    public:

        SampleCondPredictor (void)
        {
        }

        void setup(const std::string& config_path)
        {
            float ftrl_alpha = 0.4f;
            float ftrl_beta = 0.8f;
            float ftrl_l1 = 0.2f;
            float ftrl_l2 = 0.6f;
            
            bool config_loaded = false;
            if (!config_path.empty()) {
                std::ifstream infile(config_path);
                if (infile.is_open()) {
                    std::string line;
                    while (std::getline(infile, line)) {
                        if (line.empty() || line[0] == '#') continue;
                        std::istringstream iss(line);
                        std::string type;
                        iss >> type;
                        
                        if (type == "ftrl") {
                            iss >> ftrl_alpha >> ftrl_beta >> ftrl_l1 >> ftrl_l2;
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
                            } else if (type == "RecencyPos") {
                                index_manager.add_feature(std::make_unique<RecencyPosFeature>(args));
                            } else if (type == "IMLI") {
                                index_manager.add_feature(std::make_unique<IMLIFeature>(args));
                            } else if (type == "BlurryPath") {
                                index_manager.add_feature(std::make_unique<BlurryPathFeature>(args));
                            } else if (type == "ReturnStackHist") {
                                index_manager.add_feature(std::make_unique<ReturnStackHistFeature>(args));
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
            ftrl = std::make_unique<FTRL>(index_manager.get_num_features(), 
                ftrl_alpha,
                ftrl_beta,
                ftrl_l1,
                ftrl_l2
            );

            // 3. Initialize FSC (fast path cache)
            // Use tag bits: 8 bits for all
            std::vector<int> tag_bits(index_manager.get_num_features(), 8);
            fsc = std::make_unique<FSC>(index_manager.get_num_features(), tag_bits);
        }

        void terminate()
        {
            debugger.print_stats();
        }

        // sample function to get unique instruction id
        uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const
        {
            assert(piece < 16);
            return (seq_no << 4) | (piece & 0x000F);
        }

        bool predict (uint64_t seq_no, uint8_t piece, uint64_t PC, const bool tage_pred, const int confidence)
        {
            // 1. Get Indices
            std::vector<uint64_t> indices;
            index_manager.get_indices(PC, indices);
            
            // 2. Get FSC Prediction (Sum of weights)
            float fsc_sum = fsc->get_prediction(indices);
            
            // 3. Get Tage Signed Weight
            float tage_weight = 0.0f;
            float sign = tage_pred ? 1.0f : -1.0f;
            
            if (confidence == 2) tage_weight = 1.0f;       // High
            else if (confidence == 1) tage_weight = 0.5f;  // Med
            else tage_weight = 0.1f;                       // Low
            
            tage_weight *= sign;
            
            // 4. Combine
            float total_sum = tage_weight + fsc_sum;
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
            
            pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), std::move(entry));
            
            return final_pred;
        }

        void history_update (uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken, uint64_t nextPC)
        {
            // Update all features
            // Note: InstClass not passed here in sample interface? 
            // spec_update in interface passes inst_class but history_update signature here doesn't have it.
            // We can assume condBranch for now or update signature. 
            // The interface calls this only for cond branches.
            index_manager.update_features(InstClass::condBranchInstClass, PC, taken); 
        }

        void update (uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC, uint64_t cycle)
        {
            auto it = pred_time_histories.find(get_unique_inst_id(seq_no, piece));
            if (it == pred_time_histories.end()) return;
            
            const auto& entry = it->second;
            const auto& state = entry.state;
            
            // Update Debugger
            debugger.update(entry.tage_pred, entry.combined_pred, resolveDir);
            
            // Train FTRL (Update weights in full model)
            // Use the indices we captured at prediction time
            // User request: "return the prob as 1.0f / (1.0f + std::exp(-dot)); as the pred float instead of this predDir ? 1.0f : 0.0f"
            float prob = 1.0f / (1.0f + std::exp(-state.sum));
            ftrl->update(state.indices, prob, resolveDir, cycle);
            
            // If FSC was incorrect, zero out its weights so FTRL can refresh them
            bool fsc_pred_dir = (entry.fsc_sum >= 0.0f);
            if (fsc_pred_dir != resolveDir) {
                fsc->update_weights_fast(state.indices, resolveDir);
            }
            
            pred_time_histories.erase(it);
        }
        
        // Called at commit time to move 1 weight from FTRL to FSC
        void commit(uint64_t seq_no, uint8_t piece, uint64_t pc, bool pred_dir, bool resolve_dir) {
            // if (!ftrl) return;

            // ActiveWeight entry;
            // ftrl->pop_active_weight(entry);
            // fsc->allocate(entry.hash, entry.feature_idx, entry.weight);
        }

        void timestep(uint64_t cycle) {
             if (cycle < 20) return;
             uint64_t threshold = cycle - 20;
             std::vector<ActiveWeight> old_weights;
             ftrl->pop_active_weights_older_than(threshold, old_weights);
             for (const auto& w : old_weights) {
                 fsc->allocate(w.hash, w.feature_idx, w.weight);
             }
        }

        void print_performance() const {
             debugger.print_performance();
        }
};

#endif
static SampleCondPredictor cond_predictor_impl;
