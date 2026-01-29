#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <deque>
#include "../tage_prediction.h"

struct ActiveWeight {
    uint64_t hash;  // The raw feature index/hash provided by GetIndex
    float weight;
    size_t feature_idx;
    uint64_t cycle;
};

class FTRL {
private:
    struct TableState {
        std::vector<float> z_table;
        std::vector<float> n_table;
        
        TableState(size_t size) 
            : z_table(size, 0.0f), 
              n_table(size, 0.0f)
        {}
    };

    std::vector<TableState> tables;
    std::deque<ActiveWeight> nonzero_weights;
    
    std::vector<uint8_t> bloom_taken;
    std::vector<uint8_t> bloom_not_taken;
    uint64_t bloom_decay_period;
    
    // Hard Branch Table (HBT) for detecting hard-to-predict branches
    struct HBTEntry {
        uint8_t misp_counter;  // 5-bit saturating counter (0-31)
        uint64_t tag;          // Partial PC tag for identification
        bool valid;            // Entry validity flag
        
        HBTEntry() : misp_counter(0), tag(0), valid(false) {}
    };
    
    std::vector<HBTEntry> hbt;
    uint64_t hbt_decay_period;
    uint8_t hbt_decay_amount;
    uint64_t retired_branches;

    static const size_t TABLE_SIZE = 1 << 14;  // 16K entries for HBT and Bloom
    
    float alpha;
    float beta;
    float l1;
    float l2;
    
    float compute_weight_internal(float z, float n) const {
        if (std::abs(z) <= l1) return 0.0f;
        return -(z - (z > 0 ? 1.0f : -1.0f) * l1) / (((beta + std::sqrt(n)) / alpha) + l2);
    }
    
    // Helper methods
    uint64_t hash_index(uint64_t pc) const {
        return pc & (TABLE_SIZE - 1);
    }
    
    uint64_t compute_tag(uint64_t pc) const {
        // Use upper bits of PC as tag
        return (pc >> 14) & 0xFFFF;
    }
    
    bool is_hard_to_predict(uint64_t pc) const {
        size_t idx = hash_index(pc);
        const auto& entry = hbt[idx];
        if (!entry.valid || entry.tag != compute_tag(pc)) return false;
        // Counter saturates at 31 (5 bits)
        return entry.misp_counter >= 31;
    }
    
    void update_hbt(uint64_t pc, bool mispredicted, bool hit_in_last) {
        // Only allocate/update if branch hit in last history table
        if (!mispredicted) return;
        
        size_t idx = hash_index(pc);
        auto& entry = hbt[idx];
        uint64_t tag = compute_tag(pc);
        
        // Allocate new entry or update existing
        if (entry.tag != tag) {
            // Only allocate if counter is 0 (allows overwriting old entries)
            if (entry.misp_counter == 0 && hit_in_last) {
                entry.valid = true;
                entry.tag = tag;
                entry.misp_counter = 1;
            }
        } else {
            // Saturate at 31
            if (entry.misp_counter < 31) entry.misp_counter++;
        }
    }
    
    void apply_leaky_bucket_hbt() {
        for (auto& entry : hbt) {
            if (entry.valid && entry.misp_counter > 0) {
                uint8_t decrement = std::min(hbt_decay_amount, entry.misp_counter);
                entry.misp_counter -= decrement;
                // Invalidate entry if counter reaches 0
                if (entry.misp_counter == 0) {
                    entry.valid = false;
                }
            }
        }
    }

    void apply_leaky_bucket_bloom() {
        for (size_t i = 0; i < TABLE_SIZE; ++i) {
            if (bloom_taken[i] > 0) bloom_taken[i]--;
            if (bloom_not_taken[i] > 0) bloom_not_taken[i]--;
        }
    }
    
    void update_bloom(uint64_t pc, bool taken) {
        size_t idx = hash_index(pc);
        if (taken) {
            if (bloom_taken[idx] < 3) bloom_taken[idx]++;
        } else {
            if (bloom_not_taken[idx] < 3) bloom_not_taken[idx]++;
        }
    }

    bool should_train(uint64_t pc) const {
        size_t idx = hash_index(pc);
        return (bloom_taken[idx] > 0 && bloom_not_taken[idx] > 0);
    }

public:
    FTRL(size_t num_features, float _alpha, float _beta, float _l1, float _l2, uint64_t _bloom_decay_period, uint64_t _hbt_decay_period, uint8_t _hbt_decay_amount)
        : alpha(_alpha), beta(_beta), l1(_l1), l2(_l2),
          bloom_decay_period(_bloom_decay_period),
          hbt_decay_period(_hbt_decay_period), hbt_decay_amount(_hbt_decay_amount),
          bloom_taken(TABLE_SIZE, 0), bloom_not_taken(TABLE_SIZE, 0),
          hbt(TABLE_SIZE),
          retired_branches(0)
    {
        if (alpha <= 0.0f) throw std::runtime_error("FTRL alpha must be > 0");
        tables.reserve(num_features);
        size_t fixed_size = 1 << 14; 
        for (size_t i = 0; i < num_features; ++i) {
            tables.emplace_back(fixed_size);
        }
    }
    
    // Update weights based on indices provided by IndexManager
    // Requires TagePrediction for feedback
    void update(const std::vector<uint64_t>& indices, float pred, bool actual, uint64_t cycle, uint64_t pc, const TagePrediction& tage_pred) {
        
        retired_branches++;
        
        // 1. Update Bloom Filters
        update_bloom(pc, actual);
        
        // Bloom Filter decay
        if (retired_branches % bloom_decay_period == 0) {
            apply_leaky_bucket_bloom();
        }

        // HBT decay
        if (retired_branches % hbt_decay_period == 0) {
            apply_leaky_bucket_hbt();
        }
        
        // 2. Update Hard Branch Table
        bool mispredicted = (tage_pred.prediction != actual);
        update_hbt(pc, mispredicted, tage_pred.hit_in_last_history_table);
        
        // 3. FTRL Training Gate
        if (should_train(pc)) {
            float g = pred - (actual ? 1.0f : 0.0f);
            float g2 = g * g;
            
            size_t count = std::min(indices.size(), tables.size());
            
            for (size_t i = 0; i < count; ++i) {
                uint64_t raw_idx = indices[i];
                auto& table = tables[i];
                
                size_t idx = raw_idx & 0x3FFF;
                
                float zi = table.z_table[idx];
                float ni = table.n_table[idx];
                
                float weight = compute_weight_internal(zi, ni);
                
                float si = (std::sqrt(ni + g2) - std::sqrt(ni)) / alpha;
                table.z_table[idx] += g - si * weight;
                table.n_table[idx] += g2;
            }
        }
        
        // 4. Weight Synchronization to FSC via HBT check
        bool hard = is_hard_to_predict(pc);

        // We need to send weights for this PC's indices. 
        // If Hard: send actual weights. 
        // If Not Hard: send 0.0f.
        // NOTE: We only need to send updates if we trained OR if we need to clear (send 0).
        // Actually, "If it is not hard to predict, send over zero weights (they should zero the weights if theyre there already)."
        // This implies we should ALWAYS send updates potentially? Or only if we think they are in FSC?
        // Since we don't know if they are in FSC efficiently, we should probably send them.
        // But sending every cycle is expensive? 
        // The prompt says "Then at the end... send the weights over." implies every update call.
        
        size_t count = std::min(indices.size(), tables.size());
        for (size_t i = 0; i < count; ++i) {
            uint64_t raw_idx = indices[i];
            float weight_to_send = 0.0f;
            
            if (hard) {
                const auto& table = tables[i];
                size_t idx = raw_idx & 0x3FFF;
                weight_to_send = compute_weight_internal(table.z_table[idx], table.n_table[idx]);
            }
            
            // Push to queue
            nonzero_weights.push_back({raw_idx, weight_to_send, i, cycle});
        }
    }
    
    // Get weight for a specific table and index
    float get_weight(size_t table_idx, uint64_t idx) const {
        if (table_idx >= tables.size()) return 0.0f;
        const auto& table = tables[table_idx];
        size_t effective_idx = idx % table.z_table.size();
        
        return compute_weight_internal(table.z_table[effective_idx], table.n_table[effective_idx]);
    }

    const std::deque<ActiveWeight>& get_nonzero_weights() const {
        return nonzero_weights;
    }
    
    // Pop one weight from the queue (for FSC sync)
    bool pop_active_weight(ActiveWeight& out) {
        if (nonzero_weights.empty()) return false;
        out = nonzero_weights.front();
        nonzero_weights.pop_front();
        return true;
    }

    void pop_active_weights_older_than(uint64_t cycle_threshold, std::vector<ActiveWeight>& out) {
        while (!nonzero_weights.empty()) {
            const auto& front = nonzero_weights.front();
            if (front.cycle <= cycle_threshold) {
                out.push_back(front);
                nonzero_weights.pop_front();
            } else {
                // Since weights are pushed in chronological order, we can stop early
                break;
            }
        }
    }

    void clear_nonzero_queue() {
        nonzero_weights.clear();
    }
    
    size_t size_bytes() const {
        size_t total = 0;
        for (const auto& t : tables) {
            total += t.z_table.size() * sizeof(float);
            total += t.n_table.size() * sizeof(float);
        }
        total += hbt.size() * sizeof(HBTEntry);
        return total;
    }
};
