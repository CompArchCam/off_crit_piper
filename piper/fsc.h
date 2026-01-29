#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

class FSC {
private:
    struct FscEntry {
        uint16_t tag = 0;
        float weight = 0.0f;
        // Could add valid bit, but weight 0.0f or initial tag could serve? 
        // Better to have explicit empty state or initialization. 
        // Let's use weight 0.0f as implicit empty or just overwrite.
        // User said "If there are no non zero weights in the way", implying nonzero weight means occupied.
    };

    static constexpr size_t NUM_WAYS = 4;
    static constexpr size_t NUM_SETS = 128; // 128 entries per way = 512 total
    
    // Each feature has its own table
    std::vector<std::vector<FscEntry>> tables;
    std::vector<int> tag_bits;

    // Helper: Compute tag from raw index
    uint16_t compute_tag(uint64_t index, int bits) const {
        // Simple tag derivation: mix index or just shift
        // Since set index uses low bits (index % 128), tag should use higher bits
        // Let's just shift right by 7 (since 128 = 2^7)
        // But user said "configurable tag sizes".
        uint64_t shifted = index >> 7; 
        uint64_t mask = (1ULL << bits) - 1;
        return static_cast<uint16_t>(shifted & mask);
    }
    
    // Helper: Get set index
    size_t get_set(uint64_t index) const {
        return index % NUM_SETS;
    }

public:
    FSC(size_t num_features, const std::vector<int>& _tag_bits) 
        : tag_bits(_tag_bits) {
        if (tag_bits.size() != num_features) {
            throw std::runtime_error("FSC: tag_bits size must match num_features");
        }
        
        tables.resize(num_features);
        for (auto& table : tables) {
            table.resize(NUM_SETS * NUM_WAYS);
        }
    }

    // Get prediction probability by summing weights of hits
    float get_prediction(const std::vector<uint64_t>& indices) const {
        if (indices.size() != tables.size()) {
            // Should match, typically
        }

        float sum = 0.0f;
        size_t count = std::min(indices.size(), tables.size());

        for (size_t i = 0; i < count; ++i) {
            uint64_t index = indices[i];
            size_t set = get_set(index);
            uint16_t tag = compute_tag(index, tag_bits[i]);
            
            const auto& table = tables[i];
            size_t base = set * NUM_WAYS;
            
            for (size_t w = 0; w < NUM_WAYS; ++w) {
                const auto& entry = table[base + w];
                if (entry.weight != 0.0f && entry.tag == tag) {
                    sum += entry.weight;
                    break; // Assumes only one match per set (enforced by allocation)
                }
            }
        }

        return sum;
    }

    // Allocate/Update entry
    void allocate(uint64_t index, size_t feature_idx, float weight) {
        size_t set = get_set(index);
        uint16_t tag = compute_tag(index, tag_bits[feature_idx]);
        
        auto& table = tables[feature_idx];
        size_t base = set * NUM_WAYS;
        
        // 1. Check for existing match to update
        for (size_t w = 0; w < NUM_WAYS; ++w) {
            if (table[base + w].tag == tag) {
                table[base + w].weight = weight;
                return;
            }
        }
        
        if (weight == 0.0f) return; // Don't allocate zero weights

        // 2. Find empty slot
        int victim_way = -1;
        for (size_t w = 0; w < NUM_WAYS; ++w) {
            if (table[base + w].weight == 0.0f) {
                victim_way = w;
                break;
            }
        }

        // 3. If no empty slot, find smallest absolute weight
        if (victim_way == -1) {
            float min_abs_weight = std::numeric_limits<float>::max();
            for (size_t w = 0; w < NUM_WAYS; ++w) {
                float abs_w = std::abs(table[base + w].weight);
                if (abs_w < min_abs_weight) {
                    min_abs_weight = abs_w;
                    victim_way = w;
                }
            }
        }

        // 4. Update victim
        if (victim_way != -1) {
            table[base + victim_way].tag = tag;
            table[base + victim_way].weight = weight;
        }
    }
    // Zero weights that disagreed with the actual outcome
    void update_weights_fast(const std::vector<uint64_t>& indices, bool taken) {
        size_t count = std::min(indices.size(), tables.size());
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t index = indices[i];
            size_t set = get_set(index);
            uint16_t tag = compute_tag(index, tag_bits[i]);
            
            auto& table = tables[i];
            size_t base = set * NUM_WAYS;
            
            for (size_t w = 0; w < NUM_WAYS; ++w) {
                if (table[base + w].tag == tag) {
                    float weight = table[base + w].weight;
                    // If taken (true), we want positive weights. If weight < 0, zero it.
                    // If not taken (false), we want negative weights. If weight > 0, zero it.
                    if ((taken && weight < 0.0f) || (!taken && weight > 0.0f)) {
                        table[base + w].weight = 0.0f;
                    }
                    break; 
                }
            }
        }
    }
};
