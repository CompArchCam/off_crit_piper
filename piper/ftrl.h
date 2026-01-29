#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <deque>

struct ActiveWeight {
    uint64_t hash;  // The raw feature index/hash provided by GetIndex
    float weight;
    size_t feature_idx;
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
    
    float alpha;
    float beta;
    float l1;
    float l2;
    
    float compute_weight_internal(float z, float n) const {
        if (std::abs(z) <= l1) return 0.0f;
        return -(z - (z > 0 ? 1.0f : -1.0f) * l1) / (((beta + std::sqrt(n)) / alpha) + l2);
    }

public:
    FTRL(size_t num_features, float _alpha, float _beta, float _l1, float _l2)
        : alpha(_alpha), beta(_beta), l1(_l1), l2(_l2)
    {
        if (alpha <= 0.0f) throw std::runtime_error("FTRL alpha must be > 0");
        tables.reserve(num_features);
        // Hardcode size to 2^14
        size_t fixed_size = 1 << 14; 
        for (size_t i = 0; i < num_features; ++i) {
            tables.emplace_back(fixed_size);
        }
    }
    
    // Update weights based on indices provided by IndexManager
    void update(const std::vector<uint64_t>& indices, float pred, bool actual) {
        float g = pred - (actual ? 1.0f : 0.0f);
        float g2 = g * g;
        
        size_t count = std::min(indices.size(), tables.size());
        
        for (size_t i = 0; i < count; ++i) {
            uint64_t raw_idx = indices[i];
            auto& table = tables[i];
            
            // Map raw index/hash to table size using simple modulo
            // Table size is fixed 1<<14, so mask is (1<<14)-1 = 0x3FFF
            size_t idx = raw_idx & 0x3FFF;
            
            float zi = table.z_table[idx];
            float ni = table.n_table[idx];
            
            float weight = compute_weight_internal(zi, ni);
            
            float si = (std::sqrt(ni + g2) - std::sqrt(ni)) / alpha;
            table.z_table[idx] += g - si * weight;
            table.n_table[idx] += g2;
            
            float new_weight = compute_weight_internal(table.z_table[idx], table.n_table[idx]);
            
            // Push active weight update (including zero, for FSC clearing)
            nonzero_weights.push_back({raw_idx, new_weight, i});
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

    void clear_nonzero_queue() {
        nonzero_weights.clear();
    }
    
    size_t size_bytes() const {
        size_t total = 0;
        for (const auto& t : tables) {
            total += t.z_table.size() * sizeof(float);
            total += t.n_table.size() * sizeof(float);
        }
        return total;
    }
};
