#pragma once

#include "feature.h"
#include "utils/rolling_history.h"
#include "utils/hash.h"
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cassert>

// ============================================================================
// Simple PC-based Feature
// ============================================================================
class PCFeature : public Feature {
public:
    explicit PCFeature(const std::vector<std::string>& args) : Feature(args) {}
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class; (void)pc; (void)actual_outcome;
    }
    
    uint64_t get_index(uint64_t pc) const override {
        return pc;
    }
};

// ============================================================================
// Global Direction History Feature
// ============================================================================
class GHistFeature : public Feature {
private:
    RollingHistory<5> history;
    size_t hist_len;
    
    uint8_t compute_5bit_pattern(uint64_t pc, bool taken) const {
        uint64_t pc_s = pc >> 2;
        uint64_t mix = pc_s ^ (pc_s >> 2) ^ (pc_s >> 4) ^ (taken ? 1 : 0);
        return mix & 0x1F;
    }

public:
    explicit GHistFeature(const std::vector<std::string>& args) : Feature(args) {
        if (args.size() < 1) throw std::runtime_error("GHIST requires: hist_len");
        hist_len = std::stoull(args[0]);
        history.require(hist_len);
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class;
        uint8_t pattern = compute_5bit_pattern(pc, actual_outcome);
        history.update(pattern);
    }
    
    uint64_t get_index(uint64_t pc) const override {
        return pc ^ history.get_hash(hist_len);
    }
};

// ============================================================================
// Global Path History Feature
// ============================================================================
class GPathFeature : public Feature {
private:
    RollingHistory<8> history;
    size_t entries;
    size_t bits_per_pc;

public:
    explicit GPathFeature(const std::vector<std::string>& args) : Feature(args), bits_per_pc(8) {
        if (args.size() < 1) throw std::runtime_error("GPATH requires: entries");
        entries = std::stoull(args[0]);
        if (entries > 0) history.require(entries);
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class; (void)actual_outcome;
        uint64_t hashed = hash_64(pc);
        uint8_t pattern = hashed & ((1 << bits_per_pc) - 1);
        history.update(pattern);
    }
    
    uint64_t get_index(uint64_t pc) const override {
        uint64_t h = pc;
        if (entries > 0) h ^= history.get_hash(entries);
        return h;
    }
};

// ============================================================================
// Local History Feature (per-PC history table)
// ============================================================================
class LHistFeature : public Feature {
private:
    struct LocalHist {
        uint64_t bits;
        size_t width;
        
        LocalHist(size_t w) : bits(0), width(w) {}
        
        void update(bool outcome) {
            bits = ((bits << 1) | (outcome ? 1 : 0)) & ((1ULL << width) - 1);
        }
        
        uint64_t hash() const {
            return hash_64(bits);
        }
    };
    
    std::vector<LocalHist> histories;
    size_t num_entries;
    size_t hist_len;

public:
    explicit LHistFeature(const std::vector<std::string>& args) : Feature(args), num_entries(1024) {
        if (args.size() < 1) throw std::runtime_error("LHIST requires: hist_len");
        hist_len = std::stoull(args[0]);
        
        histories.reserve(num_entries);
        for (size_t i = 0; i < num_entries; ++i) {
            histories.emplace_back(hist_len);
        }
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class;
        size_t idx = hash_64(pc) % num_entries;
        histories[idx].update(actual_outcome);
    }
    
    uint64_t get_index(uint64_t pc) const override {
        size_t idx = hash_64(pc) % num_entries;
        return pc ^ histories[idx].hash();
    }
};

// ============================================================================
// IMLI (Inner-Most Loop Iteration) Feature
// ============================================================================
class IMLIFeature : public Feature {
private:
    uint64_t counter;
    bool is_backward;

public:
    explicit IMLIFeature(const std::vector<std::string>& args) 
        : Feature(args), counter(0), is_backward(true) {
        if (args.size() >= 1) {
            std::string dir = args[0];
            is_backward = (dir == "backward" || dir == "bwd");
        }
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class;
        
        bool backward = (pc & 0x80000000) != 0;
        
        if (is_backward) {
            if (backward) {
                counter = actual_outcome ? (counter + 1) : 0;
            }
        } else {
            if (!backward) {
                counter = actual_outcome ? 0 : (counter + 1);
            }
        }
    }
    
    uint64_t get_index(uint64_t pc) const override {
        return pc ^ counter;
    }
};

// ============================================================================
// Recency Feature (LRU stack of recent branch PCs)
// ============================================================================
class RecencyFeature : public Feature {
private:
    std::vector<uint64_t> stack;
    size_t stack_depth;
    size_t addr_shift;

public:
    explicit RecencyFeature(const std::vector<std::string>& args) : Feature(args) {
        if (args.size() < 2) throw std::runtime_error("RECENCY requires: depth addr_shift");
        stack_depth = std::stoull(args[0]);
        addr_shift = std::stoull(args[1]);
        stack.resize(stack_depth, 0);
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class; (void)actual_outcome;
        
        for (size_t i = 0; i < stack_depth; ++i) {
            if (stack[i] == pc) {
                for (size_t j = i; j > 0; --j) {
                    stack[j] = stack[j - 1];
                }
                stack[0] = pc;
                return;
            }
        }
        
        for (size_t i = stack_depth - 1; i > 0; --i) {
            stack[i] = stack[i - 1];
        }
        stack[0] = pc;
    }
    
    uint64_t get_index(uint64_t pc) const override {
        uint64_t h = pc;
        for (size_t i = 0; i < stack_depth; ++i) {
            h = (h << addr_shift) ^ hash_64(stack[i]);
        }
        return h;
    }
};

// ============================================================================
// Recency Position Feature (position of PC in recency stack)
// ============================================================================
class RecencyPosFeature : public Feature {
private:
    std::vector<uint64_t> stack;
    size_t search_depth;

public:
    explicit RecencyPosFeature(const std::vector<std::string>& args) : Feature(args) {
        if (args.size() < 1) throw std::runtime_error("RECENCY_POS requires: depth");
        search_depth = std::stoull(args[0]);
        stack.resize(search_depth, 0);
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class; (void)actual_outcome;
        
        for (size_t i = 0; i < search_depth; ++i) {
            if (stack[i] == pc) {
                for (size_t j = i; j > 0; --j) {
                    stack[j] = stack[j - 1];
                }
                stack[0] = pc;
                return;
            }
        }
        
        for (size_t i = search_depth - 1; i > 0; --i) {
            stack[i] = stack[i - 1];
        }
        stack[0] = pc;
    }
    
    uint64_t get_index(uint64_t pc) const override {
        // Find position of PC in stack
        size_t pos = search_depth;
        for (size_t i = 0; i < search_depth; ++i) {
            if (stack[i] == pc) {
                pos = i;
                break;
            }
        }
        return pc ^ pos;
    }
};

// ============================================================================
// Blurry Path Feature (region-based coarse-grained path tracking)
// ============================================================================
class BlurryPathFeature : public Feature {
private:
    RollingHistory<8> history;
    size_t num_regions;
    size_t shift_amount;
    uint64_t current_region;

public:
    explicit BlurryPathFeature(const std::vector<std::string>& args) 
        : Feature(args), shift_amount(8), current_region(0) {
        if (args.size() < 1) throw std::runtime_error("BLURRY_PATH requires: num_regions");
        num_regions = std::stoull(args[0]);
        if (args.size() >= 2) {
            shift_amount = std::stoull(args[1]);
        }
        history.require(num_regions);
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class; (void)actual_outcome;
        
        uint64_t region = hash_64(pc >> shift_amount) & 0xFF;
        if (region != current_region) {
            history.update(static_cast<uint8_t>(current_region));
            current_region = region;
        }
    }
    
    uint64_t get_index(uint64_t pc) const override {
        return pc ^ history.get_hash(num_regions);
    }
};

// ============================================================================
// Return Stack History Feature (cyclic stack of direction histories)
// ============================================================================
class ReturnStackHistFeature : public Feature {
private:
    struct StackEntry {
        uint64_t bits;
        size_t width;
        
        StackEntry(size_t w) : bits(0), width(w) {}
        
        void update(bool outcome) {
            bits = ((bits << 1) | (outcome ? 1 : 0)) & ((1ULL << width) - 1);
        }
        
        uint64_t hash() const {
            return hash_64(bits);
        }
    };
    
    std::vector<StackEntry> stack;
    size_t stack_depth;
    size_t hist_len;
    size_t head;

public:
    explicit ReturnStackHistFeature(const std::vector<std::string>& args) 
        : Feature(args), stack_depth(8), head(0) {
        if (args.size() < 1) throw std::runtime_error("RET_STACK requires: hist_len");
        hist_len = std::stoull(args[0]);
        
        stack.reserve(stack_depth);
        for (size_t i = 0; i < stack_depth; ++i) {
            stack.emplace_back(hist_len);
        }
    }
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)pc;
        
        stack[head].update(actual_outcome);
        
        if (inst_class == InstClass::callDirectInstClass || 
            inst_class == InstClass::callIndirectInstClass) {
            size_t new_head = (head + 1) % stack_depth;
            stack[new_head] = stack[head];
            head = new_head;
        } else if (inst_class == InstClass::ReturnInstClass) {
            head = (head == 0) ? (stack_depth - 1) : (head - 1);
        }
    }
    
    uint64_t get_index(uint64_t pc) const override {
        return pc ^ stack[head].hash();
    }
};

// ============================================================================
// BrIMLI Feature (Branch Immediate Loop Iteration)
// ============================================================================
class BrIMLIFeature : public Feature {
private:
    uint64_t counter;
    uint64_t current_region;

public:
    explicit BrIMLIFeature(const std::vector<std::string>& args) 
        : Feature(args), counter(0), current_region(0) {}
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class;
        
        bool backward = (pc & 0x80000000) != 0;
        
        if (backward && actual_outcome) {
            uint64_t region = pc >> 8;
            if (region == current_region) {
                counter++;
            } else {
                current_region = region;
                counter = 0;
            }
        }
    }
    
    uint64_t get_index(uint64_t pc) const override {
        return pc ^ counter;
    }
};

// ============================================================================
// TaIMLI Feature (Target Inner-Most Loop Iteration)
// Similar to BrIMLI but uses target region (simplified without target)
// ============================================================================
class TaIMLIFeature : public Feature {
private:
    uint64_t counter;
    uint64_t current_region;

public:
    explicit TaIMLIFeature(const std::vector<std::string>& args) 
        : Feature(args), counter(0), current_region(0) {}
    
    void update(InstClass inst_class, uint64_t pc, 
                bool actual_outcome) override {
        (void)inst_class;
        
        bool backward = (pc & 0x80000000) != 0;
        
        if (backward && actual_outcome) {
            uint64_t region = pc >> 8;
            if (region == current_region) {
                counter++;
            } else {
                current_region = region;
                counter = 0;
            }
        }
    }
    
    uint64_t get_index(uint64_t pc) const override {
        return pc ^ counter;
    }
};
