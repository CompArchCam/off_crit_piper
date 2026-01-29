#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <cassert>
#include "hash.h"

template<size_t PATTERN_SIZE>
class RollingHistory {
    static_assert(PATTERN_SIZE >= 1 && PATTERN_SIZE <= 8, "PATTERN_SIZE must be between 1 and 8");
    
private:
    std::vector<uint8_t> hist;
    std::vector<uint64_t> hashes;
    std::vector<size_t> lens;
    size_t head = 0;
    size_t max_len = 0;

public:
    RollingHistory() = default;

    void update(uint8_t new_val) {
        // history at head is being overwritten.
        // But for each hash of length L, the value leaving the window is at (head - L).
        // The value at head was inserted max_len ago (if wrapped).
        
        for (size_t i = 0; i < hashes.size(); ++i) {
            size_t len = lens[i];
            if (len == 0) continue;
            
            // Current head points to the slot we are about to overwrite.
            // But we need the value that was inserted 'len' steps ago.
            // That value is at: (head + size - len) % size
            
            // Note: If hist is not full/circular yet, we might be reading 0s or doing logic that assumes 0s.
            // Since we resize and zero-init, this is fine.
            if (hist.empty()) continue;

            size_t old_idx = (head + hist.size() - len) % hist.size();
            uint8_t old_val = hist[old_idx];
            
            // Offset corresponding to the oldest element which was rotated (len-1) times.
            // We want to cancel Rot(old_val, 5*(len-1)). 
            // Since we XOR this BEFORE the next rotation by 5, the effective cancellation is Rot(Rot(old_val, 5*(len-1)), 5) = Rot(old_val, 5*len).
            // Wait, no. We cancel `Rot(old_val, 5*(len-1))` from the current hash.
            // Then we rotate the result by 5. 
            // The cancelled term becomes `Rot(Rot(old_val, 5*(len-1)), 5) = Rot(old_val, 5*len)`.
            // The term for old_val in the new hash (if we didn't cancel) would be `Rot(val[t-len], 5*len)`.
            // So this cancellation is correct.
            
            uint64_t offset = (PATTERN_SIZE * len + 64 - PATTERN_SIZE) % 64;
            uint64_t old_shifted = (static_cast<uint64_t>(old_val) << offset) | (static_cast<uint64_t>(old_val) >> (64 - offset));
            
            hashes[i] ^= old_shifted;
            
            // Left rotate by PATTERN_SIZE
            hashes[i] = (hashes[i] << PATTERN_SIZE) | (hashes[i] >> (64 - PATTERN_SIZE));
            
            // Shift in new value
            hashes[i] ^= static_cast<uint64_t>(new_val);
        }
        
        if (!hist.empty()) {
            hist[head] = new_val;
            head++;
            if (head == hist.size()) head = 0;
        }
    }

    void require(size_t needed_len) {
        lens.push_back(needed_len);
        hashes.push_back(0);
        
        if (needed_len > max_len) {
             // Resize history buffer
             std::vector<uint8_t> new_hist(needed_len + 1, 0); // +1 safety/sentinel
             
             if (!hist.empty()) {
                 size_t count = std::min(hist.size(), needed_len);
                 size_t start = (head + hist.size() - count) % hist.size();
                 for (size_t i = 0; i < count; ++i) {
                     new_hist[i] = hist[(start + i) % hist.size()];
                 }
                 head = count;
             } else {
                 head = 0;
             }
             
             hist = std::move(new_hist);
             max_len = needed_len;
        }
    }

    uint64_t get_hash(size_t n) const {
        for (size_t i = 0; i < lens.size(); ++i) {
            if (lens[i] == n) return hash_64(hashes[i]);
        }
        std::cout << "Error: Length " << n << " not found in rolling history." << std::endl;
        std::cout << "Histories: ";
        for (size_t i = 0; i < lens.size(); ++i) {
            std::cout << lens[i] << " ";
        }
        assert(false);
        return 0;
    }

    size_t num_tracked() const {
        return lens.size();
    }
    
    size_t get_max_len() const {
        return max_len;
    }
    
    size_t size_bytes() const {
        return hist.size() * sizeof(uint8_t) + hashes.size() * sizeof(uint64_t) + lens.size() * sizeof(size_t);
    }
};
