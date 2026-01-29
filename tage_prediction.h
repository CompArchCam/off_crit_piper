#pragma once

struct TagePrediction {
    bool prediction;
    int confidence;
    bool hit_in_last_history_table;

    // Default constructor for safety, though explicit initialization is better
    TagePrediction() : prediction(false), confidence(0), hit_in_last_history_table(false) {}

    // Constructor to enforce setting all fields
    TagePrediction(bool pred, int conf, bool hit) 
        : prediction(pred), confidence(conf), hit_in_last_history_table(hit) {}
};
