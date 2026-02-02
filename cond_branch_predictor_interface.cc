/*
  Copyright (C) ARM Limited 2008-2025  All rights reserved.

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// This file provides a sample predictor integration based on the interface
// provided.

#include "lib/sim_common_structs.h"
#include "my_cond_branch_predictor.h"
#include "tage/cbp2016_tage_no_sc.h"
#include "tage/cbp2016_tage_sc_l.h"
#include "tage/tagesc_full.h"
#include "tage/tagesc_mini.h"
#include <cassert>
#include <unordered_map>

// Struct to track predictions for comparison at resolve time
struct PredictionTrack {
  bool tage_64_pred;
  bool tage_64_nosc_pred;
  bool tage_192_pred;
  bool mini_tage_pred;
  bool my_pred;
};
static std::unordered_map<uint64_t, PredictionTrack> prediction_tracker;

// Stats for comparison
static uint64_t br_count = 0;
static uint64_t tage_64_misp = 0;
static uint64_t tage_64_nosc_misp = 0;
static uint64_t tage_192_misp = 0;
static uint64_t mini_tage_misp = 0;
static uint64_t my_misp = 0;

struct StatsPrinter {
  ~StatsPrinter() {
    printf("\nFINAL_MISPREDICTIONS:\n");
    printf("cbp2016_tage_sc_l: %llu\n", (unsigned long long)tage_64_misp);
    printf("cbp2016_tage_no_sc: %llu\n", (unsigned long long)tage_64_nosc_misp);
    printf("cond_predictor_impl: %llu\n", (unsigned long long)my_misp);
    printf("minitage: %llu\n", (unsigned long long)mini_tage_misp);
    printf("tage_192: %llu\n", (unsigned long long)tage_192_misp);
  }
};
static StatsPrinter stats_printer;

// Helper for unique ID
static uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) {
  assert(piece < 16);
  return (seq_no << 4) | (piece & 0x000F);
}

static tage_192::CBP2025 tage_192_impl;

//
// beginCondDirPredictor()
//
// This function is called by the simulator before the start of simulation.
// It can be used for arbitrary initialization steps for the contestant's code.
//
static uint64_t last_cycle = 0;

void beginCondDirPredictor(int argc, char **argv) {
  std::string config_path;
  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == "-c" && i + 1 < argc) {
      config_path = argv[i + 1];
      break;
    }
  }

  // setup sample_predictor
  cond_predictor_impl.setup(config_path);
  mini_tage::cbp2025.setup();
  tage64_no_sc::cbp2016_tage_sc_l.setup();
  tage64_nosc::cbp2016_tage_nosc_l.setup();
  tage_192_impl.setup();
}

//
// notify_instr_fetch(uint64_t seq_no, uint8_t piece, uint64_t pc, const
// uint64_t fetch_cycle)
//
// This function is called when any instructions(not just branches) gets
// fetched. Along with the unique identifying ids(seq_no, piece), PC of the
// instruction and fetch_cycle are also provided as inputs
//
void notify_instr_fetch(uint64_t seq_no, uint8_t piece, uint64_t pc,
                        const uint64_t fetch_cycle) {
  if (fetch_cycle > last_cycle) {
    for (uint64_t c = last_cycle + 1; c <= fetch_cycle; ++c) {
      cond_predictor_impl.timestep(c);
    }
    last_cycle = fetch_cycle;
  }
}

//
// get_cond_dir_prediction(uint64_t seq_no, uint8_t piece, uint64_t pc, const
// uint64_t pred_cycle)
//
// This function is called by the simulator for predicting conditional branches.
// input values are unique identifying ids(seq_no, piece) and PC of the branch.
// return value is the predicted direction.
//
bool get_cond_dir_prediction(uint64_t seq_no, uint8_t piece, uint64_t pc,
                             const uint64_t pred_cycle) {
  const TagePrediction mini_pred =
      mini_tage::cbp2025.predict(seq_no, piece, pc);
  const bool tage_192_pred = tage_192_impl.predict(seq_no, piece, pc);
  const bool tage64_pred =
      tage64_no_sc::cbp2016_tage_sc_l.predict(seq_no, piece, pc);
  const bool tage64_nosc_pred =
      tage64_nosc::cbp2016_tage_nosc_l.predict(seq_no, piece, pc).prediction;

  const bool my_prediction =
      cond_predictor_impl.predict(seq_no, piece, pc, mini_pred);

  // Store predictions for later comparison
  PredictionTrack track;
  track.tage_64_pred = tage64_pred;
  track.tage_64_nosc_pred = tage64_nosc_pred;
  track.tage_192_pred = tage_192_pred;
  track.mini_tage_pred = mini_pred.prediction;
  track.my_pred = my_prediction;
  prediction_tracker[get_unique_inst_id(seq_no, piece)] = track;

  return my_prediction;
}

//
// spec_update(uint64_t seq_no, uint8_t piece, uint64_t pc, InstClass
// inst_class, const bool resolve_dir, const bool pred_dir, const uint64_t
// next_pc)
//
// This function is called by the simulator for updating the history vectors and
// any state that needs to be updated speculatively. The function is called for
// all the branches (not just conditional branches). To faciliate accurate
// history updates, spec_update is called right after a prediction is made.
// input values are unique identifying ids(seq_no, piece), PC of the
// instruction, instruction class, predicted/resolve direction and the next_pc
//
void spec_update(uint64_t seq_no, uint8_t piece, uint64_t pc,
                 InstClass inst_class, const bool resolve_dir,
                 const bool pred_dir, const uint64_t next_pc) {
  assert(is_br(inst_class));
  int br_type = 0;
  switch (inst_class) {
  case InstClass::condBranchInstClass:
    br_type = 1;
    break;
  case InstClass::uncondDirectBranchInstClass:
    br_type = 0;
    break;
  case InstClass::uncondIndirectBranchInstClass:
    br_type = 2;
    break;
  case InstClass::callDirectInstClass:
    br_type = 0;
    break;
  case InstClass::callIndirectInstClass:
    br_type = 2;
    break;
  case InstClass::ReturnInstClass:
    br_type = 2;
    break;
  default:
    assert(false);
  }

  if (inst_class == InstClass::condBranchInstClass) {
    tage64_no_sc::cbp2016_tage_sc_l.history_update(
        seq_no, piece, pc, br_type, pred_dir, resolve_dir, next_pc);
    tage64_nosc::cbp2016_tage_nosc_l.history_update(
        seq_no, piece, pc, br_type, pred_dir, resolve_dir, next_pc);
    cond_predictor_impl.history_update(seq_no, piece, pc, resolve_dir, next_pc);
    mini_tage::cbp2025.history_update(seq_no, piece, pc, br_type, resolve_dir,
                                      next_pc);
    tage_192_impl.history_update(seq_no, piece, pc, br_type, resolve_dir,
                                 next_pc);
  } else {
    tage64_no_sc::cbp2016_tage_sc_l.TrackOtherInst(pc, br_type, pred_dir,
                                                   resolve_dir, next_pc);
    tage64_nosc::cbp2016_tage_nosc_l.TrackOtherInst(pc, br_type, pred_dir,
                                                    resolve_dir, next_pc);
    mini_tage::cbp2025.TrackOtherInst(pc, br_type, resolve_dir, next_pc);
    tage_192_impl.TrackOtherInst(pc, br_type, resolve_dir, next_pc);
  }
}

//
// notify_instr_decode(uint64_t seq_no, uint8_t piece, uint64_t pc, const
// DecodeInfo& _decode_info, const uint64_t decode_cycle)
//
// This function is called when any instructions(not just branches) gets
// decoded. Along with the unique identifying ids(seq_no, piece), PC of the
// instruction, decode info and cycle are also provided as inputs
//
// For the sample predictor implementation, we do not leverage decode
// information
void notify_instr_decode(uint64_t seq_no, uint8_t piece, uint64_t pc,
                         const DecodeInfo &_decode_info,
                         const uint64_t decode_cycle) {}

//
// notify_agen_complete(uint64_t seq_no, uint8_t piece, uint64_t pc, const
// DecodeInfo& _decode_info, const uint64_t mem_va, const uint64_t mem_sz, const
// uint64_t agen_cycle)
//
// This function is called when any load/store instructions complete agen.
// Along with the unique identifying ids(seq_no, piece), PC of the instruction,
// decode info, mem_va and mem_sz and agen_cycle are also provided as inputs
//
void notify_agen_complete(uint64_t seq_no, uint8_t piece, uint64_t pc,
                          const DecodeInfo &_decode_info, const uint64_t mem_va,
                          const uint64_t mem_sz, const uint64_t agen_cycle) {}

//
// notify_instr_execute_resolve(uint64_t seq_no, uint8_t piece, uint64_t pc,
// const bool pred_dir, const ExecuteInfo& _exec_info, const uint64_t
// execute_cycle)
//
// This function is called when any instructions(not just branches) gets
// executed. Along with the unique identifying ids(seq_no, piece), PC of the
// instruction, execute info and cycle are also provided as inputs
//
// For conditional branches, we use this information to update the predictor.
// At the moment, we do not consider updating any other structure, but the
// contestants are allowed to  update any other predictor state.
void notify_instr_execute_resolve(uint64_t seq_no, uint8_t piece, uint64_t pc,
                                  const bool pred_dir,
                                  const ExecuteInfo &_exec_info,
                                  const uint64_t execute_cycle) {
  const bool is_branch = is_br(_exec_info.dec_info.insn_class);
  if (is_branch) {
    if (is_cond_br(_exec_info.dec_info.insn_class)) {
      const bool _resolve_dir = _exec_info.taken.value();
      const uint64_t _next_pc = _exec_info.next_pc;

      // Check tracking
      uint64_t id = get_unique_inst_id(seq_no, piece);
      auto it = prediction_tracker.find(id);
      assert(it != prediction_tracker.end());

      tage64_no_sc::cbp2016_tage_sc_l.update(seq_no, piece, pc, _resolve_dir,
                                             pred_dir, _next_pc);
      tage64_nosc::cbp2016_tage_nosc_l.update(seq_no, piece, pc, _resolve_dir,
                                              pred_dir, _next_pc);
      cond_predictor_impl.update(seq_no, piece, pc, _resolve_dir, pred_dir,
                                 _next_pc, execute_cycle);
      mini_tage::cbp2025.update(seq_no, piece, pc, _resolve_dir, pred_dir,
                                _next_pc, (it->second.my_pred != _resolve_dir));
      tage_192_impl.update(seq_no, piece, pc, _resolve_dir, pred_dir, _next_pc);

      br_count++;
      if (it->second.tage_64_pred != _resolve_dir)
        tage_64_misp++;
      if (it->second.tage_64_nosc_pred != _resolve_dir)
        tage_64_nosc_misp++;
      if (it->second.tage_192_pred != _resolve_dir)
        tage_192_misp++;
      if (it->second.mini_tage_pred != _resolve_dir)
        mini_tage_misp++;
      if (it->second.my_pred != _resolve_dir)
        my_misp++;

      if ((br_count % 100000ULL) == 0) {
        fprintf(stderr,
                "[COND] branches=%llu miniTAGE_misp=%llu MY_misp=%llu "
                "64_nosc_misp=%llu 64_tage_misp=%llu 192_tage_misp=%llu\n",
                (unsigned long long)br_count,
                (unsigned long long)mini_tage_misp, (unsigned long long)my_misp,
                (unsigned long long)tage_64_nosc_misp,
                (unsigned long long)tage_64_misp,
                (unsigned long long)tage_192_misp);
      }
      prediction_tracker.erase(it);

    } else {
      assert(pred_dir);
    }
  }
}

//
// notify_instr_commit(uint64_t seq_no, uint8_t piece, uint64_t pc, const bool
// pred_dir, const ExecuteInfo& _exec_info, const uint64_t commit_cycle)
//
// This function is called when any instructions(not just branches) gets
// committed. Along with the unique identifying ids(seq_no, piece), PC of the
// instruction, execute info and cycle are also provided as inputs
//
// For the sample predictor implementation, we do not leverage commit
// information
void notify_instr_commit(uint64_t seq_no, uint8_t piece, uint64_t pc,
                         const bool pred_dir, const ExecuteInfo &_exec_info,
                         const uint64_t commit_cycle) {
  bool resolve_dir = false;
  if (_exec_info.taken.has_value())
    resolve_dir = _exec_info.taken.value();
  cond_predictor_impl.commit(seq_no, piece, pc, pred_dir, resolve_dir);
}

//
// endCondDirPredictor()
//
// This function is called by the simulator at the end of simulation.
// It can be used by the contestant to print out other contestant-specific
// measurements.
//
void endCondDirPredictor() {
  tage64_no_sc::cbp2016_tage_sc_l.terminate();
  tage64_nosc::cbp2016_tage_nosc_l.terminate();
  cond_predictor_impl.terminate();
  mini_tage::cbp2025.terminate();
  tage_192_impl.terminate();
}

void print_my_stats() { cond_predictor_impl.print_performance(); }
