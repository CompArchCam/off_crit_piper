

// Developped by A. Seznec
// This simulator is derived from the simulator provided with  the slide set
// "TAGE: an engineering cookbook by Andre Seznec, November 2024" and the
// CBP2016 winner as coded in the CBP2025 framework

// The two  initial simulators were  developped to fit CBP2016 framework
// In this file : added Local Prediction + Different threshold
// Added global interleaving of logical TAGE tables

// The  loop predictor is present in the file, but is not enabled
// It brings only marginal accuracy benefit

#ifndef MINITAGE___TAGE_PREDICTOR_H_
#define MINITAGE___TAGE_PREDICTOR_H_
#include <array>
#include <assert.h>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

#include "../piper/utils/hash.h"
#include "tage_prediction.h"
#include <vector>

namespace mini_tage {
// To get the predictor storage budget on stderr  uncomment the next line
#define MINITAGE__PRINTSIZE

#define MINITAGE__LOGSCALE 2
#define MINITAGE__LOGT                                                         \
  (7 + MINITAGE__LOGSCALE) /* logsize of a logical  TAGE tables */
#define MINITAGE__LOGB                                                         \
  (11 + MINITAGE__LOGSCALE) // log of number of entries in bimodal predictor

#define MINITAGE__MINHIST 5
#define MINITAGE__MAXHIST 960
// Did not explore exactly which history length could be the best one, but it
// seems that 1000 is not bad :-)

// parameters of the loop predictor
#define MINITAGE__LOGL 5
#define MINITAGE__WIDTHNBITERLOOP                                              \
  10 // we predict only loops with less than 1K iterations
#define MINITAGE__LOOPTAG 10 // tag width in the loop predictor

#define MINITAGE__NHIST 14
#define MINITAGE__NBANK 7
#define MINITAGE__TBITS 12

#define MINITAGE__UWIDTH 2
#define MINITAGE__LOGASSOC                                                     \
  1 // but partial skewed associativity (option MINITAGE__PSK) might be
    // interesting

#define MINITAGE__LOGG                                                         \
  (MINITAGE__LOGT - MINITAGE__LOGASSOC) // size of way in a logical TAGE table
#define MINITAGE__ASSOC (1 << MINITAGE__LOGASSOC)

#define MINITAGE__HYSTSHIFT                                                    \
  2 // bimodal hysteresis shared among (1<< MINITAGE__HYSTSHIFT) entries
#define MINITAGE__BIMWIDTH 2 //  with of the counter in the bimodal predictor
// A. Seznec: I just played using 3-bit counters in the simulator, using 2-bit
// counters but MINITAGE__HYSTSHIFT=0 brings similar accuracy

/////////////////////////////////////////////
// Options  for optimizations of TAGE

int BANK1;
// we implement a banked interleaved predictor: all the logical tagged tables
// share the whole space

/////////////////////////////////////////////////
// the replacement/allocation policies described in the slide set
#define MINITAGE__OPTTAGE
#ifdef MINITAGE__OPTTAGE
#define MINITAGE__OPTGEOHIST // we can do better than geometric series
#define MINITAGE__FILTERALLOCATION
#define MINITAGE__FORCEU 1 // don't work if only one U  bit

#if (MINITAGE__LOGASSOC == 1)
// A. Seznec: partial skewed associativity, remember that I invented it in 1993
// :-)
#define MINITAGE__PSK 1
#define MINITAGE__REPSK                                                        \
  1 // this optimization is funny, if no "useless" entry, move the entry on the
    // other way to make room,it brings a little bit of accuracy
// for caches it was published as the Elbow cache in 2003, or the Zcache in
// 2010.
#else
// A. Seznec:  I do not have implemented skewed associativity for more than
// two-way
#define MINITAGE__PSK 0
#define MINITAGE__REPSK 0
#endif
#define MINITAGE__PROTECTRECENTALLOCUSEFUL                                     \
  1 // Recently allocated entries  are protected against the smart u reset:
#define MINITAGE__UPDATEALTONWEAKMISP                                          \
  1 // When the Longest match is weak and wrong, one updates also the alternate
    // prediction and HCPred :

// #define MINITAGE__BIM_HIST // incorporate global history into bimodal index
// static const int MINITAGE__BIM_HIST_BITS = 5; // max 8
#else

#define MINITAGE__PSK 0
#define MINITAGE__REPSK 0
#endif
//////////////////////////////////////////////

// marginal benefit

////  FOR TAGE //////

#define MINITAGE__HISTBUFFERLENGTH                                             \
  8192 // we use a 8K entries history buffer to store the branch history: 5000
       // are needed in the simulator + 5 * the number of inflight branches
uint8_t ghist[MINITAGE__HISTBUFFERLENGTH];

#define MINITAGE__BORNTICK 4096
// for the allocation policy
//  utility class for index computation
//  this is the cyclic shift register for folding
//  a long global history into a smaller number of bits; see P. Michaud's
//  PPM-like predictor at CBP-1

class folded_history {
public:
  unsigned comp;
  int CLENGTH;
  int OLENGTH;
  int OUTPOINT;
  int INTEROUT;

  folded_history() {}

  void init(int original_length, int compressed_length, int N) {
    comp = 0;
    OLENGTH = original_length;
    CLENGTH = compressed_length;
    OUTPOINT = OLENGTH % CLENGTH;
  }

  void update(uint8_t *h, int PT) {
    comp = (comp << 1) ^ h[PT & (MINITAGE__HISTBUFFERLENGTH - 1)];

    comp ^= h[(PT + OLENGTH) & (MINITAGE__HISTBUFFERLENGTH - 1)] << OUTPOINT;
    comp ^= (comp >> CLENGTH);
    comp = (comp) & ((1 << CLENGTH) - 1);
  }
};

class bentry // TAGE bimodal table entry
{
public:
  int8_t hyst;
  int8_t pred;
  bentry() {
    pred = 0;
    hyst = 1;
  }
};

class gentry // TAGE global table entry
{
public:
  int8_t ctr;
  uint tag;
  int8_t u;

  gentry() {
    ctr = 0;
    u = 0;
    tag = 0;
  }
};

bool alttaken; // alternate   TAGE prediction if the longest match was not
               // hitting: needed for updating the u bit
bool HCpred;   // longest not low confident match or base prediction if no
               // confident match

bool tage_pred; // TAGE prediction
bool LongestMatchPred;
int HitBank;    // longest matching bank
int AltBank;    // alternate matching bank
int HCpredBank; // longest non weak  matching bank
int HitAssoc;
int AltAssoc;
int HCpredAssoc;
int Seed;   // for the pseudo-random number generator
int8_t BIM; // the bimodal prediction

// Counters to guide allocation/replacement in TAGE
int8_t CountMiss11 = -64; // more or less than 11% of misspredictions

#define MINITAGE__LOGCOUNT 6
#define MINITAGE__INDCOUNT                                                     \
  ((PCBR ^ (PCBR >> MINITAGE__LOGCOUNT)) & ((1 << MINITAGE__LOGCOUNT) - 1))
int8_t COUNT50[(1 << MINITAGE__LOGCOUNT)][(MINITAGE__NHIST / 4) + 1] = {
    {7}}; // more or less than 50%  misprediction on weak LongestMatchPred
int8_t COUNT16_31[(1 << MINITAGE__LOGCOUNT)][(MINITAGE__NHIST / 4) + 1] = {
    {7}}; // more or less than 16/31th  misprediction on weak LongestMatchPred
int8_t COUNT8_15[(1 << MINITAGE__LOGCOUNT)][(MINITAGE__NHIST / 4) + 1] = {{7}};
int8_t Count50[(MINITAGE__NHIST / 4) + 1] = {7};
int8_t Count16_31[(MINITAGE__NHIST / 4) + 1] = {7};
int8_t Count8_15[(MINITAGE__NHIST / 4) + 1] = {7};

int TAGECONF; // TAGE confidence  from 0 (weak counter) to 3 (saturated)
int ALTCONF;
int HCCONF;
int BIMCONF;
bool BIMPRED;
#define MINITAGE__PHISTWIDTH 27 // width of the path history used in TAGE
#define MINITAGE__CWIDTH 3 // predictor counter width on the TAGE tagged tables

// the counter(s) to chose between longest match and alternate prediction on
// TAGE when weak counters: only plain TAGE
#define MINITAGE__ALTWIDTH 5
static const int UANA_PC_BITS = 2;     // 4-way PC buckets
static const int UANA_BANK_GROUPS = 4; // group HitBank by /4
static const int UANA_ALTCONF = 2;     // {low, high}
int8_t use_alt_on_na[UANA_BANK_GROUPS][(1 << UANA_PC_BITS)][UANA_ALTCONF];
int TICK, TICKH; // for the reset of the u counter
// TICKH  used only if (MINITAGE__UWIDTH =1)

class lentry // loop predictor entry
{
public:
  uint16_t NbIter;      // 10 bits
  uint8_t confid;       // 4bits
  uint16_t CurrentIter; // 10 bits

  uint16_t TAG; // 10 bits
  uint8_t age;  // 4 bits
  bool dir;     // 1 bit

  // 39 bits per entry
  lentry() {
    confid = 0;
    CurrentIter = 0;
    NbIter = 0;
    TAG = 0;
    age = 0;
    dir = false;
  }
};
// For the TAGE predictor
bentry *btable;                      // bimodal TAGE table
gentry *gtable[MINITAGE__NHIST + 1]; // tagged TAGE tables
int m[MINITAGE__NHIST + 1];
uint GI[MINITAGE__NHIST +
        1]; // indexes to the different tables are computed only once
uint GGI[MINITAGE__ASSOC][MINITAGE__NHIST + 1]; // indexes to the different ways
                                                // for tables  in TAGE
uint GTAG[MINITAGE__NHIST +
          1];    // tags for the different tables are computed only once
int BI;          // index of the bimodal table
bool pred_taken; // prediction
int Provider;

// COPYPASTE
using tage_index_t = std::array<folded_history, MINITAGE__NHIST + 1>;
using tage_tag_t = std::array<folded_history, MINITAGE__NHIST + 1>;

struct cbp_hist_t {
  // Begin Conventional Histories

  //      std::array<uint8_t, MINITAGE__HISTBUFFERLENGTH> ghist;No need to
  //      checkpoint that
  uint64_t phist; // path history
  int ptghist;
  tage_index_t ch_i;
  std::array<tage_tag_t, 2> ch_t;

#ifdef LOOPPREDICTOR
  std::vector<lentry> ltable;
  int8_t WITHLOOP;
#endif
  cbp_hist_t() {
#ifdef LOOPPREDICTOR
    ltable.resize(1 << (MINITAGE__LOGL));
    WITHLOOP = -1;
#endif
  }
};

int incval(int8_t ctr) {

  return (2 * ctr + 1);
  // to center the sum
}

int predictorsize() {
  int STORAGESIZE = 0;
  int inter = 0;

  STORAGESIZE += MINITAGE__NBANK * (1 << MINITAGE__LOGG) *
                 (MINITAGE__CWIDTH + MINITAGE__UWIDTH + MINITAGE__TBITS) *
                 MINITAGE__ASSOC;
  STORAGESIZE += UANA_BANK_GROUPS * (1 << UANA_PC_BITS) * UANA_ALTCONF *
                 MINITAGE__ALTWIDTH;
  // the use_alt counter
  STORAGESIZE +=
      (1 << MINITAGE__LOGB) +
      (MINITAGE__BIMWIDTH - 1) * (1 << (MINITAGE__LOGB - MINITAGE__HYSTSHIFT));
  STORAGESIZE += m[MINITAGE__NHIST];   // the history bits
  STORAGESIZE += MINITAGE__PHISTWIDTH; // phist
  STORAGESIZE += 12;                   // the TICK counter
  if (MINITAGE__UWIDTH == 1)
    STORAGESIZE += 12; // the TICKH counter

  STORAGESIZE +=
      3 * 7 *
      ((MINITAGE__NHIST / 4) * (1 << MINITAGE__LOGCOUNT) +
       1); // counters COUNT50 COUNT16_31 COUNT8_31 Count50 Count16_31 Count8_15
  STORAGESIZE += 8;  // CountMiss11
  STORAGESIZE += 36; // for the random number generator
  fprintf(stderr, " (TAGE %d) ", STORAGESIZE);
#ifdef LOOPPREDICTOR

  inter = (1 << MINITAGE__LOGL) *
          (2 * MINITAGE__WIDTHNBITERLOOP + MINITAGE__LOOPTAG + 4 + 4 + 1);
  fprintf(stderr, " (LOOP %d) ", inter);
  STORAGESIZE += inter;
#endif

#ifdef MINITAGE__PRINTSIZE

  fprintf(stderr, " (TOTAL MINI TAGE %d, %d Kbits)\n  ", STORAGESIZE,
          STORAGESIZE / 1024);
  fprintf(stdout, " (TOTAL MINI TAGE %d %d Kbits)\n  ", STORAGESIZE,
          STORAGESIZE / 1024);
#endif

  return (STORAGESIZE);
}

class CBP2025 {
public:
  int LSUM;

  cbp_hist_t active_hist; // running history always updated accurately
  std::unordered_map<uint64_t /*key*/, cbp_hist_t /*val*/> pred_time_histories;
  // Begin LOOPPREDICTOR State
  bool predloop; // loop predictor prediction
  int LIB;
  int LI;
  int LHIT;    // hitting way in the loop predictor
  int LTAG;    // tag on the loop predictor
  bool LVALID; // validity of the loop predictor prediction
  // End LOOPPREDICTOR State
  CBP2025(void) {

    reinit(active_hist);

#ifdef MINITAGE__PRINTSIZE
    predictorsize();
#endif
  }
  void setup() {}

  void terminate() {}
  uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    assert(piece < 16);
    return (seq_no << 4) | (piece & 0x000F);
  }

  void reinit(cbp_hist_t &current_hist) {

    if ((MINITAGE__LOGASSOC != 1) || (MINITAGE__PSK == 0)) {
#if (MINITAGE__REPSK == 1)

      printf("Sorry MINITAGE__REPSK only with associativity 2 and "
             "MINITAGE__PSK activated\n");
      exit(1);

#endif
    }
    LVALID = false;

#ifdef MINITAGE__OPTGEOHIST

#if (MINITAGE__NHIST == 14)
#define MINITAGE__NNHIST 14
#endif

    int mm[MINITAGE__NNHIST + 1];
    mm[1] = MINITAGE__MINHIST;

    for (int i = 2; i <= MINITAGE__NNHIST; i++) {
      mm[i] =
          (int)(((double)MINITAGE__MINHIST *
                 pow((double)(MINITAGE__MAXHIST) / (double)MINITAGE__MINHIST,
                     (double)(i - 1) / (double)((MINITAGE__NNHIST - 1)))) +
                0.5);
    }
    for (int i = 2; i <= MINITAGE__NNHIST; i++)
      if (mm[i] <= mm[i - 1] + 1)
        mm[i] = mm[i - 1] + 1;

#if (MINITAGE__NHIST == 28)
    int PT = 1;
    for (int i = 1; i <= 7; i += 2) {
      m[PT] = mm[i];
      PT++;
    }

    for (int i = 9; i <= 28; i++)

    {
      m[PT] = mm[i];
      PT++;
    }
    PT = MINITAGE__NHIST;

    for (int i = MINITAGE__NNHIST; i >= 30; i -= 2) {
      m[PT] = mm[i];
      PT--;
    }
#endif

#if (MINITAGE__NHIST == 16)
    int PT = 1;
    for (int i = 1; i <= 5; i += 2) {
      m[PT] = mm[i];
      PT++;
    }

    for (int i = 7; i < 17; i++)

    {
      m[PT] = mm[i];
      PT++;
    }
    PT = MINITAGE__NHIST;

    for (int i = MINITAGE__NNHIST; i >= 18; i -= 2) {
      m[PT] = mm[i];
      PT--;
    }
#endif

#if (MINITAGE__NHIST == 14)
    for (int i = 1; i <= MINITAGE__NHIST; i++)
      m[i] = mm[i];
#endif
#else
    m[1] = MINITAGE__MINHIST;

    for (int i = 2; i <= MINITAGE__NHIST; i++) {
      m[i] = (int)(((double)MINITAGE__MINHIST *
                    pow((double)(MINITAGE__MAXHIST) / (double)MINITAGE__MINHIST,
                        (double)(i - 1) / (double)((MINITAGE__NHIST - 1)))) +
                   0.5);
    }
    for (int i = 3; i <= MINITAGE__NHIST; i++)
      if (m[i] <= m[i - 1])
        m[i] = m[i - 1] + 1;
#endif

    for (int i = 1; i <= MINITAGE__NHIST; i++)

      m[i] *= 5;
    // 5 bits per block: 5 is prime with the size of the folded register used in
    // the predictor

    for (int i = 1; i <= MINITAGE__NHIST; i++)
      printf("%d ", m[i]);
    printf("\n");
    // Tables are interleaved on all the banks
    gtable[1] =
        new gentry[(1 << (MINITAGE__LOGG)) * MINITAGE__ASSOC * MINITAGE__NBANK];
    for (int i = 2; i <= MINITAGE__NHIST; i++)
      gtable[i] = gtable[1];

    btable = new bentry[1 << MINITAGE__LOGB];
    for (int i = 1; i <= MINITAGE__NHIST; i++) {
      current_hist.ch_i[i].init(m[i], 23, i - 1);
#if ((MINITAGE__TBITS == 13) || (MINITAGE__TBITS == 12))
      current_hist.ch_t[0][i].init(current_hist.ch_i[i].OLENGTH, 13, i);
      current_hist.ch_t[1][i].init(current_hist.ch_i[i].OLENGTH, 11, i + 2);
#endif
#if (MINITAGE__TBITS == 14)
      current_hist.ch_t[0][i].init(current_hist.ch_i[i].OLENGTH, 13, i);
      current_hist.ch_t[1][i].init(current_hist.ch_i[i].OLENGTH, 14, i + 2);
#endif
    }
    for (int i = 0; i < UANA_BANK_GROUPS; i++)
      for (int j = 0; j < (1 << UANA_PC_BITS); j++)
        for (int k = 0; k < UANA_ALTCONF; k++)
          use_alt_on_na[i][j][k] = 0;
  }

  // index function for the bimodal table

  // the index functions for the tagged tables uses path history as in the OBIAS
  // predictor
  // F serves to mix path history: not very important impact

  int F(uint64_t A, int size, int bank) {
    int A1, A2;
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << MINITAGE__LOGG) - 1));
    A2 = (A >> MINITAGE__LOGG);
    if (bank < MINITAGE__LOGG)
      A2 = ((A2 << bank) & ((1 << MINITAGE__LOGG) - 1)) ^
           (A2 >> (MINITAGE__LOGG - bank));
    A = A1 ^ A2;
    if (bank < MINITAGE__LOGG)
      A = ((A << bank) & ((1 << MINITAGE__LOGG) - 1)) ^
          (A >> (MINITAGE__LOGG - bank));
    return (A);
  }

  // gindex computes a full hash of PC, ghist and phist
  int gindex(uint64_t PC, int bank, uint64_t hist, const tage_index_t &ch_i) {
    uint index;
    int logg = MINITAGE__LOGG;
    int M = (m[bank] > MINITAGE__PHISTWIDTH) ? MINITAGE__PHISTWIDTH : m[bank];
    index = hash_64(PC) ^ ch_i[bank].comp ^ (ch_i[bank].comp >> logg) ^
            F(hist, M, bank);
    index = hash_64(index);
    uint32_t X =
        (index ^ (index >> logg) ^ (index >> 2 * logg)) & ((1 << logg) - 1);

    return (X);
  }

  //  tag computation

  uint16_t gtag(unsigned int PC, int bank, uint64_t hist, const tage_tag_t &ch0,
                const tage_tag_t &ch1)

  {
    int tag = hash_64(PC);
    int M = (m[bank] > MINITAGE__PHISTWIDTH) ? MINITAGE__PHISTWIDTH : m[bank];
    tag = (tag >> 1) ^ ((tag & 1) << 10) ^ F(hist, M, bank);
    tag ^= ch0[bank].comp ^ (ch1[bank].comp << 1);
    tag ^= tag >> MINITAGE__TBITS;
    tag ^= (tag >> (MINITAGE__TBITS - 2));

    return tag & ((1 << MINITAGE__TBITS) - 1);
  }

  // up-down saturating counter
  void ctrupdate(int8_t &ctr, bool taken, int nbits) {
    if (taken) {
      if (ctr < ((1 << (nbits - 1)) - 1))
        ctr++;
    } else {
      if (ctr > -(1 << (nbits - 1)))
        ctr--;
    }
  }

  bool getbim() {
    BIM = (btable[BI].pred) ? (btable[BI >> MINITAGE__HYSTSHIFT].hyst)
                            : -1 - (btable[BI >> MINITAGE__HYSTSHIFT].hyst);

    TAGECONF =
        (btable[BI >> MINITAGE__HYSTSHIFT].hyst); // used when OTHERTABLES

    ALTCONF = TAGECONF;
    HCCONF = TAGECONF;

    return (btable[BI].pred != 0);
  }

  void baseupdate(bool Taken) {
    int8_t inter = BIM;
    ctrupdate(inter, Taken, MINITAGE__BIMWIDTH);
    btable[BI].pred = (inter >= 0);
    btable[BI >> MINITAGE__HYSTSHIFT].hyst = (inter >= 0) ? inter : -inter - 1;
  };
  uint32_t MYRANDOM() {

    // This pseudo-random function: just to be sure that the simulator is
    // deterministic
    //  results are within +- 0.002 MPKI in average with some larger difference
    //  on individual benchmarks
    Seed++;
    Seed += active_hist.phist;
    Seed = (Seed >> 21) + (Seed << 11);
    Seed += active_hist.ptghist;
    Seed = (Seed >> 10) + (Seed << 22);
    Seed += GTAG[4];
    return (Seed);
  };

  //  TAGE PREDICTION: same code at fetch or retire time but the index and tags
  //  must recomputed
  void Tagepred(uint64_t PCBR, const cbp_hist_t &hist_to_use)
  // void Tagepred (uint64_t  PC)
  {

    HitBank = 0;
    AltBank = 0;
    HCpredBank = 0;
    Provider = 0;

    for (int i = 1; i <= MINITAGE__NHIST; i++) {
      GI[i] = gindex(PCBR, i, hist_to_use.phist, hist_to_use.ch_i);
      GTAG[i] = gtag(PCBR, i, hist_to_use.phist, hist_to_use.ch_t[0],
                     hist_to_use.ch_t[1]);
    }

    BANK1 = (hash_64(PCBR) ^ (hist_to_use.phist & ((1 << m[1]) - 1))) %
            MINITAGE__NBANK;

    for (int i = 1; i <= MINITAGE__NHIST; i++)
      GI[i] += ((BANK1 + i) % MINITAGE__NBANK) * (1 << (MINITAGE__LOGG));

    for (int i = 1; i <= MINITAGE__NHIST; i++)

      GI[i] *= MINITAGE__ASSOC;

#ifdef MINITAGE__BIM_HIST
    uint32_t gh_bits = 0;
    for (int i = 0; i < MINITAGE__BIM_HIST_BITS; i++) {
      gh_bits |= (uint32_t(ghist[(hist_to_use.ptghist + i) &
                                 (MINITAGE__HISTBUFFERLENGTH - 1)])
                  << i);
    }
    uint64_t bhash =
        hash_64(PCBR) ^ (uint64_t(gh_bits) << 2) ^ (uint64_t(gh_bits) << 7);
    BI = bhash & ((1 << MINITAGE__LOGB) - 1);
#else
    BI = hash_64(PCBR) & ((1 << MINITAGE__LOGB) - 1);
#endif
    for (int i = 1; i <= MINITAGE__NHIST; i++) {
      for (int j = 0; j < MINITAGE__ASSOC; j++)
        GGI[j][i] = GI[i];
      if (MINITAGE__PSK == 1) {
        for (int j = 1; j < MINITAGE__ASSOC; j++)
          GGI[j][i] ^= ((GTAG[i]) & 0xff) << (1);
        // skewed associative !!
      }
    }

    alttaken = getbim();
    HCpred = alttaken;
    tage_pred = alttaken;
    LongestMatchPred = alttaken;
    // Look for the bank with longest matching history
    for (int i = MINITAGE__NHIST; i > 0; i--) {
      for (int j = 0; j < MINITAGE__ASSOC; j++) {

        if (gtable[i][GGI[j][i] + j].tag == GTAG[i]) {
          HitBank = i;
          HitAssoc = j;

          LongestMatchPred =
              (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr >= 0);
          TAGECONF =
              (abs(2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr +
                   1)) >>
              1;

          break;
        }
      }
      if (HitBank > 0)
        break;
    }
    // should be noted that when LongestMatchPred is not low conf then alttaken
    // is the 2nd not-low conf:  not a critical path, needed only on update.
    for (int i = HitBank - 1; i > 0; i--) {
      for (int j = 0; j < MINITAGE__ASSOC; j++)
        if (gtable[i][GGI[j][i] + j].tag == GTAG[i]) {

          {
            AltAssoc = j;
            AltBank = i;
            break;
          }
        }
      if (AltBank > 0)
        break;
    }
    if (HitBank > 0) {

      if (abs(2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr + 1) ==
          1) {
        for (int i = HitBank - 1; i > 0; i--) {
          for (int j = 0; j < MINITAGE__ASSOC; j++)
            if (gtable[i][GGI[j][i] + j].tag == GTAG[i]) {
              if (abs(2 * gtable[i][GGI[j][i] + j].ctr + 1) != 1)
              // slightly better to pick alternate prediction as not low
              // confidence
              {
                HCpredBank = i;

                HCpredAssoc = j;
                HCpred = (gtable[i][GGI[j][i] + j].ctr >= 0);
                HCCONF = abs(2 * gtable[i][GGI[j][i] + j].ctr + 1) >> 1;
                break;
              }
            }
          if (HCpredBank > 0)
            break;
        }
      }

      else {
        HCpredBank = HitBank;
        HCpredAssoc = HitAssoc;
        HCpred = LongestMatchPred;
        HCCONF = TAGECONF;
      }
    }

    // computes the prediction and the alternate prediction

    if (HitBank > 0) {
      ALTCONF = 0;
      if (AltBank > 0) {
        alttaken =
            (gtable[AltBank][GGI[AltAssoc][AltBank] + AltAssoc].ctr >= 0);
        ALTCONF =
            abs(2 * gtable[AltBank][GGI[AltAssoc][AltBank] + AltAssoc].ctr +
                1) >>
            1;
      }

      // if the entry is recognized as a newly allocated entry and
      // USE_ALT_ON_NA is positive  use the alternate prediction
      int bg = std::min(HitBank / 4, UANA_BANK_GROUPS - 1);
      int pcbin = (hash_64(PCBR) >> 1) & ((1 << UANA_PC_BITS) - 1);
      int ac = (ALTCONF >= 2);
      bool Huse_alt_on_na = (use_alt_on_na[bg][pcbin][ac] >= 0);

      if ((!Huse_alt_on_na) ||
          (abs(2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr + 1) >
           1)) {
        tage_pred = LongestMatchPred;
        Provider = HitBank;
      } else {
        tage_pred = HCpred;
        Provider = HCpredBank;
      }
    }
  }

  // compute the prediction
  TagePrediction predict(uint64_t seq_no, uint8_t piece, uint64_t PC)
  // checkpoint current hist
  {
    pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), active_hist);
    const TagePrediction pred = predict_using_given_hist(
        seq_no, piece, PC, active_hist, true /*pred_time_predict*/);
    return pred;
  }

  TagePrediction predict_using_given_hist(uint64_t seq_no, uint8_t piece,
                                          uint64_t PCBRANCH,
                                          const cbp_hist_t &hist_to_use,
                                          const bool pred_time_predict) {
    // computes the TAGE table addresses and the partial tags
    uint64_t PCBR = PCBRANCH >> 2;
    Tagepred(PCBR, hist_to_use);
    pred_taken = tage_pred;

    int confidence = 0;
    if (TAGECONF >= 2)
      confidence = 2; // High
    else if (TAGECONF == 1)
      confidence = 1; // Med
    else
      confidence = 0; // Low

    bool hit_in_last = (HitBank == MINITAGE__NHIST);

    return TagePrediction(pred_taken, confidence, hit_in_last, Provider, HCpred,
                          LongestMatchPred);
  }

  void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, int brtype,
                      bool taken, uint64_t nextPC) {
    HistoryUpdate(PC, brtype, taken, nextPC);
  }

  void TrackOtherInst(uint64_t PC, int brtype, bool taken, uint64_t nextPC) {
    HistoryUpdate(PC, brtype, taken, nextPC);
  }

  void HistoryUpdate(uint64_t PCBRANCH, int brtype, bool taken,
                     uint64_t branchTarget) {

    auto &Y = active_hist.ptghist;

    auto &H = active_hist.ch_i;
    auto &G = active_hist.ch_t[0];
    auto &J = active_hist.ch_t[1];
    auto &X = active_hist.phist;

    uint64_t PCBR = (PCBRANCH >> 2);

    if (brtype & 1) {
      uint64_t PC = PCBR;
#ifdef LOOPPREDICTOR
      // only for conditional branch
      if (LVALID) {
        if (pred_taken != predloop)
          ctrupdate(active_hist.WITHLOOP, (predloop == pred_taken), 7);
      }

      loopupdate(PC, pred_taken, false /*alloc*/, active_hist.ltable);
#endif
    }

    int T = PCBR ^ (PCBR >> 2) ^ (PCBR >> 4) ^ (branchTarget >> 1) ^ taken;
    int PATH = ((PCBR ^ (PCBR >> 2))) ^ (branchTarget >> 3);
    active_hist.phist = (active_hist.phist << 5) ^ PATH;
    active_hist.phist = (active_hist.phist & ((1 << 27) - 1));
    for (int t = 0; t < 5; t++) {
      int DIR = (T & 1);
      T >>= 1;
      int PATHBIT = PATH;
      PATH >>= 1;
      Y--;
      ghist[Y & (MINITAGE__HISTBUFFERLENGTH - 1)] = DIR;
      for (int i = 1; i <= MINITAGE__NHIST; i++) {

        H[i].update(ghist, Y);
        G[i].update(ghist, Y);
        J[i].update(ghist, Y);
      }
    }
  }

  // END UPDATE  HISTORIES

  // PREDICTOR UPDATE

  void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir,
              bool predDir, uint64_t branchTarget, bool misprediction) {
    const auto pred_hist_key = get_unique_inst_id(seq_no, piece);
    const auto &pred_time_history = pred_time_histories.at(pred_hist_key);
    const TagePrediction pred = predict_using_given_hist(
        seq_no, piece, PC, pred_time_history, false /*pred_time_predict*/);
    const bool pred_taken = pred.prediction;

    // remove checkpointed hist
    update(PC, resolveDir, pred_taken, branchTarget, pred_time_history,
           misprediction);
    pred_time_histories.erase(pred_hist_key);
  }

  void update(uint64_t PCBRANCH, bool resolveDir, bool pred_taken,
              uint64_t branchTarget, const cbp_hist_t &hist_to_use,
              bool misprediction /*We can pass in FSC override here*/) {

    uint64_t PCBR = PCBRANCH >> 2;
    uint64_t PC = PCBR; // don't ask why :-)

    bool DONE = false;

    // TAGE UPDATE
    bool ALLOC = (HitBank < MINITAGE__NHIST);
    ALLOC &= (LongestMatchPred != resolveDir);
    // ALLOC &= (misprediction);

    //////////////////////////////////////////////////

    if (HitBank > 0) {

      bool PseudoNewAlloc =
          (abs(2 * gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr +
               1) <= 1);
      // an entry is considered as newly allocated if its prediction counter
      // is weak

      if (PseudoNewAlloc) {
        if (LongestMatchPred == resolveDir)
          ALLOC = false;

        if (LongestMatchPred != HCpred) {
          int bg = std::min(HitBank / 4, UANA_BANK_GROUPS - 1);
          int pcbin = (hash_64(PCBR) >> 1) & ((1 << UANA_PC_BITS) - 1);
          int ac = (ALTCONF >= 2);

          ctrupdate(use_alt_on_na[bg][pcbin][ac], (HCpred == resolveDir),
                    MINITAGE__ALTWIDTH);
          // pure TAGE only
        }
      }
    }

    /////////////////////////
#ifdef MINITAGE__FILTERALLOCATION
    // filter allocations: all of this is done at update, not on the critical
    // path
    //  try to evaluate if the misprediction rate is above 1/9th

    if ((tage_pred != resolveDir) || ((MYRANDOM() & 31) < 4))
      ctrupdate(CountMiss11, (tage_pred != resolveDir), 8);

    if (HitBank > 0) {
      bool PseudoNewAlloc = (TAGECONF == 0);

      if (PseudoNewAlloc) {
        // Here we count correct/wrong weak counters to guide allocation
        for (int i = HitBank / 4; i <= MINITAGE__NHIST / 4; i++) {

          ctrupdate(COUNT50[MINITAGE__INDCOUNT][i],
                    (resolveDir == LongestMatchPred), 7);
          ctrupdate(Count50[i], (resolveDir == LongestMatchPred), 7);
          // more or less than 50 % good predictions on weak counters
          if ((LongestMatchPred != resolveDir) || ((MYRANDOM() & 31) > 1)) {
            ctrupdate(COUNT16_31[MINITAGE__INDCOUNT][i],
                      (resolveDir == LongestMatchPred), 7);
            ctrupdate(Count16_31[i], (resolveDir == LongestMatchPred), 7);
          }
          // more or less than 16/31  good predictions on weak counters
          if ((LongestMatchPred != resolveDir) || ((MYRANDOM() & 31) > 3)) {
            ctrupdate(COUNT8_15[MINITAGE__INDCOUNT][i],
                      (resolveDir == LongestMatchPred), 7);
            ctrupdate(Count8_15[i], (resolveDir == LongestMatchPred), 7);
          }
          // more or less than 16/31  good predictions on weak counters
        }
      }
    }
    //  when allocating a new entry is unlikely to result in a good
    //  prediction, rarely allocate
    if (TAGECONF < 2) {
      if ((COUNT50[MINITAGE__INDCOUNT][(HitBank + 1) / 4] < 0) &
          (Count50[(HitBank + 1) / 4] < 0)) {
        ALLOC &= ((MYRANDOM() & ((1 << (4 - TAGECONF)) - 1)) == 0);
      } else
        // the future allocated entry is not that likely to be correct
        if ((COUNT16_31[MINITAGE__INDCOUNT][(HitBank + 1) / 4] < 0) &
            (Count16_31[(HitBank + 1) / 4] < 0)) {
          ALLOC &= ((MYRANDOM() & ((1 << (2 - TAGECONF)) - 1)) == 0);
        } else if ((COUNT8_15[MINITAGE__INDCOUNT][(HitBank + 1) / 4] < 0) &
                   (Count8_15[(HitBank + 1) / 4] < 0)) {
          ALLOC &= ((MYRANDOM() & ((1 << (1 - TAGECONF)) - 1)) == 0);
        }
    }
    // The benefit is essentially to decrease the number of allocations
#endif

    if (ALLOC) {

      int MaxNALLOC =
          (TAGECONF) +
          (!((COUNT50[MINITAGE__INDCOUNT][(HitBank + 1) / 4] < 0) &
             (Count50[(HitBank + 1) / 4] < 0))) +
          (!((COUNT16_31[MINITAGE__INDCOUNT][(HitBank + 1) / 4] < 0) &
             (Count16_31[(HitBank + 1) / 4] < 0))) +
          (!((COUNT8_15[MINITAGE__INDCOUNT][(HitBank + 1) / 4] < 0) &
             (Count8_15[(HitBank + 1) / 4] < 0)));

      int NA = 0;
      int DEP = HitBank + 1;
      int Penalty = 0;
      DEP += ((MYRANDOM() & 1) == 0);
      DEP += ((MYRANDOM() & 3) == 0);
      if (DEP == HitBank)
        DEP = HitBank + 1;

      bool First = true;
      bool Test = false;

      for (int i = DEP; i <= MINITAGE__NHIST; i++) {
        bool done = false;
        uint j = (MYRANDOM() % MINITAGE__ASSOC);
        {
          bool REP[2] = {false};
          int IREP[2] = {0};
          bool MOVE[2] = {false};
          for (int J = 0; J < MINITAGE__ASSOC; J++) {
            j++;
            j = j % MINITAGE__ASSOC;
            if (gtable[i][GGI[j][i] + j].u == 0) {
              REP[j] = true;
              IREP[j] = GGI[j][i] + j;
            } else if (MINITAGE__REPSK == 1) {
              if (MINITAGE__PSK == 1)
                IREP[j] = (GGI[j][i] ^
                           (((gtable[i][GGI[j][i] + j].tag) & 0xff) << (1))) +
                          (j ^ 1);
              REP[j] = (gtable[i][IREP[j]].u == 0);
              MOVE[j] = true;
            }

            if (REP[j])

              if (((MINITAGE__UWIDTH == 1) &&
                       ((((MYRANDOM() &
                           ((1 << (abs(2 * gtable[i][GGI[j][i] + j].ctr + 1) >>
                                   1)) -
                            1)) == 0))) ||
                   (TICKH >= MINITAGE__BORNTICK / 2)) ||
                  (MINITAGE__UWIDTH == 2)) {
                done = true;
                if (MOVE[j]) {
                  gtable[i][IREP[j]].u = gtable[i][GGI[j][i] + j].u;
                  gtable[i][IREP[j]].tag = gtable[i][GGI[j][i] + j].tag;
                  gtable[i][IREP[j]].ctr = gtable[i][GGI[j][i] + j].ctr;
                }

                gtable[i][GGI[j][i] + j].tag = GTAG[i];
#ifndef MINITAGE__FORCEU
                gtable[i][GGI[j][i] + j].u = 0;
#else

                gtable[i][GGI[j][i] + j].u =
                    ((MINITAGE__UWIDTH == 2) ||
                     (TICKH >= MINITAGE__BORNTICK / 2)) &
                    (First ? 1 : 0);
#endif
                gtable[i][GGI[j][i] + j].ctr = (resolveDir) ? 0 : -1;

                NA++;
                if ((i >= 3) || (!First))
                  MaxNALLOC--;
                First = false;
                i += 2;
                i -= ((MYRANDOM() & 1) == 0);
                i += ((MYRANDOM() & 1) == 0);
                i += ((MYRANDOM() & 3) == 0);
                break;
              }
          }
          if (MaxNALLOC < 0)
            break;
          if (!done) {
#ifdef MINITAGE__FORCEU

            for (int j = 0; j < MINITAGE__ASSOC; j++) {
              {
                // some just allocated entries  have been set to useful
                if ((MYRANDOM() &
                     ((1 << (1 + MINITAGE__LOGASSOC + MINITAGE__REPSK)) - 1)) ==
                    0)
                  if (abs(2 * gtable[i][GGI[j][i] + j].ctr + 1) == 1)
                    if (gtable[i][GGI[j][i] + j].u == 1)
                      gtable[i][GGI[j][i] + j].u--;
                if (MINITAGE__REPSK == 1)
                  if ((MYRANDOM() &
                       ((1 << (1 + MINITAGE__LOGASSOC + MINITAGE__REPSK)) -
                        1)) == 0)
                    if (abs(2 * gtable[i][IREP[j]].ctr + 1) == 1)
                      if (gtable[i][IREP[j]].u == 1)
                        gtable[i][IREP[j]].u--;
              }
            }
#endif
            Penalty++;
          }
        }

        //////////////////////////////////////////////
      }

      // we set two counts to monitor: "time to reset u" and "almost time
      // reset u": TICKH is useful only if MINITAGE__UWIDTH =1
#ifndef MINITAGE__PROTECTRECENTALLOCUSEFUL
      TICKH += Penalty - NA;
      TICK += Penalty - 2 * NA;
#else
      TICKH += 2 * Penalty - 3 * NA;
      TICK += Penalty - (2 + 2 * (CountMiss11 >= 0)) * NA;
#endif
      if (TICKH < 0)
        TICKH = 0;
      if (TICKH >= MINITAGE__BORNTICK)
        TICKH = MINITAGE__BORNTICK;

      if (TICK < 0)
        TICK = 0;
      if (TICK >= MINITAGE__BORNTICK) {
        for (int j = 0;
             j < MINITAGE__ASSOC * (1 << MINITAGE__LOGG) * MINITAGE__NBANK;
             j++) {
          // this is not realistic: in a real processor:    gtable[1][j].u >>=
          // 1;
          if (gtable[1][j].u > 0)
            gtable[1][j].u--;
        }
        TICK = 0;
        TICKH = 0;
      }
    }

    // update TAGE predictions

    if (HitBank > 0) {
#ifdef MINITAGE__UPDATEALTONWEAKMISP
      // This protection, when prediction is low confidence
      if (TAGECONF == 0) {
        if (LongestMatchPred != resolveDir) {
          if (AltBank != HCpredBank)
            ctrupdate(gtable[AltBank][GGI[AltAssoc][AltBank] + AltAssoc].ctr,
                      resolveDir, MINITAGE__CWIDTH);
          if (HCpredBank > 0) {
            ctrupdate(
                gtable[HCpredBank][GGI[HCpredAssoc][HCpredBank] + HCpredAssoc]
                    .ctr,
                resolveDir, MINITAGE__CWIDTH);

          }

          else
            baseupdate(resolveDir);
        }
      }

#endif
      ctrupdate(gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].ctr,
                resolveDir, MINITAGE__CWIDTH);

    } else
      baseupdate(resolveDir);
    ////////: note that here it is alttaken that is used: the second hitting
    /// entry

    if (LongestMatchPred != alttaken) {

      if (LongestMatchPred == resolveDir) {
#ifdef MINITAGE__PROTECTRECENTALLOCUSEFUL
        // if (TICKH >= MINITAGE__BORNTICK)
        if (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u == 0)
          gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u++;
          // Recent useful will survive one smart reset
#endif
        if (gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u <
            (1 << MINITAGE__UWIDTH) - 1)
          gtable[HitBank][GGI[HitAssoc][HitBank] + HitAssoc].u++;
      }
    }
  }
#ifdef LOOPPREDICTOR
  int lindex(uint64_t PC) {
    return ((hash_64(PC) & ((1 << (MINITAGE__LOGL - 2)) - 1)) << 2);
  }

// loop prediction: only used if high confidence
// skewed associative 4-way
// At fetch time: speculative
#define MINITAGE__CONFLOOP 15
  bool getloop(uint64_t PC, const cbp_hist_t &hist_to_use) {
    const auto &ltable = hist_to_use.ltable;
    LHIT = -1;

    LI = lindex(PC);
    LIB = ((PC >> (MINITAGE__LOGL - 2)) & ((1 << (MINITAGE__LOGL - 2)) - 1));
    LTAG = (PC >> (MINITAGE__LOGL - 2)) & ((1 << 2 * MINITAGE__LOOPTAG) - 1);
    LTAG ^= (LTAG >> MINITAGE__LOOPTAG);
    LTAG = (LTAG & ((1 << MINITAGE__LOOPTAG) - 1));

    for (int i = 0; i < 4; i++) {
      int index = (LI ^ ((LIB >> i) << 2)) + i;

      if (ltable[index].TAG == LTAG) {
        LHIT = i;
        LVALID = ((ltable[index].confid == MINITAGE__CONFLOOP) ||
                  (ltable[index].confid * ltable[index].NbIter > 128));

        if (ltable[index].CurrentIter + 1 == ltable[index].NbIter)
          return (!(ltable[index].dir));
        return ((ltable[index].dir));
      }
    }

    LVALID = false;
    return (false);
  }

  void loopupdate(uint64_t PC, bool Taken, bool ALLOC,
                  std::vector<lentry> &ltable) {
    if (LHIT >= 0) {
      int index = (LI ^ ((LIB >> LHIT) << 2)) + LHIT;
      // already a hit
      if (LVALID) {
        if (Taken != predloop) {
          // free the entry
          ltable[index].NbIter = 0;
          ltable[index].age = 0;
          ltable[index].confid = 0;
          ltable[index].CurrentIter = 0;
          return;

        } else if ((predloop != tage_pred) || ((MYRANDOM() & 7) == 0))
          if (ltable[index].age < MINITAGE__CONFLOOP)
            ltable[index].age++;
      }

      ltable[index].CurrentIter++;
      ltable[index].CurrentIter &= ((1 << MINITAGE__WIDTHNBITERLOOP) - 1);
      // loop with more than 2** MINITAGE__WIDTHNBITERLOOP iterations are not
      // treated correctly; but who cares :-)
      if (ltable[index].CurrentIter > ltable[index].NbIter) {
        ltable[index].confid = 0;
        ltable[index].NbIter = 0;
        // treat like the 1st encounter of the loop
      }
      if (Taken != ltable[index].dir) {
        if (ltable[index].CurrentIter == ltable[index].NbIter) {
          if (ltable[index].confid < MINITAGE__CONFLOOP)
            ltable[index].confid++;
          if (ltable[index].NbIter < 3)
          // just do not predict when the loop count is 1 or 2
          {
            // free the entry
            ltable[index].dir = Taken;
            ltable[index].NbIter = 0;
            ltable[index].age = 0;
            ltable[index].confid = 0;
          }
        } else {
          if (ltable[index].NbIter == 0) {
            // first complete nest;
            ltable[index].confid = 0;
            ltable[index].NbIter = ltable[index].CurrentIter;
          } else {
            // not the same number of iterations as last time: free the entry
            ltable[index].NbIter = 0;
            ltable[index].confid = 0;
          }
        }
        ltable[index].CurrentIter = 0;
      }

    } else if (ALLOC)

    {
      uint64_t X = MYRANDOM() & 3;

      if ((MYRANDOM() & 3) == 0)
        for (int i = 0; i < 4; i++) {
          int loop_hit_way_loc = (X + i) & 3;
          int index =
              (LI ^ ((LIB >> loop_hit_way_loc) << 2)) + loop_hit_way_loc;
          if (ltable[index].age == 0) {
            ltable[index].dir = !Taken;
            // most of mispredictions are on last iterations
            ltable[index].TAG = LTAG;
            ltable[index].NbIter = 0;
            ltable[index].age = 7;
            ltable[index].confid = 0;
            ltable[index].CurrentIter = 0;
            break;

          } else
            ltable[index].age--;
          break;
        }
    }
  }
#endif // LOOPPREDICTOR
};
#endif
#undef UINT64

static CBP2025 cbp2025;
}