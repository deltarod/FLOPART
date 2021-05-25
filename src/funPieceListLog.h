#include <list>
#include <vector>

// NOTE: please only define prototypes in this file (do not define
// methods directly -- instead define them in funPieceList.cpp). This
// is more compatible with R's Makefile, which automatically
// recompiles object files when there are changes to *.cpp but not *.h
// files.

class MinimizeResult{
public:
  double cost;
  double log_mean;
  double prev_log_mean;
  int prev_seg_end;
  int prev_seg_offset;
  void write_mean_end_state(double *mean_vec, int *end_vec, int *state_vec, int offset);
};

class PoissonLossPieceLog {
public:
  double Linear;
  double Log;
  double Constant;
  double min_log_mean;
  double max_log_mean;
  int data_i;
  double prev_log_mean;
  PoissonLossPieceLog();
  PoissonLossPieceLog
    (double li, double lo, double co, double m, double M, int i, double);
  double argmin();
  double argmin_mean();
  double get_smaller_root(double);
  double get_larger_root(double);
  bool has_two_roots(double);
  double getCost(double mean);
  double getDeriv(double);
  double PoissonLoss(double);
  double PoissonDeriv(double);
};

typedef std::list<PoissonLossPieceLog> PoissonLossPieceListLog;

class PiecewisePoissonLossLog {
public:
  PoissonLossPieceListLog piece_list;
  bool is_infinite();
  void set_infinite();
  void set_to_min_less_of(PiecewisePoissonLossLog *, int);
  void set_to_min_more_of(PiecewisePoissonLossLog *, int);
  void set_to_min_env_of
    (PiecewisePoissonLossLog *, PiecewisePoissonLossLog *, int);
  int check_min_of(PiecewisePoissonLossLog *, PiecewisePoissonLossLog *);
  void push_min_pieces
    (PiecewisePoissonLossLog *, PiecewisePoissonLossLog *,
     PoissonLossPieceListLog::iterator, PoissonLossPieceListLog::iterator, int);
  void push_piece(PoissonLossPieceListLog::iterator, double, double);
  void add(double Linear, double Log, double Constant);
  void multiply(double);
  void set_prev_seg_end(int prev_seg_end);
  void findMean(MinimizeResult *bestMinResult);
  double findCost(double mean);
  void Minimize
    (MinimizeResult *res);
  void adjustWeights
  (const double,const double,const double*,const int,const int*);
  void addPenalty(double penalty, double cum_weight_prev_i);
};

bool sameFuns(PoissonLossPieceListLog::iterator, PoissonLossPieceListLog::iterator);

class CostMatrix {
 public:
  std::vector<PiecewisePoissonLossLog> cost_vec;
  int data_count;
  MinimizeResult minimize();
  CostMatrix(int);
  void copy_min_cost_intervals(double*, int*);
  void decode_optimal_mean_end_state(double*, int*, int*);
};

