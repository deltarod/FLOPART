#include <vector>
#include <stdio.h>
#include <iostream>
#include "funPieceListLog.h"
#include <math.h>
#include "FLOPART.h"

int FLOPART
  (const int *data_vec, const double *weight_vec, const int data_count,
   const double penalty,
   const int *label_types,
   const int *label_starts,
   const int *label_ends,
   const int label_count,
   // the following matrices are for output.
   // cost_mat and intervals_mat store the optimal cost and number of intervals
   // at each time point, for the up and down cost models.
   // end_vec and mean_vec store the best model up to and including the
   // last data point. 
   double *cost_mat, //data_count x 2.
   int *end_vec,//data_count
   double *mean_vec,//data_count
   int *intervals_mat,//data_count x 2
   int *state_vec
   ){
  for(int label_i=0; label_i < label_count; label_i++){
    if(label_starts[label_i] < 0){
      return ERROR_LABEL_START_MUST_BE_AT_LEAST_ZERO;
    }
    if(data_count <= label_ends[label_i]){
      return ERROR_LABEL_END_MUST_BE_LESS_THAN_DATA_SIZE;
    }
    if(label_ends[label_i] < label_starts[label_i]){
      return ERROR_LABEL_END_MUST_BE_AT_LEAST_LABEL_START;
    }
    if(0 < label_i && label_starts[label_i] <= label_ends[label_i-1]){
      return ERROR_LABEL_START_SHOULD_BE_GREATER_THAN_PREVIOUS_LABEL_END;
    }
    switch(label_types[label_i]){
    case LABEL_NOPEAKS:
    case LABEL_PEAKSTART:
    case LABEL_PEAKEND:
      break;
    default:
      return ERROR_UNRECOGNIZED_LABEL_TYPE;
    }
  }
  bool at_beginning = false;
  bool at_end = false;  
  int curr_label_index = 0;
  int curr_label_type = LABEL_UNLABELED; //1 = peakStart, 0 = noPeak, -1 = peakEnd
  double min_log_mean=INFINITY, max_log_mean=-INFINITY;
  for(int data_i=0; data_i<data_count; data_i++){
    
    double log_data = log( (double)(data_vec[data_i]) );
    if(log_data < min_log_mean){
      min_log_mean = log_data;
    }
    if(max_log_mean < log_data){
      max_log_mean = log_data;
    }
  }
  if(min_log_mean == max_log_mean){
    return ERROR_MIN_MAX_SAME;
  }
  CostMatrix cost_funs(data_count);
  PiecewisePoissonLossLog *up_cost, *down_cost, *up_cost_prev, *down_cost_prev;
  // initialize to avoid warnings.
  up_cost_prev = &cost_funs.cost_vec[0];
  down_cost_prev = &cost_funs.cost_vec[0];
  PiecewisePoissonLossLog cost_of_change_up, cost_of_change_down, initial_cost;
  // initialization C_1(m)=loss_1(m)/w_1
  initial_cost.piece_list.emplace_back
    (1.0, -data_vec[0], 0.0,
     min_log_mean, max_log_mean, -1, false);
  int verbose=0;
  double cum_weight_i = 0.0, cum_weight_prev_i = 0.0;
  for(int data_i=0; data_i<data_count; data_i++){
    cum_weight_i += weight_vec[data_i];
    up_cost = &cost_funs.cost_vec[data_i];
    down_cost = &cost_funs.cost_vec[data_i + data_count];
    at_beginning = false;
    at_end = false;
    //if we haven't checked all the labels 
    if(curr_label_index < label_count){
      if(data_i >= label_starts[curr_label_index] && data_i <= label_ends[curr_label_index]){
        //if >= current start and <= current end
        curr_label_type = label_types[curr_label_index];
        if(data_i == label_starts[curr_label_index]){
          at_beginning = true;
        }
        if(data_i == label_ends[curr_label_index]){
          at_end = true;
          curr_label_index++;
        }
      }else{
        curr_label_type = LABEL_UNLABELED;
      }
    }else{
      curr_label_type = LABEL_UNLABELED;
    }
    //---UP COST CALCULATIONS---
    if(data_i == 0){
      *up_cost = initial_cost;
    } else if(curr_label_type == LABEL_PEAKEND && !at_beginning && !at_end){
      *up_cost = *up_cost_prev;
    } else if(curr_label_type == LABEL_NOPEAKS
	      || (curr_label_type == LABEL_PEAKSTART && at_beginning)
              || (curr_label_type == LABEL_PEAKEND && at_end)){
      //if in no peaks or beginning of peak start or end of peak end
      up_cost->set_infinite();
    } else if(curr_label_type == LABEL_UNLABELED 
              || (curr_label_type== LABEL_PEAKSTART && !at_beginning) 
              || (curr_label_type == LABEL_PEAKEND && at_beginning)){
      //if unlabeled or not beginning of peak start or beginning of peak end  
      //up cost is min of prev_up, min_less(prev+down + penalty)
      cost_of_change_up.set_to_min_less_of(down_cost_prev, verbose);
      cost_of_change_up.set_prev_seg_end(data_i-1);
      cost_of_change_up.addPenalty(penalty, cum_weight_prev_i);
      up_cost->set_to_min_env_of(up_cost_prev, &cost_of_change_up, verbose);
    }
    //---DOWN COST CALCULATIONS--- 
    if(data_i==0){
      *down_cost = initial_cost;
    } else if((curr_label_type == LABEL_PEAKEND && at_beginning) ||
            (curr_label_type == LABEL_PEAKSTART && at_end)){
      //if at beginning of peak end or end of peak start, infinite down
      down_cost->set_infinite();
    } else if((curr_label_type == LABEL_NOPEAKS && !at_beginning)
	      || (curr_label_type == LABEL_PEAKSTART && !at_beginning && !at_end)){
      //if not at beginning of noPeaks or middle of peak start
      *down_cost = *down_cost_prev;
    } else if(curr_label_type == LABEL_UNLABELED 
              || (curr_label_type == LABEL_PEAKSTART && at_beginning)
              || (curr_label_type == LABEL_NOPEAKS && at_beginning)
              || (curr_label_type == LABEL_PEAKEND && !at_beginning)){
      //if unlabeled or first or no peaks/peakStart or not at beginning of peak end
      cost_of_change_down.set_to_min_more_of(up_cost_prev, verbose);
      cost_of_change_down.set_prev_seg_end(data_i-1);
      cost_of_change_down.addPenalty(penalty, cum_weight_prev_i);
      down_cost->set_to_min_env_of(down_cost_prev, &cost_of_change_down, verbose);
    } 
    up_cost->adjustWeights(cum_weight_prev_i, cum_weight_i, weight_vec,
                           data_i, data_vec);
    down_cost->adjustWeights(cum_weight_prev_i, cum_weight_i, weight_vec,
                             data_i, data_vec);
    cum_weight_prev_i = cum_weight_i;
    up_cost_prev = up_cost;
    down_cost_prev = down_cost;
  }
  // Then store number of intervals and optimal cost value, for each
  // cost function.
  cost_funs.copy_min_cost_intervals(cost_mat, intervals_mat);
  cost_funs.decode_optimal_mean_end_state(mean_vec, end_vec, state_vec);
  return 0;
}

