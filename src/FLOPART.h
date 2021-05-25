#define ERROR_MIN_MAX_SAME 1
#define ERROR_UNRECOGNIZED_LABEL_TYPE 2
#define ERROR_LABEL_START_SHOULD_BE_GREATER_THAN_PREVIOUS_LABEL_END 3
#define ERROR_LABEL_END_MUST_BE_AT_LEAST_LABEL_START 4
#define ERROR_LABEL_END_MUST_BE_LESS_THAN_DATA_SIZE 5
#define ERROR_LABEL_START_MUST_BE_AT_LEAST_ZERO 6
#define LABEL_NOPEAKS 0
#define LABEL_PEAKSTART 1
#define LABEL_PEAKEND -1
#define LABEL_UNLABELED -2

int FLOPART
(const int *data_vec,
 const double *weight_vec,
 const int data_count,
 const double penalty,
 const int *label_types,
 const int *label_starts,
 const int *label_ends,
 const int label_count,
 // the following matrices are for output.
 double *cost_mat,
 int *end_vec,
 double *mean_vec,
 int *intervals_mat,
 int *state_vec
 );

