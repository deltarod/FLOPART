# FLOPART 
Functional Labeled Optimal Partitioning - A peak detection algorithm that allows for three types of labels.

# Installation

```r
install.packages("FLOPART")
##OR
if(require("remotes"))install.packages("remotes")
remotes::install_github("alyssajs/FLOPART)
```

# Usage
The main driver function is FLOPART, which takes  
Markup : * `data_vec` An integer vector of data   
        * `weight_vec` A double vector of weights for each data point  
        * `data_count` The number of data points  
        * `penalty` The penalty value  
        * `cost_mat` A vector used for output - after execution, the first half of this vector contains the cost of being in the peak state for each data point, and the second half of this 
vector contains the cost of being in the background state for each data point  
        * `end_vec``` A vector of segment ends - finite positive values represent segment endspoints, infinite or negative values are used as placeholders  
        * `mean_vec` A vector of segment means in reverse order  
        * `intervals_mat`  
        * `label_starts` A vector of indices where labels start (0-based)  
        * `label_ends` A vector of indices where labels end (0-based)  
        * `label_types` A vector of label types where -1 corresponds to a peakEnd label, 0 corresponds to a noPeaks label, and 1 corresponds to a peakStart label  
        * `label_count` The number of labels (should match the length of the label_starts, label_ends, and label_types vector)  

     