#nearest neighbor methods work best with numeric features
#first normalize categorical features using min-max or z-score normalization
#One way of handling categorical features is with the Value Difference Metric (VDM). You are
#not required to use VDM, but you may find this to be a useful method
#VDM only works with classification. If you have categorical features in regression,
#you will need to apply a different approach
#(such as one hot coding with Hamming distance, which would also work for regression)


#when finding nearest neighbors use Minkowski's metric of the form