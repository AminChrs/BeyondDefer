# In this file, I am finding a thresholding over the D(P(Y|X), P(Y|X, M))
# Based on which I decide whether to collect the human feature or not
# Afterwards, I update my belief about the human label and re-train the model
# The threshold is found via a grid search in validation set


