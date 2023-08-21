# In this file, I am finding a thresholding over the D(P(Y|X), P(Y|X, M))
# Based on which I decide whether to collect the human feature or not
# Afterwards, I update my belief about the human label and re-train the model
# The threshold is found via a grid search in validation set


# First, I define a class similar to Mozannar's that has fit and test
from human_ai_deferral/basemethod import BaseMethod, BaseSurrogateMethod
