import math
import random

class CrossValidation(object):

  def __init__(self, data_frame,k_folds):
    self.data_frame = data_frame
    self.bagging(k_folds)
    
  def bagging(self, k):
    folds = self.data_frame.stratify(k)

    self.test_set = self.data_frame.create_subset(folds.pop(0)) 

    merged_folds = []
    for fold in folds:
      for i in range(len(fold.values)):
        merged_folds.append(fold.values[i])

    self.train_set = self.data_frame.create_subset(merged_folds)

  def validate(self, classifications):
    right_predictions = 0

    for c in classifications:
      (instance, label, prediction) = c

      if label == prediction:
        right_predictions += 1

    accuracy = right_predictions / len(classifications)

    print("Accuracy: " + "{:.2f}".format(accuracy*100) + "%" )
