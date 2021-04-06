
from tree.algorithm import DecisionTree
import numpy as np
import pandas as pd

class RandomForest(object):
  def __init__(self, data_frame, n_trees):
    print("Generating random forest...")

    self.data_frame = data_frame
    self.n_trees = n_trees
    self.trees = []
    self.training_sets = []
    self.testing_sets = []

    self.generate()

  def cross_validation(self, k):
    print("oi")


  def generate(self):
    k = 10
    # 1 - Stratify the data_frame and divides the dataset in k-folds, where 1 fold is for test
      # test_set => take 1 fold
      # train_set => merge of others folds 

    folds = self.data_frame.stratify(k)
    test_set = self.data_frame.create_subset(folds.pop(0)) 

    merged_folds = []
    for fold in folds:
      for i in range(len(fold.values)):
        merged_folds.append(fold.values[i])

    train_set = self.data_frame.create_subset(merged_folds)

    # 2 - With the train_set 
      # bootstrap and generate n trees
      # each tree generated with one bootstrap generated with the train_set

    for n in range(self.n_trees):
      (data_frame_train, data_frame_test) = train_set.bootstrap()

      tree = DecisionTree(data_frame_train.discretize_by_neighborhood(), 10)
      self.trees.append(tree)

    # 3 - Classify all the test_set instance
      # Each tree generate a label
      # Make the majority voting  

    classifications = self.classify_dataset(test_set)

    # 4 - Evaluate
      # Accuracy mean and deviation

    accuracy = self.validate(classifications)

    print("Accuracy: " + "{:.2f}".format(accuracy*100) + "%" )

  def classify(self, test_instance):
    instance_predictions = []

    for tree in self.trees:
      instance_predictions.append(tree.classify(test_instance))

    return self.voting(instance_predictions)

  def classify_dataset(self, test_dataset):
    target = test_dataset._target_class

    predicts = 0
    right_predicts = 0

    classifications = []

    for instance in test_dataset._data_frame.to_dict(orient="records"):
      predict = self.classify(instance)
      classifications.append((instance, instance[target], predict))
    return classifications
    

  def voting(self, predictions):
    counter = {}
    unique_labels = []

    for p in predictions:
      if p not in unique_labels:
        unique_labels.append(p)
        counter[p] = 0

    for p in predictions: 
      counter[p] = counter[p] + 1

    return max(counter, key=counter.get)

  def validate(self, classifications):

    right_predictions = 0

    for c in classifications:
      (instance, label, prediction) = c

      if label == prediction:
        right_predictions += 1

    accuracy = right_predictions / len(classifications)

    return accuracy

