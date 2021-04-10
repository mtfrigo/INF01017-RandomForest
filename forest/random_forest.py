
from tree.tree import DecisionTree
import numpy as np
import pandas as pd
#from pandas_ml import ConfusionMatrix

class RandomForest(object):
  def __init__(self, train_set, n_trees):
    print("Generating random forest...")

    self.data_frame = train_set
    self.n_trees = n_trees
    self.trees = []

    self.generate()

  def generate(self):
    self.bootstrap()

  def bootstrap(self):
    for n in range(self.n_trees):
      (data_frame_train, data_frame_test) = self.data_frame.bootstrap(1)
      tree = DecisionTree(data_frame_train.discretize_by_neighborhood())
      # tree = DecisionTree(data_frame_train.discretize_by_mean())
      self.trees.append(tree)

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

  


