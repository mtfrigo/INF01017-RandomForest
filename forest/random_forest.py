
from tree.algorithm import DecisionTree
import numpy as np
import pandas as pd
#from pandas_ml import ConfusionMatrix

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

    accuracy = self.validate(classifications, train_set.get_classes())

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

  def validate(self, classifications, classes):

    all_predictions = []
    all_classes = []
    total_TP = 0
    total_TN = 0
    total_FP = 0
    total_FN = 0

    true_positives = {}
    true_negatives = {}
    false_positives = {}
    false_negatives = {}
    precision = {}
    recall = {}
    f_score = {}

    for class_ in classes:
      true_positives[class_] = 0
      true_negatives[class_] = 0
      false_positives[class_] = 0
      false_negatives[class_] = 0
      precision[class_] = 0
      recall[class_] = 0
      f_score[class_] = 0

    # Creating confusion matrix
    for c in classifications:
      (instance, label, prediction) = c
      all_classes.append(label)
      all_predictions.append(prediction)

    y_actu = pd.Series(all_classes, name='Classe verdadeira')
    y_pred = pd.Series(all_predictions, name='Classe predita')
    confusion = pd.crosstab(y_actu, y_pred)

    print(confusion)

    # Sum of all confusion matrix values
    confusion_matrix_sum = sum([sum(row) for row in confusion.values])

    # TP, FP, FN and TN for each class
    for i in range(len(classes)):
      row = sum(confusion.values[i][:])
      col = sum([row[i] for row in confusion.values])
      true_positives[classes[i]] = confusion.values[i][i]
      false_positives[classes[i]] = row - true_positives[classes[i]]
      false_negatives[classes[i]] = col - true_positives[classes[i]]
      true_negatives[classes[i]] = confusion_matrix_sum - row - col + true_positives[classes[i]]

    # Macro F-Score
    for class_ in classes:
      total_TP += true_positives[class_]
      total_FP += false_positives[class_]
      total_FN += false_negatives[class_]
      precision[class_] = true_positives[class_] / (true_positives[class_] + false_positives[class_])
      recall[class_] = true_positives[class_] / (true_positives[class_] + false_negatives[class_])
      f_score[class_] = 2 * ((precision[class_] * recall[class_]) / (precision[class_] + recall[class_]))
      print("Class", class_, "Precision: ", "{:.2f}".format(precision[class_] * 100) + "%")
      print("Class", class_, "Recall: ", "{:.2f}".format(recall[class_] * 100) + "%")
      print("Class", class_, "F-Score: ", "{:.2f}".format(f_score[class_] * 100) + "%")

    # Micro F-Score
    recall = total_TP / (total_TP + total_FN)
    precision = total_TP / (total_TP + total_FP)
    accuracy = total_TP / confusion_matrix_sum
    f_score = 2 * ((precision * recall) / (precision + recall))
    print("Random Forest Accuracy: " + "{:.2f}".format(accuracy*100) + "%")
    print("Random Forest Precision: " + "{:.2f}".format(precision * 100) + "%")
    print("Random Forest Recall: " + "{:.2f}".format(recall * 100) + "%")
    print("Random Forest F-Score: " + "{:.2f}".format(f_score * 100) + "%")

    return accuracy