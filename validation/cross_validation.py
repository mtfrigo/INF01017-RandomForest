import math
import random
import pandas as pd

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
    classes = self.data_frame.get_classes()

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
