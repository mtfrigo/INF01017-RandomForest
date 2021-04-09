import math
import pandas as pd
import random
import numpy as np

# 549 #311 #777 
random.seed(654)
np.random.seed(654)

class DataFrame(object):
  _data_frame = None
  _header = []
  _target_class = None


  def __init__(self, data_frame, attributes_types, target_class):
    self._data_frame = data_frame
    self.attributes_types = attributes_types
    self._header = list(self._data_frame.columns.values)
    self._target_class = target_class

  def pre_process(self):
    data_frame_copy = self._data_frame.copy()

    for attribute in self.get_attributes():
      if self.attributes_types[attribute] == 'numeric':
        attribute_values = (self._data_frame.get(attribute)).copy()
        avg = attribute_values.mean()*1.0
        for i, v in enumerate(attribute_values):
          if math.isnan(v):
            attribute_values[i] = avg

        data_frame_copy[attribute] = attribute_values

    self._data_frame = data_frame_copy

  def get_instances(self):
    return self._data_frame

  def get_attributes(self):
    attributes = list(self._data_frame.columns.values)

    if attributes.index(self._target_class) >= 0:
      attributes.remove(self._target_class)

    return attributes

  def get_instances_by_class(self):
    classes = self.get_classes()

    data = {}

    for _class in classes:
      values = (self._data_frame.loc[self._data_frame[self._target_class] == _class])
      data[_class] = values
      # data[_class] = self._data_frame.loc[self._data_frame[self._target_class] == _class]

    return data

  def get_instances_by_attribute(self):
    attributes = self.get_attributes()

    data = {}

    for _attribute in attributes:
      data[_attribute] = (self._data_frame.get(_attribute)).values

    return data

  def get_instances_by_attribute_value(self, attribute, value):
    data_frame = DataFrame(self._data_frame, self.attributes_types, self._target_class)
    instances_by_attribute = data_frame.get_instances_by_attribute()
    return DataFrame(data_frame._data_frame.loc[data_frame._data_frame[attribute] == value], self.attributes_types, self._target_class)

  def get_classes(self):
    return self._data_frame.get(self._target_class).unique()

  def get_most_frequent_class(self):
    instances = self.get_instances_by_class()
    classes = list(instances.keys())

    most_occurred_class_count = max([len(value) for value in instances.values()])
    most_occurred_class = [k for k, value in instances.items() if len(value) == most_occurred_class_count]

    try:
      return most_occurred_class[random.randint(0, 1)]
    except IndexError:
      return most_occurred_class[0]

  def get_most_frequent_attribute_value(self, attribute):
    instances_by_atribute = self.get_instances_by_attribute()
    values = instances_by_atribute[attribute]

    dict = {}

    count, value = 0, ''
    for item in reversed(values):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count :
            count, value = dict[item], item
    return value

  def get_information_gain(self, attribute):

    instances_by_atribute = self.get_instances_by_attribute()

    values = instances_by_atribute[attribute]
    num_of_values = len(instances_by_atribute[attribute])
    value_counter = {}

    info_attribute = 0

    for value in values:
      if value in list(value_counter):
          value_counter[value] += 1
      else:
          value_counter[value] = 1

    for value in value_counter:
      info = (self.get_instances_by_attribute_value(attribute, value)).get_entropy()
      info_attribute += ((value_counter[value] / num_of_values ) * info)
    
    info = self.get_entropy()

    return info - info_attribute

  def get_entropy(self):
    instance_by_class = self.get_instances_by_class()

    info = 0

    for _class in instance_by_class:
      pi = len(instance_by_class[_class]) / len(self._data_frame.values)

      info -= pi * math.log(pi, 2)

    return info

  def get_attribute_unique_values(self, attribute):
    return self._data_frame.get(attribute).unique()

  def discretize_by_neighborhood(self):
    instances_by_atribute = self.get_instances_by_attribute()
    data_frame_copy = self._data_frame.copy()

    for attribute in self.get_attributes():

      if self.attributes_types[attribute] == 'numeric':
        values = (self._data_frame[[attribute, self._target_class]]).copy()
        values_sorted = (values.sort_values(attribute)).copy()

        # Get all cut points values
        previous_value = (values_sorted[attribute].values[0], values_sorted[self._target_class].values[0])

        cut_values = []
        for i in range(1, len(values_sorted)):
          value = (values_sorted[attribute].values[i], values_sorted[self._target_class].values[i])
          
          if previous_value[1] != value[1]:
            avg = (previous_value[0] + value[0])/2
            if avg not in cut_values:
              cut_values.append(avg)

          previous_value = value

        # Start discretizing
        attribute_values = (self._data_frame.get(attribute)).copy()

        for i, v in enumerate(attribute_values):
          if v <= cut_values[0]:
            attribute_values[i] = 0
            attribute_values[i] = "{0:.3f}<=" + str(cut_values[0])
          elif v >= cut_values[len(cut_values) - 1]:
            attribute_values[i] = 0
            attribute_values[i] = "{0:.3f}>=" + str(cut_values[len(cut_values) - 1])
          else:
            # TODO 0 or 1 to len(cut_values) or len(cut_values) - 1?
            for j in range(1, len(cut_values)):
              if cut_values[j-1] <= v and v < cut_values[j]:
                attribute_values[i] = str(cut_values[j- 1]) + "<={0:.3f}<" + str(cut_values[j])

        data_frame_copy[attribute] = attribute_values


    return DataFrame(data_frame_copy, self.attributes_types, self._target_class)

  def discretize_by_mean(self):
    instances_by_atribute = self.get_instances_by_attribute()
    data_frame_copy = self._data_frame.copy()

    for attribute in self.get_attributes():

      if self.attributes_types[attribute] == 'numeric':
        attribute_values = (self._data_frame.get(attribute)).copy()

        try:
          avg = float("{0:.3f}".format(attribute_values.mean()*1.0))
          for i, v in enumerate(attribute_values):

            if isinstance(attribute_values[i]*1.0, float):
              if attribute_values[i] <= avg:
                attribute_values[i] = "{0:.3f}<=" + str(avg)
              else:
                attribute_values[i] = "{0:.3f}>" + str(avg)
            else:
              raise ValueError
          data_frame_copy[attribute] = attribute_values
        except:
          pass

      # elif self.attributes_types[attribute] == 'categoric':
        # attribute_values = (self._data_frame.get(attribute)).copy()
        # most_frequent_value = self.get_most_frequent_attribute_value(attribute)

        # for i, v in enumerate(attribute_values):
        #   if attribute_values[i] == most_frequent_value:
        #     attribute_values[i] = "{%s}==" + str(most_frequent_value)
        #   else:
        #     attribute_values[i] = "{%s}!=" + str(most_frequent_value)

        # data_frame_copy[attribute] = attribute_values

    return DataFrame(data_frame_copy, self.attributes_types, self._target_class)

  def normalize(self):
    instances_by_atribute = self.get_instances_by_attribute()
    data_frame_copy = self._data_frame.copy()

    for attribute in self.get_attributes():

      if(self.attributes_types[attribute] == 'numeric'):

        attribute_values = (self._data_frame.get(attribute)).copy()

        try:

          min_value = min(attribute_values)
          max_value = max(attribute_values)

          for i, v in enumerate(attribute_values):

            if isinstance(attribute_values[i]*1.0, float):
              attribute_values[i] = (attribute_values[i] - min_value) / (max_value - min_value)
            else:
              raise ValueError
          data_frame_copy[attribute] = attribute_values
        except:
          pass

    return DataFrame(data_frame_copy, self.attributes_types, self._target_class)

  def bootstrap(self, ratio = 0.9): 
    data_frame_copy = self._data_frame
    columns = data_frame_copy.columns.values

    values = data_frame_copy.values

    all_samples_i = np.arange(start=0, stop=len(values))

    train_samples_i = np.random.choice(range(len(values)), size=round(len(values)*ratio), replace=True)
    train_samples = values[train_samples_i]
    
    # These test samples doesnt matter
    test_samples_i = np.setdiff1d(all_samples_i, train_samples_i)
    test_samples = values[test_samples_i]

    # print(len(train_samples))

    data_frame_train = pd.DataFrame(train_samples, columns = columns)
    data_frame_test = pd.DataFrame(test_samples, columns = columns)

    return (DataFrame(data_frame_train, self.attributes_types, self._target_class), DataFrame(data_frame_test, self.attributes_types, self._target_class))

  def stratify(self, k):
    columns = self._data_frame.columns.values

    folds = [[] for i in range(0, k)]

    instances = self._data_frame.copy()
    instances_by_class = self.get_instances_by_class()

    classes = list(instances_by_class.keys())


    n_fold_instances = math.ceil(len(instances)/k)


    for label in classes:
      proportion = len(instances_by_class[label])/ len(instances)

      n_label_fold_instances = math.floor(n_fold_instances * proportion)

      instances_by_label = instances_by_class[label].values
      instances_by_label_i = list(range(len(instances_by_label)))

      for f in range(0, k):
        for i in range(0, n_label_fold_instances):
          
          raffled_index = random.randint(0, len(instances_by_label_i) - 1)

          if len(instances_by_label_i) > 1:
            instance_i = instances_by_label_i.pop(raffled_index)
            folds[f].append(instances_by_label[instance_i])
          else:
            instance_i = instances_by_label_i[0]
            instances_by_label_i = []
            break

    return [pd.DataFrame(folds[f], columns = columns) for f in range(0, k)]

  def create_subset(self, values):
    columns = self._data_frame.columns.values
    return DataFrame(pd.DataFrame(values, columns = columns) , self.attributes_types, self._target_class)