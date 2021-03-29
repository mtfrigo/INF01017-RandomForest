import math
import pandas as pd
import random

class DataFrame(object):

  _data_frame = None
  _header = []
  _target_class = None


  def __init__(self, data_frame, target_class):
    self._data_frame = data_frame
    self._header = list(self._data_frame.columns.values)
    self._target_class = target_class

  def get_instances(self):
    
    return self._data_frame

  def get_attributes(self):
    """
    Get attributes removing the target class from the header
    """
    attributes = list(self._data_frame.columns.values)

    if attributes.index(self._target_class):
      attributes.remove(self._target_class)

    return attributes

  def get_instances_by_class(self):
    classes = self.get_classes()

    data = {}

    for _class in classes:
      data[_class] = (self._data_frame.loc[self._data_frame[self._target_class] == _class]).values
      # data[_class] = self._data_frame.loc[self._data_frame[self._target_class] == _class]

    return data

  def get_instances_by_attribute(self):
    attributes = self.get_attributes()

    data = {}

    for _attribute in attributes:
      data[_attribute] = (self._data_frame.get(_attribute)).values

    return data

  def get_instances_by_attribute_value(self, attribute, value):
    data_frame = DataFrame(self._data_frame, self._target_class)
    instances_by_attribute = data_frame.get_instances_by_attribute()
    return DataFrame(data_frame._data_frame.loc[data_frame._data_frame[attribute] == value], self._target_class)

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

  def get_informative_gain(self, attribute):

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

    return info -  info_attribute

  def get_entropy(self):
    instance_by_class = self.get_instances_by_class()

    info = 0

    for _class in instance_by_class:
      pi = len(instance_by_class[_class]) / len(self._data_frame.values)

      info -= pi * math.log(pi, 2)

    return info

  def get_attribute_unique_values(self, attribute):
    return self._data_frame.get(attribute).unique()

  def discretize(self):
    
    instances_by_atribute = self.get_instances_by_attribute()
    data_frame_copy = self._data_frame.copy()


    for attribute in self.get_attributes():
      attribute_values = (self._data_frame.get(attribute)).copy()
      try:
        avg = float("{0:.3f}".format(attribute_values.mean()))

        for i, v in enumerate(attribute_values):
          if attribute_values[i] <= avg:
            new_value = "{0:.3f}<=" + str(avg)
          else:
            new_value = "{0:.3f}>" + str(avg)

          attribute_values[i] = new_value

        data_frame_copy[attribute] = attribute_values
      except:
        pass

    return DataFrame(data_frame_copy, self._target_class)
  