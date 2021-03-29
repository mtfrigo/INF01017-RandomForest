import math
import random

class DecisionTree(object):
  _decision_tree = None
  
  def __init__(self, data_frame):
    print("Generating tree...")
    self._decision_tree = self._generate(data_frame, data_frame.get_attributes())
    print("Generated tree: \n" + str(self))

  def _generate(self, data_frame, attributes):
    # This algorithm is presented in https://www.youtube.com/watch?v=gdyESc6LgfE&list=PL2t5OdGxbjUOpFz0XkEpH9gAxZVvvVZ9B&index=3&ab_channel=MarianaMendoza
    
    # 1
    node = { "attribute": None, "value": {} }

    intances_by_class = data_frame.get_instances_by_class()
    classes = list(intances_by_class.keys())

    # 2
    if len(classes) == 1:
      node["value"] = classes[0]
      return node
    
    # 3
    if len(attributes) == 0:
      node["value"] = data_frame.get_most_frequent_class()
      return node

    # 4
    else:
      # 4.1
      attribute = self._get_best_attribute(data_frame, attributes)
      # 4.2
      node["attribute"] = attribute

      # 4.3
      try:
        attributes.remove(attribute)
      except ValueError:
        attribute = attributes[0]
        attributes = []

      # 4.4
      unique_values = data_frame.get_attribute_unique_values(attribute)

      # 4.4.1
      for value in unique_values:
        sub_data_frame = data_frame.get_instances_by_attribute_value(attribute, value)

        # 4.4.2
        if len(sub_data_frame.get_instances()) == 0:
          node["attribute"] = None
          node["value"] = data_handler.most_occurred_class()

          return node
        
        # 4.4.3
        node["value"][value] = self._generate(sub_data_frame, attributes)

      # 4.5
      return node

  def _get_best_attribute(self, data_frame, attributes):
    info_gain_by_attribute = {}

    for attribute in attributes:
      informative_gain = data_frame.get_informative_gain(attribute)
      info_gain_by_attribute[attribute] = informative_gain

    
    return max(info_gain_by_attribute, key=info_gain_by_attribute.get)

  def classify(self, test_instance):
    node = self._decision_tree

    while node["attribute"] is not None:
      old_node = node

      for value in node["value"]:
        if test_instance[node["attribute"]].values[0] == value:
          node = node["value"][value]
          break

      if node == old_node:
        node = node["value"][value]

    return node["value"]