import math
import random

class DecisionTree(object):
  _root = None

  def __init__(self, data_frame, attributes_types, n_attributes = 10):
    print("Generating tree...")

    self.attributes_types = attributes_types
    self._n_attributes = n_attributes
    self._root = self._generate(data_frame, data_frame.get_attributes())
    print("Generated tree: \n" + str(self))

  def _generate(self, data_frame, attributes):
    # This algorithm is presented in https://www.youtube.com/watch?v=gdyESc6LgfE&list=PL2t5OdGxbjUOpFz0XkEpH9gAxZVvvVZ9B&index=3&ab_channel=MarianaMendoza
    
    # 1
    node = Node(data_frame)

    intances_by_class = node.data_frame.get_instances_by_class()
    classes = list(intances_by_class.keys())

    # TODO critérios de parada
    # 2 - Only one class available
    if len(classes) == 1:
      node.label = classes[0]
      node.is_leaf = True
      return node
    
    # 3 - No more attributes
    if len(attributes) == 0:
      node.label = node.data_frame.get_most_frequent_class()
      node.is_leaf = True
      return node

    
    # 4
    else:
      # 4.1
      attributes_pool = self._select_attributes(attributes)
      (attribute, info_gain) = self._get_best_attribute(node.data_frame, attributes_pool)

      # 4.2
      node.attribute = attribute
      node.info_gain = info_gain
      node.attributes_pool = attributes_pool

      # 4.3
      try:
        node.attribute_list = attributes.copy()
        node.attribute_list.remove(attribute)
        
      except ValueError:
        attribute = node.attribute_list[0]
        node.attribute_list = []

      # 4.4
      unique_values = node.data_frame.get_attribute_unique_values(attribute)

      # 4.4.1
      for value in unique_values:
        sub_data_frame = node.data_frame.get_instances_by_attribute_value(attribute, value)

        # 4.4.2
        if len(sub_data_frame.get_instances()) == 0:
          node.attribute = None
          node.label = node.data_frame.get_most_frequent_class()
          node.is_leaf = True
          return node
        
        # 4.4.3
        node.append_child(value, self._generate(sub_data_frame, attributes))

      # 4.5
      return node

  def _get_best_attribute(self, data_frame, attributes):
    info_gain_by_attribute = {}

    for attribute in attributes:
      information_gain = data_frame.get_information_gain(attribute)
      info_gain_by_attribute[attribute] = information_gain

    return max(info_gain_by_attribute, key=info_gain_by_attribute.get), max(info_gain_by_attribute.values())

  def classify(self, test_instance):
    node = self._root

    while node.is_leaf is False:
      old_node = node

      for value in node.value:
        # Numeric
        
        # if node.get_attribute_type() == 'numeric':
        if self.attributes_types[node.attribute] == 'numeric':

          if isinstance(float(test_instance[node.attribute].values[0]*1.0), float) :
            expression = value.format(float(test_instance[node.attribute].values[0]))
            # print(expression, bool(eval(expression)))

            if bool(eval(expression)):
              node = node.value[value]
              break

        # Categoric
        # elif node.get_attribute_type() == 'categoric':
        elif self.attributes_types[node.attribute] == 'categoric':
          if test_instance[node.attribute].values[0] == value:
            node = node.value[value]
            break

      # Force a change
      if node == old_node:
        node = node.value[value]

    return node.label

  def _print_node(self, node, level = 0):
    if node.is_leaf is True:
      return ("|\t" * level) + "|Class: " + str(node.label) + "\n"
    else:
      text = ("|\t" * level) + "|Attr: " + str(node.attribute) + " Gain (" + str(round(node.info_gain,3)) + ")\n"

      for item in node.value:
        text += ("|\t" * (level + 1)) + "|Value: " + str(item) + "\n"
        text += self._print_node(node.value[item], (level + 2))

      return text

    return self._print_node(self._root_old)

  def print_tree(self):
    return print(self._print_node(self._root))

  def _select_attributes(self, attributes):
    random.shuffle(attributes)
    
    max_attributes = self._n_attributes
    num_of_attributes = len(attributes)

    if num_of_attributes > max_attributes:
      num_of_selections = int(math.ceil(math.sqrt(len(attributes))))
      new_attributes = []

      while(len(new_attributes) < num_of_selections):
        new_attribute = attributes[random.randint(0, num_of_attributes - 1 )]

        if new_attribute not in new_attributes:
          new_attributes.append(new_attribute)

      return new_attributes
    else:
      return attributes

  def predict_random_sample(self, sample, target_class):
    predict = self.classify(sample)
    print('Y:', sample[target_class].values[0], 'Predict: ', predict)


class Node(object):
  def __init__(self, data_frame):
    self.data_frame = data_frame
    
    self.attribute = None
    self.value = {}
    self.label = None
    self.is_leaf = False
    self.info_gain = 0

    self.attributes_pool = []
    self.attributes_list = []

  def append_child(self, value, Node):
    self.value[value] = Node

  def get_attribute_type(self):
    return self.data_frame._get_attribute_type(self.attribute)
