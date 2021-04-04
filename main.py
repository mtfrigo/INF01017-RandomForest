import argparse
import random
from data.handler import DataFrame
from tree.algorithm import DecisionTree
import pandas as pd

if __name__ == '__main__':
  datasets = ['Benchmark', 'Wine', 'Stroke']

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default='Wine', help="The dataset to test. List of available datasets: " + str(datasets))
  parser.add_argument("--num_of_trees", type=int, default=10, help="The number of trees to ensemble in the random forest. The default is 8.")
  parser.add_argument("--attributes_per_division", type=int, default=10, help="The number of attributes to consider on each node division. The default is 10.")

  args = parser.parse_args()

  if args.dataset is not None:
    if args.dataset in datasets:
      filename = ""
      delimiter = ""
      target_class = ""
      ignore_colums = []
      id_attr = None

      if args.dataset.strip() == "Benchmark":
        filename = "datasets/Benchmark.csv"
        delimiter = ";"
        ignore_colums = []
        target_class = "Joga"

      elif args.dataset.strip() == "Stroke":
        filename = "datasets/Stroke.csv"
        delimiter = ","
        ignore_colums = ["id"]
        target_class = "Target_Stroke"
      
      elif args.dataset.strip() == "Wine":
        filename = "datasets/Wine.csv"
        delimiter = ","
        ignore_colums = []
        target_class = "Target_WineType"

      
      data_frame = DataFrame(pd.read_csv(filename, sep=delimiter).drop(ignore_colums, axis=1), target_class)

      # Discretizing the data for numeric values

      attributes_types = {}

      for attribute in data_frame.get_attributes():
        attributes_types[attribute] = data_frame._get_attribute_type(attribute)

      normalized_data_frame = data_frame.normalize()

      # data_frame = data_frame.discretize_by_mean()
      discrete_data_frame = normalized_data_frame.discretize_by_neighborhood()

      tree = DecisionTree(discrete_data_frame, attributes_types, args.attributes_per_division)

      # for i in range(1, 50):
      #   tree.predict_random_sample(normalized_data_frame._data_frame.sample(), target_class)

      tree.print_tree()

    else:
      print("The chosen dataset is not supported")
  else:
    print("Unknown error")