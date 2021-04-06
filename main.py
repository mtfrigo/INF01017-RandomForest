import argparse
import random
from data.handler import DataFrame
from tree.algorithm import DecisionTree
from forest.random_forest import RandomForest
import pandas as pd

if __name__ == '__main__':
  datasets = ['Benchmark', 'Wine', 'Stroke', 'HouseVotes', 'WineRecognition']

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
        attributes_types = {
          'Tempo': 'categoric',
          'Temperatura': 'categoric',
          'Umidade': 'categoric',
          'Ventoso': 'categoric',
        }

      elif args.dataset.strip() == "Stroke":
        filename = "datasets/Stroke.csv"
        delimiter = ","
        ignore_colums = ["id"]
        target_class = "Target_Stroke"
        attributes_types = {
          'gender': 'categoric',
          'age': 'numeric',
          'hypertension': 'categoric',
          'heart_disease': 'categoric',
          'ever_married': 'categoric',
          'work_type': 'categoric',
          'Residence_type': 'categoric',
          'avg_glucose_level': 'numeric',
          'bmi': 'numeric',
          'smoking_status': 'categoric',
        }
      
      elif args.dataset.strip() == "Wine":
        filename = "datasets/Wine.csv"
        delimiter = ","
        ignore_colums = []
        target_class = "Target_WineType"
        attributes_types = {
          'alcohol': 'numeric',
          'malicAcid': 'numeric',
          'ash': 'numeric',
          'ashAlcalinity': 'numeric',
          'magnesium': 'numeric',
          'totalPhenols': 'numeric',
          'flavanoids': 'numeric',
          'nonflavonoidsPhenols': 'numeric',
          'proanthocyanins': 'numeric',
          'colorIntensity': 'numeric',
          'Hue': 'numeric',
          'od280Od315': 'numeric',
          'proline': 'numeric'
        }

      elif args.dataset.strip() == "WineRecognition":
        filename = "datasets/WineRecognition.tsv"
        delimiter = "\t"
        ignore_colums = []
        target_class = "target"
        attributes_types = {
          '1': 'numeric',
          '2': 'numeric',
          '3': 'numeric',
          '4': 'numeric',
          '5': 'numeric',
          '6': 'numeric',
          '7': 'numeric',
          '8': 'numeric',
          '9': 'numeric',
          '10': 'numeric',
          '11': 'numeric',
          '12': 'numeric',
          '13': 'numeric'
        }

      elif args.dataset.strip() == "HouseVotes":
        filename = "datasets/HouseVotes.tsv"
        delimiter = "\t"
        ignore_colums = []
        target_class = "target"
        attributes_types = {
          "handicapped-infants": 'categoric',
          "water-project-cost-sharing": 'categoric',
          "adoption-of-the-budget-resolution": 'categoric',
          "physician-fee-freeze": 'categoric',
          "el-salvador-adi": 'categoric',
          "religious-groups-in-schools": 'categoric',
          "anti-satellite-test-ban": 'categoric',
          "aid-to-nicaraguan-contras": 'categoric',
          "mx-missile": 'categoric',
          "immigration": 'categoric',
          "synfuels-corporation-cutback": 'categoric',
          "education-spending": 'categoric',
          "superfund-right-to-sue": 'categoric',
          "crime": 'categoric',
          "duty-free-exports": 'categoric',
          "export-administration-act-south-africa": 'categoric'
        }

      
      data_frame = DataFrame(pd.read_csv(filename, sep=delimiter).drop(ignore_colums, axis=1), attributes_types, target_class)
      data_frame.pre_process()
      # Discretizing the data for numeric values
      # normalized_data_frame = data_frame.normalize()

      # data_frame = data_frame.discretize_by_mean()
      # discrete_data_frame = normalized_data_frame.discretize_by_neighborhood()

      forest = RandomForest(data_frame.normalize(), 10)
      # tree = DecisionTree(data_frame.discretize_by_neighborhood(), args.attributes_per_division)

      # for i in range(1, 50):
      #   tree = DecisionTree(data_frame.discretize_by_neighborhood(), args.attributes_per_division)
      #   for i in range(1, 10):
      #     tree.predict_random_sample(data_frame.normalize()._data_frame.sample(), target_class)
      # tree.print_tree()

    else:
      print("The chosen dataset is not supported")
  else:
    print("Unknown error")