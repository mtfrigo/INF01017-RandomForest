import argparse
import random
import pandas as pd

if __name__ == '__main__':
  datasets = ['Benchmark', 'Wine', 'Stroke']

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", type=str, default='Wine', help="The dataset to test. List of available datasets: " + str(datasets))
  parser.add_argument("--num_of_trees", type=int, default=10, help="The number of trees to ensemble in the random forest. The default is 8.")

  args = parser.parse_args()

  if args.dataset is not None:
    if args.dataset in datasets:
      filename = ""
      delimiter = ""
      class_attr = ""
      id_attr = None

      if args.dataset.strip() == "Benchmark":
        filename = "datasets/Benchmark.csv"
        delimiter = ";"
        target_class_attbr = "Joga"

      elif args.dataset.strip() == "Stroke":
        filename = "datasets/Stroke.csv"
        delimiter = ","
        target_class_attbr = "Target_Stroke"
      
      elif args.dataset.strip() == "Wine":
        filename = "datasets/Wine.csv"
        delimiter = ","
        target_class_attbr = "Target_WineType"

      data = pd.read_csv(filename, sep=delimiter)

      print(data)

      # Discretizing the data for numeric values

      # Processing the data with chosen algorithm and parameters
    
    
    else:
      print("The chosen dataset is not supported")
  else:
    print("Unknown error")