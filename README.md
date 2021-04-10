# INF01017-RandomForest
A Python implementation of the Random Forest algorithm for the INF01017 class

## Requirements
 - Python 3+

## Installation

- Install [pandas](https://pandas.pydata.org/)

## Running the algorithm

To generate the model, use the *main.py* script:

```python
usage: main.py [-h] dataset num_of_trees

Run the Random Forest algorithm.

positional arguments:
  dataset        The dataset to test.
  n_trees        The number of trees to ensemble in the random forest. The default is 10.
  k_folds        The number of folds used in the cross-validation. The default is 10.
  algorithm      The algorithm to be executed. Could be either RandomForest or DecisionTree.
  benchmark      If the execution should follow the benchmark walkthough.
  print_tree     If the tree generated in the DecisionTree algorithm should be printed.
  
optional arguments:
  -h, --help  show this help message and exit
```