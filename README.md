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
  dataset             The dataset to test.
  num_of_trees        The number of trees to ensemble in the random forest. The default is 8.

optional arguments:
  -h, --help  show this help message and exit
```