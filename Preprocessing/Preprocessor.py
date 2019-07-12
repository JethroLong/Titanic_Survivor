import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self):
        self.origin_train_df = pd.read_csv("/dataset/train.csv")
        self.origin_test_df = pd.read_csv("/dataset/test.csv")

    def preprocess_data(self):

        train_X = None
        train_y = None
        test_X = None

        return train_X, train_y, test_X