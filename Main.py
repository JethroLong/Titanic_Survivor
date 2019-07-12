import sys
import pandas as pd

from ANN_Logistic.ANN_Algo import ANN
from SVM.SVM_Algo import SVM


def main(model):
    dataset = pd.read_csv()

    if model == "ANN":

        ANN.train_ANN()
    elif model == "SVM":
        SVM.train_SVM()


if __name__ == '__main__':
    main(sys.argv[1])
