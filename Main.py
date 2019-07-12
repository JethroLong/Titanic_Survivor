import sys
import pandas as pd

from ANN_Logistic.ANN_Algo import ANN
from Preprocessing import Preprocessor
from SVM.SVM_Algo import SVM


def main(model):
    preprocessor = Preprocessor()
    train_X, train_y, test_X =preprocessor.preprocess_data()

    if model == "ANN":
        deepNN = ANN()
        parameters = deepNN.train_L_Layer_Model(X=train_X, Y=train_y, layer_dims=deepNN.layer_dims)

        # evaluation
        eval_result = deepNN.predict(X=test_X, y=train_y, parameters=parameters)

        # prediction
        predict_result = deepNN.predict(X=test_X, y=None, parameters=parameters)

    elif model == "SVM":
        SVM.train_SVM()


if __name__ == '__main__':
    main(sys.argv[1])
