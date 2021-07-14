

import argparse
import numpy as np
import tensorflow.keras
import os
import pickle

from load import load_dataset
from util import load

STEP = 256


def truncate(x):
    x_trunc = STEP * int(len(x[0]) / STEP)
    return x[:, :x_trunc]


def predict2(data_path, model_path):
    preproc = load(os.path.dirname(model_path))
    with open(data_path, "rb") as fp:
        x = pickle.load(fp)
    x = preproc.process_x(x)
    x = truncate(x)
    model = tensorflow.keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)
    print(model.summary())
    print(model.layers[-6].get_weights())
    print(model.layers[-6].get_weights()[0].shape)
    print(model.layers[-6].get_weights()[-1].shape)

    output = data_path.strip('.txt') + '_pred.txt'
    with open(output, "wb") as fp:
        pickle.dump(probs, fp)
    return probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to data txt")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs = predict2(args.data_path, args.model_path)
