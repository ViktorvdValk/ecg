

import argparse
import numpy as np
import tensorflow.keras
import os
import pickle

from load import load_dataset
from util import load

def predict(data_json, model_path):
    preproc = load(os.path.dirname(model_path))
    dataset = load_dataset(data_json)
    x, y = preproc.process(*dataset)
    model = tensorflow.keras.models.load_model(model_path)
    probs = model.predict(x, verbose=1)

    output = data_json.strip('.json') + '_pred.txt'
    with open(output, "wb") as fp:
        pickle.dump(probs, fp)
    return probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_json", help="path to data json")
    parser.add_argument("model_path", help="path to model")
    args = parser.parse_args()
    probs = predict(args.data_json, args.model_path)
    print(probs)
