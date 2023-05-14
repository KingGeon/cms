import os
import argparse
import pickle
import datetime

import pandas as pd
import matplotlib.pyplot as plt
#from prophet import Prophet
from omegaconf import OmegaConf


CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, 'dataset', 'preprocessed_data')
SAVE_PATH = os.path.join(CURRENT_PATH, 'pictures')

conf = OmegaConf.load(f"{CURRENT_PATH}/preprocess_cfg.yaml")
cutoff_train = conf.cutoff_train
cutoff_test = conf.cutoff_test

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--target', dest='target', action='store')
parser.add_argument('-s', '--sensor', dest='sensor', action='store')
args = parser.parse_args()

def main():
    with open(os.path.join(DATA_PATH, 'sensor_seperation.pickle'), 'rb') as fr:
        criteria = pickle.load(fr)
        type = 'normal' if args.sensor in criteria['fault'] else 'fault'

    data = pd.read_pickle(os.path.join(DATA_PATH, f'{type}-{args.sensor}.pkl')).sort_values(by='DateTime')
    data.DateTime = pd.to_datetime(data.DateTime)
    print(data)
    data = data.rename(columns={'DateTime': 'ds', args.target: 'y'})
    train = data[data.ds > cutoff_train]
    train = train[train.ds < cutoff_test]
    test = data[data.ds > cutoff_test]
    #train_dt = list(map(int, cutoff_train.split('-')))
    #test_dt = list(map(int, cutoff_test.split('-')))

    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=14)
    forecast = m.predict(future)

    plt.figure(figsize=(12, 4))
    m.plot(forecast, figsize=(12, 4))
    plt.plot(test.ds, test.y, color='r')
    plt.title("Upward Trend Sensing", fontsize=36)
    plt.savefig(os.path.join(SAVE_PATH, 'prediction.png'))
    plt.show()
    return forecast


if __name__ == '__main__':
    main()
