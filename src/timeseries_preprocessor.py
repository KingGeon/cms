import os
import pickle

import pandas as pd
from omegaconf import OmegaConf

from utils import df_divider, rolling_processor, normalizer, \
    not_operating_condition_exterminator


CURRENT_PATH = os.getcwd()
DATA_PATH = os.path.join(CURRENT_PATH, 'dataset', 'raw_data')
SAVE_PATH = os.path.join(CURRENT_PATH, 'dataset', 'preprocessed_data')
conf = OmegaConf.load(f"{CURRENT_PATH}/preprocess_cfg.yaml")


def main():
    print("==================< Data Loading... >==================")
    for item in os.listdir(SAVE_PATH):
        os.remove(os.path.join(SAVE_PATH, item))

    files = os.listdir(DATA_PATH)
    data_list = [pd.read_csv(os.path.join(DATA_PATH, files[i])) for i in range(len(files)) if files[i].endswith('csv')]
    df = pd.concat([data_list[i] for i in range(len(data_list))]).drop_duplicates()

    print("==================< Preprocessing started >==================")
    fault_detected = conf.fault_detected_sensor_list
    fault_not_detected = set(df.MeasureId.value_counts().index) - set(fault_detected)
    fault_info = {'fault': fault_detected, 'normal': fault_not_detected}

    with open(os.path.join(SAVE_PATH, 'sensor_seperation.pickle'), 'wb') as fw:
        pickle.dump(fault_info, fw)

    fault_df_dict = df_divider(df, fault_detected)
    normal_df_dict = df_divider(df, fault_not_detected)

    ###############################HARD_CODED_FAILURE###############################
    df_598 = df[df.MeasureId == 600].reset_index().copy()
    preprocessed_df = rolling_processor(df_598, window_size=168, std_window=36, envelope_window=24).reset_index()
    preprocessed_df.DateTime = pd.to_datetime(preprocessed_df.DateTime)

    outlier_of_12000 = preprocessed_df[preprocessed_df.std36 < 0.04].index

    df_961 = df[df.MeasureId == 961].reset_index().copy()
    preprocessed_df = rolling_processor(df_961, window_size=168, std_window=36, envelope_window=24).reset_index()
    preprocessed_df.DateTime = pd.to_datetime(preprocessed_df.DateTime)

    outlier_of_10000 = preprocessed_df[preprocessed_df.std36 < 0.05].index
    #################################################################################

    for i, fault_df in fault_df_dict.items():
        fault_df_dict[i] = normalizer(not_operating_condition_exterminator(fault_df, outlier_of_10000, outlier_of_12000))
        fault_df_dict[i] = rolling_processor(fault_df_dict[i], window_size=168, std_window=168, envelope_window=48)
        fault_df_dict[i].to_pickle(f'{SAVE_PATH}/fault-{i}.pkl')

    for i, normal_df in normal_df_dict.items():
        normal_df_dict[i] = normalizer(not_operating_condition_exterminator(normal_df, outlier_of_10000, outlier_of_12000))
        normal_df_dict[i] = rolling_processor(normal_df_dict[i], window_size=168, std_window=168, envelope_window=48)
        normal_df_dict[i].to_pickle(f'{SAVE_PATH}/normal-{i}.pkl')

    print("==================< Succesfully terminated >==================")


if __name__ == '__main__':
    main()
