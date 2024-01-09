import numpy as np
import os
import pandas as pd
from pyDOE2 import *


# Latin hypercube sampling, where n is the number of samples for each parameter
def lhs_sampling(n, param_range):
    param_range = np.array(param_range)
    lhd = lhs(len(param_range), samples=n, random_state=1)
    return lhd * (param_range[:, 1] - param_range[:, 0]) + param_range[:, 0]


def get_dataset(train_data_path, test_data_path, meas_num=1):
    train_data = pd.read_csv(train_data_path).values
    test_data = pd.read_csv(test_data_path).values
    # Load the dataset based on the number of intervention measures
    train_x, train_y = train_data[:, 0:meas_num * 2], train_data[:, meas_num * 2:]
    test_x, test_y = test_data[:, 0:meas_num * 2], test_data[:, meas_num * 2:]
    return train_x, train_y, test_x, test_y


def save(data, path, columns=None):
    folder_path = "./DataFolder"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    data_path = "./DataFolder/DatasetFile"
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    model_path = "./DataFolder/ModelFile"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    result_path = "./DataFolder/ResultFile"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    figure_path = "./DataFolder/FigureFile"
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    data = pd.DataFrame(data, columns=columns)
    data.to_csv(path, index=False)


def get_data_column(intervene_meas):
    meas = str(intervene_meas)
    column_dict = {
        '1': ['beta1', 'beta2'],
        '2': ['gamma1', 'gamma2'],
        '3': ['epsilon1', 'epsilon2'],
        '12': ['beta1', 'beta2', 'gamma1', 'gamma2'],
        '13': ['beta1', 'beta2', 'epsilon1', 'epsilon2'],
        '23': ['gamma1', 'gamma2', 'epsilon1', 'epsilon2'],
        '123': ['beta1', 'beta2', 'gamma1', 'gamma2', 'epsilon1', 'epsilon2'],
    }
    column = []
    for i in range(len(meas)):
        column = column + column_dict[meas[i]]
    return column
