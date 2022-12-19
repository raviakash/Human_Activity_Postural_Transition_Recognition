# %%
# general packages
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, random_split, DataLoader
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class GenerateHAPTData():
    def __init__(self) -> None:
        pass

    def get_exp_path(self, exp_num):
        raw_data_dir = "HAPT Dataset/RawData"
        for root, _, files in os.walk(raw_data_dir, topdown=False):
            path_list = [os.path.join(root, file) for file in files if "_exp" + exp_num + "_" in file]
        return path_list

    def get_exp_data(self, label_info, index):
        # get the right exp string
        if label_info["exp_num"][index] < 10:
            exp_num = "0" + str(label_info["exp_num"][index])
        else:
            exp_num = str(label_info["exp_num"][index])
        # get clip indeces
        start = label_info["label_start"][index]
        end = label_info["label_end"][index]
        start_indeces = [start]
        window_len = 128
        while True:
            start += 64
            if start + window_len > end:
                break
            start_indeces.append(start)
        # data preparation (concatenate + standardization)
        path_list = self.get_exp_path(exp_num)
        acc_data = pd.read_csv(path_list[0], header=None, delim_whitespace=True)
        gyro_data = pd.read_csv(path_list[1], header=None, delim_whitespace=True)
        concat_data = np.concatenate([acc_data, gyro_data], axis=1)
        concat_data = StandardScaler().fit_transform(concat_data)
        # stack clip data
        stack_data = [concat_data[i:i + window_len] for i in start_indeces]
        stack_data = np.stack(stack_data, axis=0)
        # create corresponding labels
        label = [label_info["act_num"][index]] * len(stack_data)
        label = np.array(label).reshape(-1, 1)
        return stack_data, label

    def run(self, change=1):
        label_path = "HAPT Dataset/RawData/labels.txt"
        label_info = pd.read_csv(label_path, header=None, delim_whitespace=True)
        label_info.columns = ["exp_num", "user_num", "act_num", "label_start", "label_end"]
        data_list = []
        label_list = []
        for index in range(len(label_info)):
            data, label = self.get_exp_data(label_info, index)
            data_list.append(data)
            label_list.append(label)
        X = np.concatenate(data_list, axis=0)
        y = np.concatenate(label_list, axis=0)
        # select label
        mask = np.where(y >= change)[0]
        X = X[mask]
        y = y[mask]
        y = y - change
        # y = y - 1
        y = tf.keras.utils.to_categorical(y)
        return X, y


class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, X, y):
        # store the inputs and outputs
        self.X = X
        self.y = y

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, train_rate):
        n_data = len(self.X)
        train_size = int(n_data * train_rate)
        test_size = n_data - train_size
        return random_split(self, [train_size, test_size])


