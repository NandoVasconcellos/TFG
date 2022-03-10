#!python
#!/usr/bin/env python
from numpy.lib.function_base import extract
from scipy.io import loadmat
import scipy.io as sio
from scipy.fft import fft, ifft
import os
import os.path
import shutil
from pprint import pprint
import numpy as np
import numpy.matlib as matlib
import pandas as pd
import math
import operator
from statistics import mode, mean


import multiprocessing as mp
from spectrum_calculus import SpectrumCalculus


class Sorter():

    all = []

    def __init__(self, processed_data: list, user_data: object, options: dict):
        digested_data = self.digest_processed_data(
            processed_data, [options['task1'], options['task2']]
        )
        self.data = digested_data
        result = self.data_spinner(options['k'])
        self.result = result['mean']
        self.std_deviation = result['std_deviation']

    def extract_data(self, data: list, tasks: list):
        extracted = []
        for line in data:
            if(line['task'] in tasks):
                extracted.append(line)
        return extracted

    def digest_processed_data(self, data_processed: list, tasks: list):
        extracted_data = []
        for cluster in data_processed:
            extracted_data.append(self.extract_data(
                cluster.processed_data, tasks)
            )
        return extracted_data

    def sorter_knn(self, test_data: list, train_data: list, k: int):
        
        result = []
        
        merged_train_data = [*train_data[0], *train_data[1], *train_data[2]]
        
        df_train = pd.DataFrame.from_records(merged_train_data)
        df_train_unravel = pd.DataFrame()
        
        df_train_unravel[
            [
                str(i)
                for i in range(len(df_train['data'].iloc[0]))
            ]
        ] = pd.DataFrame(df_train.data.to_list(), index=df_train.index)
        
        for data in test_data:
            
            subtracted = data['data']-df_train_unravel
            mathed = np.sqrt(subtracted.pow(2).sum(axis=1))
            df_mathed = mathed.to_frame(name='data')
            df_mathed['task'] = df_train['task'].values
            
            result.append(
                mode(
                    df_mathed
                    .sort_values(by='data')
                    .head(k)['task']
                    .tolist()
                )
            )
            
        return result

    def data_spinner(self, k: int):
        temp = []

        for index, cluster in enumerate(self.data):
            
            train_data = [data for i, data in enumerate(self.data) if i != index]
            
            test_data = self.data[index]
            
            result = self.sorter_knn(test_data, train_data, k)
            
            df_test = pd.DataFrame.from_records(test_data)
            
            prediction = sum(list(result == df_test['task']))
            
            temp.append((prediction/len(result))*100)

        return {
            'std_deviation': np.std(temp, ddof=1),
            'mean': mean(temp)
        }