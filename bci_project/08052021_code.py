#!python
#!/usr/bin/env python
from scipy.io import loadmat
import scipy.io as sio
from scipy.fft import fft, ifft
import os
import os.path
import shutil
from pprint import pprint
import numpy as np
import pandas as pd
import sys
import gc
import re
import json
import ast
import time


from new_procesor import Processor
from sorter import Sorter
# change directory to receive data
#os.chdir( '/home/fernando/tfg/Fernando_Gaston/codigos_iniciales' )

CONF_COLUMNS = (
    "fm",
    "description",
    "electrodes_configuration",
    "device",
    "device_configuration_code",
    "user_code",
    "num_channels",
    "electrode_position_x",
    "electrode_position_y",
    "electrode_names",
    "register_code",
    "sample_register",
    "tasks",
    "file_name",
)
OPTIONS = [
    {'task1': 122,
     'task2': 123,
     'k': 1, },
    {'task1': 122,
     'task2': 127,
     'k': 1, },
    {'task1': 123,
     'task2': 127,
     'k': 1, },

    {'task1': 122,
     'task2': 123,
     'k': 3, },
    {'task1': 122,
     'task2': 127,
     'k': 3, },
    {'task1': 123,
     'task2': 127,
     'k': 3, },

    {'task1': 122,
     'task2': 123,
     'k': 5, },
    {'task1': 122,
     'task2': 127,
     'k': 5, },
    {'task1': 123,
     'task2': 127,
     'k': 5, },

    {'task1': 122,
     'task2': 123,
     'k': 15, },
    {'task1': 122,
     'task2': 127,
     'k': 15, },
    {'task1': 123,
     'task2': 127,
     'k': 15, }
]
results = {
    '1': {
        '122-123': {
            'result': float,
            'std_deviation': float
        },
        '122-127': {
            'result': float,
            'std_deviation': float
        },
        '123-127': {
            'result': float,
            'std_deviation': float
        }
    },
    '3': {
        '122-123': {
            'result': float,
            'std_deviation': float
        },
        '122-127': {
            'result': float,
            'std_deviation': float
        },
        '123-127': {
            'result': float,
            'std_deviation': float
        }
    },
    '5': {
        '122-123': {
            'result': float,
            'std_deviation': float
        },
        '122-127': {
            'result': float,
            'std_deviation': float
        },
        '123-127': {
            'result': float,
            'std_deviation': float
        }
    },
    '15': {
        '122-123': {
            'result': float,
            'std_deviation': float
        },
        '122-127': {
            'result': float,
            'std_deviation': float
        },
        '123-127': {
            'result': float,
            'std_deviation': float
        }
    },
}


class UserData():
    all = []

    def __init__(self, file):

        file_data = UserData.data_extraction(file)

        # Copy all the data from file to class type data
        self.conf = file_data["conf"]
        self.data = file_data["data"]
        self.task = file_data["task"]
        self.complete = file_data["complete"]
        self.time = file_data["time"]

        # Append already created instance to list
        UserData.all.append(self)

    def contruct_conf_data(jdata):

        # Initialize conf data dictionary and task key as list
        conf = {}
        conf["tasks"] = []

        #         cleaning brakets          #
        jdata = jdata.replace("[[[[[", "[[")
        jdata = jdata.replace("]]]]]", "]]")
        jdata = jdata.replace("[[[", "[")
        jdata = jdata.replace("]]]", "]")
        # Convert from string to list
        data_list = ast.literal_eval(jdata)

        for index, data in enumerate(data_list):
            # clean data set of brakets
            if(isinstance(data[0], list)):
                jdata = json.dumps(data).replace("]", "").replace("[", "")
                data = ast.literal_eval(jdata)
            else:
                data = data[0]
            # Copy data into correct key
            if(index < 12):
                conf[CONF_COLUMNS[index]] = data
            if(index >= 12 and index <= 15):
                task_list = conf["tasks"]
                task_list.append(data)
                conf["tasks"] = task_list
            if(index == 16):
                conf[CONF_COLUMNS[13]] = data

        return conf

    def contruct_time_data(jdata):

        #         cleaning brakets          #
        jdata = jdata.replace("[[[[", "[")
        jdata = jdata.replace("]]]]", "]")
        jdata = jdata.replace("[[", "[")
        jdata = jdata.replace("]]", "]")
        # Convert from string to list
        data_list = ast.literal_eval(jdata)

        return data_list

    def contruct_data_data(data, electrodes):
        data_dict = {}
        for index, electrode in enumerate(electrodes):
            #print( electrode )
            #print( type( data[index] ) )
            data_dict[electrode] = data[index].tolist()
        #print( data_dict )
        return data_dict

    def data_extraction(file):
        # Set whole data file dictionary
        file_data = {}

        # Open mat file into a DataFrame
        mat = sio.loadmat(file)
        df = pd.DataFrame.from_dict(mat['session'][0])

        # loop through all the DataFrame
        for col in df.columns:
            jdata = 0

            if(col != 'data'):
                # clear first braket and json index
                digested = df[col].str[0]
                jdata = digested.to_json()
                jdata = jdata.replace('{"0":', "")
                jdata = jdata.replace('}', "")

            # Digest multiple list data
            if(col == 'conf'):
                jdata = UserData.contruct_conf_data(jdata)
            if(col == 'time'):
                jdata = UserData.contruct_time_data(jdata)
            if(col == 'data'):
                # conver a piece of dataframe into dictionary
                d_data = df[col].to_dict()
                jdata = UserData.contruct_data_data(d_data[0],
                                                    file_data['conf']['electrode_names'])
            if(col == 'task'):
                jdata = json.loads(jdata)

            # Copy all data into a dict
            file_data[col] = jdata

        return file_data


files = [
    '/home/fernando/tfg/Fernando_Gaston/codigos_iniciales/user#0091#20040101#01#reg001.mat',
    '/home/fernando/tfg/Fernando_Gaston/codigos_iniciales/user#0091#20040101#02#reg001.mat',
    '/home/fernando/tfg/Fernando_Gaston/codigos_iniciales/user#0091#20040101#03#reg001.mat',
    '/home/fernando/tfg/Fernando_Gaston/codigos_iniciales/user#0091#20040101#04#reg001.mat'
]

for file in files:
    UserData(file)

start = time.time()
for object in UserData.all:
    test = Processor(object)
    print(len(test.processed_data))

end = time.time()
print(end-start)

print(Processor.all)
start = time.time()


for option in OPTIONS:
    task_comp = str(option['task1'])+'-'+str(option['task2'])
    print(option['k'])
    print(task_comp)
    result = Sorter(Processor.all, UserData.all[0], option)
    results[str(option['k'])][task_comp]['result'] = result.result
    results[str(option['k'])][task_comp]['std_deviation'] = result.std_deviation

pprint(results)
end = time.time()
print(end-start)
