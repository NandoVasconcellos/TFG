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


class MatFileDigestor():
    all = []

    def __init__(self, file):

        file_data = self.data_extraction(file)

        # Copy all the data from file to class type data
        self.conf = file_data["conf"]
        self.data = file_data["data"]
        self.task = file_data["task"]
        self.complete = file_data["complete"]
        self.time = file_data["time"]

        # Append already created instance to list
        MatFileDigestor.all.append(self)

    def contruct_conf_data(self, jdata):

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

    def contruct_time_data(self, jdata):

        #         cleaning brakets          #
        jdata = jdata.replace("[[[[", "[")
        jdata = jdata.replace("]]]]", "]")
        jdata = jdata.replace("[[", "[")
        jdata = jdata.replace("]]", "]")
        # Convert from string to list
        data_list = ast.literal_eval(jdata)

        return data_list

    def contruct_data_data(self, data, electrodes):
        data_dict = {}
        for index, electrode in enumerate(electrodes):
            #print( electrode )
            #print( type( data[index] ) )
            data_dict[electrode] = data[index].tolist()
        #print( data_dict )
        return data_dict

    def data_extraction(self, file):
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
                jdata = self.contruct_conf_data(jdata)
            if(col == 'time'):
                jdata = self.contruct_time_data(jdata)
            if(col == 'data'):
                # conver a piece of dataframe into dictionary
                d_data = df[col].to_dict()
                jdata = self.contruct_data_data(d_data[0],
                                                    file_data['conf']['electrode_names'])
            if(col == 'task'):
                jdata = json.loads(jdata)

            # Copy all data into a dict
            file_data[col] = jdata

        return file_data
