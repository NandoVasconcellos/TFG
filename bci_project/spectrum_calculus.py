#!python
#!/usr/bin/env python
from pprint import pprint
import numpy as np
import pandas as pd
from numpy.linalg import matrix_power
import scipy.fftpack as sc
from scipy.fft import fft, ifft
import time
import matplotlib.pyplot as plt

class SpectrumCalculus():

    all = []

    def __init__(self, data, fm, task):

        data = self.spectrum_calculus(data, fm)
        self.data = data
        self.task = task
        SpectrumCalculus.all.append(self)

    def generate_calculated_array(self, data):
        calculated = []
        i = 0
        while i < len(data):
            calculated.append(data[i] + data[i+1])
            i += 2
        return calculated

    def band_filter(self, data, L):
        data_filtered = []
        abs_values_2n_array = np.absolute(data)
        reduced_2n_array = abs_values_2n_array[:, 0:int(L/2+1)]
        df = pd.DataFrame(reduced_2n_array)
        np_power = np.full((1, 257), 2)[0]
        series_power = pd.Series(np_power)
        df_powered = df.pow(series_power, axis=1)
        data_filtered = df_powered.to_numpy()[:, 8:32]
        return data_filtered

    # def traspose_result( data ):

    def calculate_fft(self, data, fm, L):
        NFFT = fm

        temp_F = int(fm)/2*np.linspace(0, 1, int(NFFT/2+1))

        np_array = np.array(data)
        calculated_fft = np.fft.fft(np_array)/L
        return calculated_fft

    def spectrum_calculus(self, data, fm):
        temp_data = []
        temp_F = []
        result = []
        L = len(data[0])

        fft_function = self.calculate_fft(data, fm, L)

        filtered_data = self.band_filter(fft_function, L)

        for row in filtered_data:
            result.append(self.generate_calculated_array(row))

        result = list(np.transpose(result))
        unique_list = []
        for row in result:
            unique_list.extend(row)
        result = list(np.transpose(unique_list))
        
        return result

