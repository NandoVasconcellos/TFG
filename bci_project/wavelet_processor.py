import pywt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import scipy
from scipy.stats import skew, kurtosis, entropy
from tqdm import tqdm
from collections import Counter
import statistics

class WaveletProcessor():

    all = []

    def __init__(self, session, electrodes):

        self.original_data = { key: [] for key in electrodes }

        self.electrodes = electrodes
        #import pdb; pdb.set_trace()
        generated_data = self.generate_electrodes_data(session.data)
        while True:
            try:
                it_data = next(generated_data)
                self.original_data[it_data[0]] = it_data[1]
            except StopIteration:
                break

        # 8-32Hz data
        self.denoised_data = self.dwt_denoising( self.original_data )
        
        #self.cwt_plot_power_spectrum(self.denoised_data['C4'])
        #import pdb; pdb.set_trace()
        self.window_intervals = self.task_window_intervals( tasks = session.task, fm = session.conf['fm'], mode = 'overlapping', overlapping = 2 )

        self.user = session.conf['user_code']
        self.filename = session.conf['file_name']
        self.electrodes = electrodes

        self.wpd_result_per_electrode = { key: [] for key in electrodes }

        for electrode, data in self.denoised_data.items():
            self.wpd_result_per_electrode[electrode] = self.construct_wpd_data(data,
                                                        self.window_intervals)

        
        self.features = ['MAV','AVP','SD','SKEW','KURT','ZC','MC','entropy','n5','n25','n75','n95','median','mean','std','var','rms','EnergySubBand','PercentageSubBand','Energytot']
        self.wpd_result_sum = { key: pd.DataFrame( columns=self.features ) for key in self.window_intervals }
        
        #import pdb; pdb.set_trace()

        for task in self.window_intervals.keys():
            
            result = pd.DataFrame( columns=self.features )
            data_electrodes = []
            
            for electrode in self.electrodes:
                data_electrodes.append(
                    pd.DataFrame(self.wpd_result_per_electrode[electrode][1][task])
                )
                    
            for col in self.features:
                aux = pd.DataFrame()
                
                for ix,i in enumerate(data_electrodes):
                    
                    if( aux.empty ):
                        aux = pd.DataFrame(i[col].to_list(), index=i.index)
                    else:
                        aux = aux.add( pd.DataFrame(i[col].to_list(), index=i.index) )
                        
                result[col] = (aux/len(data_electrodes)).values.tolist()
                
            self.wpd_result_sum[task] = result

        WaveletProcessor.all.append(self)
        #import pdb; pdb.set_trace()
        #self.cwt_plot_power_spectrum(denoised_data, tasks)

    def generate_electrodes_data(self: object, data: dict) -> iter:
        for electrode in data.keys():
            if(electrode in self.electrodes):
                yield electrode, data[electrode]

    def task_window_intervals(self: object, tasks: list, fm: int, window_interval: int = 5, mode = None, overlapping = None) -> dict:

        head = 0
        tail = 0

        """
            Dict containing as key the task code and as
            value a list of lists of index
        """
        task_intervals = { task: list() for task in set(tasks) }
        
        # Extract groups of consequent tasks
        for ix, task in enumerate(tasks):
            if( ix != 0 ):
                if( task is tasks[ix-1] ):
                    head = ix
                else:
                    if( (head-tail) > 3*fm ):
                        task_intervals[tasks[ix-1]].append( [tail, head] )
                    tail = ix
        else:
            if( (head-tail) > 3*fm ):
                task_intervals[tasks[ix-1]].append( [tail, head] )
        
        # Delete first second of samples to prevent inaccurate data
        for task, data_task in task_intervals.items():
            for ix, indexes in enumerate(data_task):
                if( (indexes[0]+512) < indexes[1] ):
                    data_task[ix][0] = indexes[0]+512
                    print(f"ix: {ix} | indexes: {indexes} | total window time: { (indexes[1]-indexes[0])*1.95/1000 }")

        windows = { task: list() for task in set(tasks) }
        if( mode == 'overlapping' ):
            if( not overlapping ):
                overlapping = fm
            # Divide the vectors into equal windows of X seconds
            for task, data_task in task_intervals.items():
                for ix, indexes in enumerate(data_task):
                    aux = indexes[0]
                    while(aux < indexes[1]):
                        if( (aux+(window_interval*fm)-aux) == window_interval*fm ):
                            if(aux == indexes[0] ):
                                windows[task].append( [aux, aux+(window_interval*fm)] )
                            else:
                                windows[task].append( [aux-overlapping, aux+(window_interval*fm)] )
                            aux = aux+(window_interval*fm)
                        else:
                            break
        else:
            # Divide the vectors into equal windows of X seconds
            for task, data_task in task_intervals.items():
                for ix, indexes in enumerate(data_task):
                    aux = indexes[0]
                    while(aux < indexes[1]):
                        if( (aux+(window_interval*fm)-aux) == window_interval*fm ):
                            windows[task].append( [aux, aux+(window_interval*fm)] )
                            aux = aux+(window_interval*fm)
                        else:
                            break

        return windows

    def construct_wpd_data(self, data, tasks_intervals):

        features = { i : list() for i in tasks_intervals.keys() }
        data_tree = { i : list() for i in tasks_intervals.keys() }

        for task in tasks_intervals:            
            for intervals in tqdm(tasks_intervals[task]):
                result = self.wavelet_packet_decomposition_tree(data[intervals[0]:intervals[1]], 6, 'extremes')
                data_tree[task].append(result[0])
                features[task].append(result[1])
        return [data_tree, features]

    """
        The resulting array will be:
            mode is bottom:
                # 2**1 (               [],                             []              )
                # 2**2 (      [],             [],             [],             []       )
                # 2**3 (  [],     [],     [],     [],     [],     [],     [],     []   )
                # 2**4 ([X],[X],[X],[X],[X],[X],[X],[X],[X],[X],[X],[X],[X],[X],[X],[X])
                # 2**n = coeffs | 
                                |-> 1/2 coeff of aproximations
                                |-> 1/2 coeff of details
            mode is extremes:
                # 2**1 (          [X],                     [X]           )
                # 2**2 (    [X],         [],         [],         [X]     )
                # 2**3 ( [X],   [],   [],   [],   [],   [],   [],   [X]  )
                # 2**4 ([X],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[X])
                # 2**n = coeffs | 
                                |-> 1/2 coeff of aproximations
                                |-> 1/2 coeff of details
    """
    def wavelet_packet_decomposition_tree(self,
                                            data,
                                            levels,
                                            mode):
        features = []

        data_tree = [
            [
                pywt.downcoef( part='a',data=data,wavelet='sym9' ),
                pywt.downcoef( part='d',data=data,wavelet='sym9' )
            ]
        ]

        if(levels > 1):

            for level in range(1, levels):
                data_tree.append([])
                for i in range(0, 2**level):
                    data_tree[level].append( 
                        pywt.downcoef( part='a',data=data_tree[level-1][i],wavelet='sym9' )
                    )
                    data_tree[level].append( 
                        pywt.downcoef( part='d',data=data_tree[level-1][i],wavelet='sym9' )
                    )
        if( mode == 'extremes'):
            coeffs_to_extract = []
            for i in range(levels):
                coeffs_to_extract.append( data_tree[i][0] )
                coeffs_to_extract.append( data_tree[i][-1] )
                
            features = self.feature_extraction(coeffs_to_extract)
        if( mode == 'bottom' ):
            features = self.feature_extraction(data_tree[levels-1])
        return data_tree, features


    def feature_extraction(self, data_tree):
        # add 'RMAV': [] to features array
        features = {
            'MAV': [],
            'AVP': [],
            'SD': [],
            'SKEW': [],
            'KURT': [],
            'ZC': [],
            'MC': [],
            'entropy': [],
            'n5': [],
            'n25': [],
            'n75': [],
            'n95': [],
            'median': [],
            'mean': [],
            'std': [],
            'var': [],
            'rms': [],
            'EnergySubBand': [],
            'PercentageSubBand': [],
            'Energytot': []
        }
        
        for ix, sub_band in enumerate(data_tree):
            mav = self.mean_absolute_value(sub_band)
            features['MAV'].append( mav )
            features['AVP'].append( self.average_power(sub_band) )
            features['SD'].append( self.standard_deviation(sub_band) )
            features['SKEW'].append( self.skewness_sub_band(sub_band) )
            features['KURT'].append( self.kurtosis_sub_band(sub_band) )            
            no_zero_crossings, no_mean_crossings = self.calculate_crossings(sub_band)
            features['ZC'].append( no_zero_crossings )
            features['MC'].append( no_mean_crossings )            
            features['entropy'].append( self.calculate_entropy(sub_band) )            
            n5, n25, n75, n95, median, mean, std, var, rms = self.calculate_statistics(sub_band)
            features['n5'].append(n5)
            features['n25'].append(n25)
            features['n75'].append(n75)
            features['n95'].append(n95)
            features['median'].append(median)
            features['mean'].append(mean)
            features['std'].append(std)
            features['var'].append(var)
            features['rms'].append(rms)
        
        temp = self.subband_energy_distribution(data_tree)
        features['EnergySubBand'] = temp['EnergySubBand']
        features['PercentageSubBand'] =temp['PercentageSubBand']
        features['Energytot'] = temp['Energytot']
        return features

    def mean_absolute_value(self, data):
        return np.sum(np.absolute(data))/len(data)

    def calculate_crossings(self, data):
        zero_crossing_indices = np.nonzero(np.diff(np.array(data) > 0))[0]
        no_zero_crossings = len(zero_crossing_indices)
        mean_crossing_indices = np.nonzero(np.diff(np.array(data) > np.nanmean(data)))[0]
        no_mean_crossings = len(mean_crossing_indices)
        return no_zero_crossings, no_mean_crossings

    def standard_deviation(self, data):
        return np.std(data)

    def average_power(self, data):
        return np.sum(
                    np.power(
                        np.absolute(data),
                        2
                    )
                )/len(data)

    def skewness_sub_band(self, data):
        return skew( data )

    def kurtosis_sub_band(self, data):
        return kurtosis(data, fisher=True)

    def calculate_entropy(self, data):
        counter_values = Counter(data).most_common()
        probabilities = [elem[1]/len(data) for elem in counter_values]
        return entropy(probabilities)

    def calculate_statistics(self, data):
        n5 = np.nanpercentile(data, 5)
        n25 = np.nanpercentile(data, 25)
        n75 = np.nanpercentile(data, 75)
        n95 = np.nanpercentile(data, 95)
        median = np.nanpercentile(data, 50)
        mean = np.nanmean(data)
        std = np.nanstd(data)
        var = np.nanvar(data)
        rms = np.nanmean(np.sqrt(data**2))
        return n5, n25, n75, n95, median, mean, std, var, rms

    def subband_energy_distribution(self, data):

        Etot = 0

        ESubBand = []

        for ix, sub_band in enumerate(data):
            ESubBand.append(
                np.sum(np.power(np.absolute(sub_band),2))
            )
            Etot += ESubBand[ix]
        
        percSubBand = []
        for ix, sub_band in enumerate(data):
            percSubBand.append( 
                round(
                    (ESubBand[ix]/Etot)*100,
                    4
                )
            )

        return {
            'EnergySubBand':ESubBand,
            'PercentageSubBand':percSubBand,
            'Energytot':Etot}

    def cwt_plot_power_spectrum(self,
                                denoised_data,
                                task=None,
                                waveletname = 'morl', 
                                cmap = 'YlGn', 
                                title = 'Wavelet Transform (Power Spectrum) of signal', 
                                ylabel = 'Scale', 
                                xlabel = 'Time'):
        tail = 0
        head = 0

        for ix, t in enumerate(task):
            if( ix > 0):
                if( t == task[ix-1] ):
                    head += 1
                elif( head-tail > 0 ):
                    fig, axs = plt.subplots(nrows=4, ncols=1)
                    # signal = denoised_data[0][tail:head]
                    # scales = np.arange(1,128)
                    coeffs, freqs = pywt.cwt(denoised_data[0][tail:head],np.arange(1,128), waveletname, 0.01)
                    
                    #coeffs, freqs = pywt.cwt(,np.arange(1,128), waveletname, 0.01)
                    
        axs[0].plot( denoised_data[0][tail:head] )
        axs[1].plot( coeffs )
        axs[2].plot( freqs )
        axs[3].imshow(coeffs, cmap = 'viridis', aspect = 'auto')
        axs[3].spines['right'].set_visible(False)
        axs[3].spines['top'].set_visible(False)
        tail = head
        plt.tight_layout()
        plt.show()

    def dwt_denoising(self, data):
        result = { key: [] for key,i in data.items() }

        #import pdb; pdb.set_trace()

        for electrode in data:
            wavelet_type = 'db8'
            DWTcoeffs = pywt.wavedec( data[electrode],
                                     'db8',
                                     mode='smooth',
                                     level=8,
                                     axis=-1)

            print("Filtering 8-32Hz")
            DWTcoeffs[8] = np.zeros_like(DWTcoeffs[8])
            DWTcoeffs[7] = np.zeros_like(DWTcoeffs[7])
            DWTcoeffs[6] = np.zeros_like(DWTcoeffs[6])
            DWTcoeffs[5] = np.zeros_like(DWTcoeffs[5])
            
            DWTcoeffs[2] = np.zeros_like(DWTcoeffs[2])
            DWTcoeffs[1] = np.zeros_like(DWTcoeffs[1])
            DWTcoeffs[0] = np.zeros_like(DWTcoeffs[0])

            ### filtered_data_dwt
            result[electrode] = pywt.waverec(DWTcoeffs,
                                            wavelet_type,
                                            mode='smooth',
                                            axis=-1)
        import pdb; pdb.set_trace()
        return result

    def cwt_freq_calculus(self, denoised_data):
    
        scales = np.arange(1,31)

        coefs, freqs = pywt.cwt(denoised_data[0], scales, 'mexh')

        coef, freqs=pywt.cwt(denoised_data[0],
        np.arange(1,20),
        'mexh',
        2)
        plt.imshow( coefs,
                    extent=[1, 128, 1, 128],
                    cmap='Blues',
                    #aspect='auto',
                    vmax=abs(coefs).max(),
                    vmin=-abs(coefs).max())  
        plt.show() # doctest: +SKIP

    def wavelet_packet_decomposition(self,
                                    data,
                                    levels):
        
        aprox = {}
        details = {}
        for level in range(1, levels+1):
            # APROXIMATION:
            aproximation = pywt.downcoef( part='a',data=data,wavelet='db4' )
            aprox[f"A{level}"] = aproximation
            # DETAIL:
            detail = pywt.downcoef(  part='d',
                                data=data,
                                wavelet='db4',
                                level=level )
            details[f"D{level}"] = detail

        return aprox | details
