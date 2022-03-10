#!python
#!/usr/bin/env python
import os.path
from pprint import pprint
import time
import argparse
from os import listdir, walk
from os.path import isfile, join

from sorter import Sorter
from matfile_digestor import MatFileDigestor

from wavelet_processor import WaveletProcessor
#FFT feature extractor
from new_procesor import Processor

class Manager():

    files = []

    def get_data_from_matfiles(self, files):
        data_obj = []
        for file in files:
            complete_path = self.folder+"/"+file
            data_obj.append( MatFileDigestor(complete_path) )
        return data_obj

    def process_data(self, mat_data, feature_extractor):

        feature_extractor_class = {
            'fft': 'Processor',
            'wavelet': 'WaveletProcessor'
        }

        processed_data = []

        for preprocess_data in mat_data:
            processed_data.append(
                eval( f"{feature_extractor_class[feature_extractor]}(preprocess_data, self.electrodes)" )
                )
        #import pdb; pdb.set_trace()
        return processed_data

    def build_options(self, tasks, k):
        options = []
        for index, task in enumerate(tasks):
            
            if( index != len(tasks)-1 ):
                for i in range(index+1, len(tasks)-index ):
                    
                    options.append({
                        'k': k,
                        'task1': task,
                        'task2': tasks[index+i]
                    })
            else:
                options.append({
                    'k': k,
                    'task1': task,
                    'task2': tasks[index-1]
                })
        return options

    def get_rates(self, options, k):

        rates = {
            str(k): {}
        }

        for option in options:

            task_comp = str(option['task1'])+'-'+str(option['task2'])
            result = Sorter(Processor.all, MatFileDigestor.all[0], option)

            rates[str(option['k'])][task_comp] = {
                'result': float,
                'std_deviation': float
            }

            rates[str(option['k'])][task_comp]['result'] = result.result
            rates[str(option['k'])][task_comp]['std_deviation'] = result.std_deviation

        return rates

    def k_cicle(self):
        rates = {}
        if( isinstance(self.k, list) ):
            for single_k in self.k:
                options = self.build_options(self.tasks, single_k)
                pprint(options)
                rates[single_k] = self.get_rates(options, single_k)
        else:
            options = self.build_options(self.tasks, self.k)
        pprint(rates)
        return True

    def check_from_folder(self, path_folder: str, user:str ):
        f = []
        for (dirpath, dirnames, filenames) in walk(path_folder):
            for file in filenames:
                if( user in file ):
                    f.append(file)
            break
        return f

    # Creation control
    def __new__(cls, *args, **kwargs):
        files = Manager.check_from_folder(cls, args[0]['folder'], args[0]['user'])
        if( files ):
            args[0]['files'] = files
            obj = super(Manager, cls).__new__(cls)
            return obj
        else:
            return f"Object not created: no file in '{args[0]['folder']}'' for {args[0]['user']} user"

    # init control
    def __init__(self, args: dict):
        self.files = args['files']
        self.user = args['user']
        self.folder = args['folder']
        self.k = args['k']
        self.tasks = args['tasks']
        self.electrodes = args['electrodes']
        self.mat_data = self.get_data_from_matfiles(self.files)

        pprint(self.mat_data[0].conf)
        
        self.data_processed = self.process_data(self.mat_data, args['processor'])
        import pdb; pdb.set_trace()
        #self.processed_data = self.fft_process_data(self.mat_data)
        self.k_cicle()
        



if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description='This app will extract features from your matlab files',
        epilog='copyright Fernando Vasconcellos'
    )

    # Required positional argument
    parser.add_argument('--folder', type= str, required=True,
                        help='Complete path to folder where there are all files from a user')
                        
    # Required positional argument
    parser.add_argument('--user', type=str, required=True,
                        help='Name of user existing in the file name. E.g. 0091, user#0091...')

    # Required positional argument
    parser.add_argument('--k', type=int, required=True, nargs="+",
                        help='Must be a list of Ks')

    # Required positional argument
    parser.add_argument('--tasks', type=int, required=True, nargs="+",
                        help='Must be a list of tasks that you want to calculate')

    # Required positional argument
    parser.add_argument('--electrodes', type=str, required=True, nargs="+",
                        help='Must be a list of electrodes. E.g. C3 CZ C4 CP1 CP2 P3 PZ P4')

    # Required positional argument
    parser.add_argument('--processor', type=str, required=True,
                        help='Choose between FFT, Wavelet')

    #convert args to dictionary
    dict_args = dict()
    for arg in parser.parse_args()._get_kwargs():
        dict_args[arg[0]] = arg[1]

    test = Manager( dict_args )