#!python
#!/usr/bin/env python
import numpy as np
from spectrum_calculus import SpectrumCalculus

ELECTRODES = [
    'C3', 'CZ', 'C4', 'CP1', 'CP2', 'P3', 'PZ', 'P4'
]


class Processor():

    all = []

    def __init__(self, session):

        data = self.process_data(session.data,
                                 session.task,
                                 session.conf)
        self.user = session.conf['user_code']
        self.filename = session.conf['file_name']
        self.processed_data = data
        # Append already created instance to list
        Processor.all.append(self)

    def generate_electrodes_data(self, data):

        for electrode in data.keys():
            if(electrode in ELECTRODES):
                yield data[electrode]

    def generate_windowed_data(self, data, head, tail):
        for record in data:
            yield record[head:tail-1]

    def check_unique(self, data_list):
        np_array = np.array(data_list)
        unique = np.unique(np_array)
        return unique[0].item() if(len(unique) == 1) else False

    def process_data(self, data, task, conf):

        signal = []
        fm = conf['fm']
        window = int(fm)
        avance = fm/4
        data_result = []

        generated_data = self.generate_electrodes_data(data)
        while True:
            try:
                it_data = next(generated_data)
                signal.append(it_data)
            except StopIteration:
                break

        tail_init = 1
        tail_end = 1
        size = len(data['C3'])
        for index in range(size):
            tail_end = tail_init + window + 1
            if(tail_end > size):
                break

            unique_task = self.check_unique(
                task[int(tail_init):int(tail_end)-1]
            )
            if(unique_task):
                windowed_data = []
                generated_data = self.generate_windowed_data(signal,
                                                             int(tail_init),
                                                             int(tail_end))
                while True:
                    try:
                        it_data = next(generated_data)
                        windowed_data.append(it_data)
                    except StopIteration:
                        break
                result = SpectrumCalculus(windowed_data, fm, unique_task)
                aux = {
                    'data': result.data,
                    'task': result.task
                }
                data_result.append(aux)
            tail_init += avance
        return data_result
