
wpd_data = WaveletProcessor.all

result.append(mode(df_mathed.sort_values(by='data').head(k)['task'].tolist()))



df_122 = pd.DataFrame({'data': mav[122], 'task':122})
df_123 = pd.DataFrame({'data': mav[123], 'task':123})
df_127 = pd.DataFrame({'data': mav[127], 'task':127})

df_mavs = pd.concat([df_122, df_123, df_127])
df_mavs = df_mavs.reset_index(drop=True)
df_mavs = df_mavs.drop('index', axis=1)

df_train = pd.DataFrame(df_mavs.data.to_list(), index=df_mavs.index)


mav = {122:[],123:[],127:[]}


for user_data in wpd_data:
    for i in user_data.features:
        for j in wpd_data[0].features[i]:
            mav[i].append(j['MAV'])
            
            
result = []
k = 15


for ix, data in enumerate( df_127['data'] ):
    df_122 = pd.DataFrame({'data': mav[122], 'task':122})
    df_123 = pd.DataFrame({'data': mav[123], 'task':123})
    df_127 = pd.DataFrame({'data': mav[127], 'task':127})
    df_mavs = pd.concat([df_122, df_123, df_127])
    df_mavs = df_mavs.reset_index(drop=True)
    #df_mavs = df_mavs.drop('index', axis=1)
    df_train = pd.DataFrame(df_mavs.data.to_list(), index=df_mavs.index)
    subtracted = data - df_train
    mathed = np.sqrt(subtracted.pow(2).sum(axis=1))
    df_mathed = mathed.to_frame(name='data')
    df_mathed['task'] = df_mavs['task'].values
    result.append( mode(df_mathed.sort_values(by='data').head(k)['task'].tolist()) )
    
    
df_train = pd.DataFrame(df_mavs.data.to_list(), index=df_mavs.index)
k = 4
    
    
subtracted = df_122['data'][0] - df_train





df_122 = pd.DataFrame({'data': pd.concat( wpd_data[0].wpd_result_sum[122].AVP,  ), 'task':122})
df_123 = pd.DataFrame({'data': wpd_data.wpd_result_sum[123].AVP, 'task':123})
df_127 = pd.DataFrame({'data': wpd_data.wpd_result_sum[127].AVP, 'task':127})

result = {122:[],123:[],127:[]}

for ix, data in enumerate( df_122 ):
    df_mavs = pd.concat([df_122.drop(ix), df_123, df_127])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))
    task_index = prediction[1].tolist()[0][0]
    result[122].append(df_mavs.iloc[task_index].task)

for ix, data in enumerate( df_123.data ):
    df_mavs = pd.concat([df_122, df_123.drop(ix), df_127])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))
    task_index = prediction[1].tolist()[0][0]
    result[123].append(df_mavs.iloc[task_index].task)
    
for ix, data in enumerate( df_127.data ):
    df_mavs = pd.concat([df_122, df_123, df_127.drop(ix)])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))
    task_index = prediction[1].tolist()[0][0]
    result[127].append(df_mavs.iloc[task_index].task)

train_data = pd.concat([df_122.drop(0), df_123, df_127])


eep = {122:[],123:[],127:[]}

for user_data in wpd_data:
  print(user_data.filename)
  for task in user_data.features:
    print(task)
    for fragment in user_data.features[task]:
        eep[task].append(fragment['EEP']['EnergySubBand'])
        
        
kurt = {122:[],123:[],127:[]}

for user_data in wpd_data:
  print(user_data.filename)
  for task in user_data.features:
    print(task)
    for fragment in user_data.features[task]:
        kurt[task].append(fragment['KURT'])


Counter().keys()
Counter().values()

neigh = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')

res = []

for ix, data in enumerate( df_122.data ):
    df_mavs = pd.concat([df_122.drop(ix), df_123, df_127])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(
        np.array(df_mavs.T.iloc[0]).reshape(-1,1),
        data)
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))[1][0]
    tasks = []
    for i in prediction:
        tasks.append(df_mavs.iloc[i].task)
    res.append(mode(tasks))
    
    
    
[6709.62146637114, 1162.5865814339993, 0.33289175774836977, 0.5096428050727232, 0.018688852202408776, 0.005071399928071504, 0.02532387353230107, 0.020990860932839403, 4.2788434034308234e-05, 1.56281590301985e-05, 0.00022913229063180866, 0.0011738232106407991, 0.008978549139216611, 0.004965107754887358, 0.0066851164591148204, 0.021439862273658145]


df_122 = pd.DataFrame({'data': avp[122], 'task':122})
df_123 = pd.DataFrame({'data': avp[123], 'task':123})
df_127 = pd.DataFrame({'data': avp[127], 'task':127})


import code; code.interact(local=vars())
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics
from wavelet_processor import WaveletProcessor
import pandas as pd
import numpy as np
from statistics import mode, mean
from collections import Counter
wpd_data = WaveletProcessor.all

features = {'MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand','Energytot'}
results = {
    'MAV': [],
    'AVP': [],
    'SD': [],            
    'SKEW': [],
    'KURT': [],
    'EnergySubBand': [],
    'PercentageSubBand': [],
    'Energytot': []
}

df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
df_tot_122.reset_index(drop=True, inplace=True)
df_tot_123.reset_index(drop=True, inplace=True)
df_tot_127.reset_index(drop=True, inplace=True)
df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})


for feature in features:
    df = pd.DataFrame()
    df = pd.concat([df_122, df_127]) #, df_127
    df.reset_index(drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.1,random_state=109) # 70% training and 10% test
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X_train.tolist())
    distances, indices = nbrs.kneighbors(X_test.tolist())
    target = y_test.tolist()
    count = 0
    results[feature] = {'mode': [], 'target': []}
    for ix, i in enumerate(indices):
        #if( ix != 0 ):
        results[feature]['mode'].append( mode( [ y_train.iloc[y] for y in i ] ) )
        results[feature]['target'].append( target[ix] )
        if( mode( [ y_train.iloc[y] for y in i ] ) == target[ix] ):
            count += 1
    results[feature].update({'rate': (count*100/len(y_test)), 'y_test': y_test, 'X_test': X_test})


df_test = pd.DataFrame(columns=['MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand','Energytot','target'])
df_test.MAV = results['MAV']['mode']
df_test.AVP = results['AVP']['mode']
df_test.SD  = results['SD']['mode']
df_test.SKEW  = results['SKEW']['mode']
df_test.KURT  = results['KURT']['mode']
df_test.EnergySubBand  = results['EnergySubBand']['mode']
df_test.PercentageSubBand  = results['PercentageSubBand']['mode']
df_test.Energytot  = results['Energytot']['mode']

df_test.target = results['Energytot']['target']

col = "target"
df1 = df_test.loc[:, df_test.columns != col]
df_test['mode'] = df1.mode(axis=1)[0].astype('int32')
df_test[['target','mode']]


count = 0
for row in df_test[['target', 'mode']].to_records():
    if(row[1]==row[2]):
        count += 1


(count/len(df_test['target']))*100




result = {122:[],123:[],127:[]}

for ix, data in enumerate( df_122.data ):
    df_mavs = pd.concat([df_122.drop(ix), df_123])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))[1][0]
    tasks = []
    for i in prediction:
        tasks.append(df_mavs.iloc[i].task)
    result[122].append(mode(tasks))

for ix, data in enumerate( df_122.data ):
    df_mavs = pd.concat([df_122.drop(ix), df_127])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))[1][0]
    tasks = []
    for i in prediction:
        tasks.append(df_mavs.iloc[i].task)
    result[122].append(mode(tasks))



for ix, data in enumerate( df_122.data ):
    df_mavs = pd.concat([df_122.drop(ix), df_123, df_127])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))[1][0]
    tasks = []
    for i in prediction:
        tasks.append(df_mavs.iloc[i].task)
    result[122].append(mode(tasks))

for ix, data in enumerate( df_123.data ):
    df_mavs = pd.concat([df_122, df_123.drop(ix), df_127])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))[1][0]
    tasks = []
    for i in prediction:
        tasks.append(df_mavs.iloc[i].task)
    result[123].append(mode(tasks))
    
for ix, data in enumerate( df_127.data ):
    df_mavs = pd.concat([df_122, df_123, df_127.drop(ix)])
    df_mavs = df_mavs.reset_index(drop=True)
    neigh.fit(df_mavs.data.tolist())
    prediction = neigh.kneighbors(np.array(data).reshape(1,-1))[1][0]
    tasks = []
    for i in prediction:
        tasks.append(df_mavs.iloc[i].task)
    result[127].append(mode(tasks))
    
    
    
    
    
    
    
    
    
    
"""
# 2**1 (         [],                   []                )
# 2**2 (   [],         [],         [],         []        )
# 2**3 ( [],   [],   [],   [],   [],   [],   [],   []    )
# 2**4 ([],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]  ) <- from here we calculate the features
# 2**n = coeffs | 
                |-> 1/2 coeff of aproximations
                |-> 1/2 coeff of details
"""
_______________________________________________________________________________________
MAV:
{122: [127, 123, 123, 122, 122, 122, 122, 122, 122, 123, 122, 127, 122, 127, 122, 122, 122, 122, 122, 123, 123, 122, 122, 122, 122, 127, 122, 122, 127, 127, 127, 123, 123, 127, 122, 122, 127, 127, 123, 122, 122, 123, 127, 127, 122, 122, 122, 123, 123, 122, 123, 123, 127, 123, 123, 127, 122, 122, 123, 122, 122, 122, 122, 123, 122, 122, 122, 122, 123, 122, 122, 122, 123, 123, 127, 127, 123, 127, 127, 127, 123, 127, 122, 122, 127, 122, 123, 122, 127], 123: [123, 122, 123, 122, 122, 127, 127, 127, 127, 122, 123, 123, 123, 122, 123, 122, 122, 122, 123, 127, 127, 127, 123, 127, 123, 123, 123, 123, 123, 123, 127, 127, 122, 123, 127, 123, 123, 127, 123, 127, 122, 127, 122, 123, 122, 123, 123, 123, 123, 123, 122, 122, 127, 127, 127, 127, 123, 122, 127, 127, 123, 122, 122, 123, 123, 123, 127, 123, 127, 127, 123, 127, 127, 127, 122, 122, 122, 123, 127, 127, 127, 127, 123, 123, 123, 127, 127, 122, 122, 122, 122], 127: [123, 127, 122, 127, 123, 123, 123, 127, 123, 122, 123, 122, 122, 122, 123, 122, 127, 127, 127, 127, 123, 123, 127, 127, 123, 123, 123, 122, 122, 127, 122, 123, 122, 123, 123, 127, 123, 123, 123, 122, 122, 122, 127, 123, 123, 122, 123, 123, 123, 123, 127, 123, 123, 127, 127, 123, 122, 123, 122, 127, 127, 123, 127, 127, 127, 127, 122, 123, 122, 122, 123, 123, 123, 127, 127, 127, 127, 122, 122, 127, 123, 127, 122, 123, 122, 123, 123, 123, 122]}
>>> mode(result[122])
122
>>> mode(result[123])
123
>>> mode(result[127])
123
_______________________________________________________________________________________
AVP:
{122: [127, 123, 123, 122, 122, 122, 122, 122, 122, 123, 122, 127, 122, 127, 122, 122, 123, 122, 122, 123, 123, 122, 122, 122, 122, 127, 122, 122, 127, 127, 127, 123, 123, 127, 122, 122, 127, 127, 123, 122, 122, 123, 127, 127, 122, 122, 122, 123, 123, 122, 123, 123, 127, 123, 123, 127, 122, 122, 123, 123, 122, 123, 127, 123, 122, 122, 122, 122, 123, 122, 122, 122, 123, 123, 127, 127, 123, 127, 127, 127, 123, 127, 122, 122, 127, 122, 123, 122, 127], 123: [123, 122, 123, 122, 122, 127, 127, 127, 127, 127, 123, 122, 123, 122, 123, 122, 122, 122, 123, 127, 127, 127, 123, 127, 123, 123, 123, 123, 123, 123, 127, 127, 122, 123, 127, 123, 123, 127, 123, 127, 122, 127, 122, 123, 122, 127, 123, 123, 122, 123, 122, 122, 127, 127, 127, 127, 123, 122, 127, 127, 123, 122, 122, 123, 122, 122, 127, 123, 127, 127, 123, 127, 127, 127, 122, 122, 122, 123, 127, 127, 127, 127, 123, 123, 123, 127, 127, 122, 122, 122, 122], 127: [123, 127, 122, 127, 123, 123, 123, 127, 123, 123, 123, 122, 122, 122, 123, 122, 127, 127, 127, 127, 127, 123, 127, 127, 123, 123, 123, 122, 122, 127, 122, 123, 122, 123, 123, 127, 123, 123, 123, 122, 122, 122, 127, 123, 123, 122, 123, 123, 123, 123, 127, 123, 123, 127, 127, 123, 122, 123, 122, 127, 127, 123, 127, 127, 127, 127, 122, 123, 122, 122, 123, 123, 123, 127, 127, 127, 127, 122, 122, 127, 123, 127, 122, 123, 127, 123, 123, 123, 122]}
>>> mode(result[122])
122
>>> mode(result[123])
127
>>> mode(result[127])
123
_______________________________________________________________________________________
SD:
{122: [127, 123, 123, 122, 122, 122, 122, 122, 122, 123, 122, 127, 122, 127, 122, 122, 123, 122, 122, 123, 123, 122, 122, 122, 122, 127, 122, 122, 127, 127, 127, 123, 123, 127, 122, 122, 127, 127, 123, 122, 122, 123, 127, 127, 122, 122, 122, 123, 123, 122, 123, 123, 127, 123, 123, 127, 122, 122, 123, 123, 122, 123, 127, 123, 122, 122, 122, 122, 123, 122, 122, 122, 123, 123, 127, 127, 123, 127, 127, 127, 123, 127, 122, 122, 127, 122, 123, 122, 127], 123: [123, 122, 123, 122, 122, 127, 127, 127, 127, 127, 123, 122, 123, 122, 123, 122, 122, 122, 123, 127, 127, 127, 123, 127, 123, 123, 123, 123, 123, 123, 127, 127, 122, 123, 127, 123, 123, 127, 123, 127, 122, 127, 122, 123, 122, 127, 123, 123, 122, 123, 122, 122, 127, 127, 127, 127, 123, 122, 127, 127, 123, 122, 122, 123, 122, 122, 127, 123, 127, 127, 123, 127, 127, 127, 122, 122, 122, 123, 127, 127, 127, 127, 123, 123, 123, 127, 127, 122, 122, 122, 122], 127: [123, 127, 122, 127, 123, 123, 123, 127, 123, 123, 123, 122, 122, 122, 123, 122, 127, 127, 127, 127, 127, 123, 127, 127, 123, 123, 123, 122, 122, 127, 122, 123, 122, 123, 123, 127, 123, 123, 123, 122, 122, 122, 127, 123, 123, 122, 123, 123, 123, 123, 127, 123, 123, 127, 127, 123, 122, 123, 122, 127, 127, 123, 127, 127, 127, 127, 122, 123, 122, 122, 123, 123, 123, 127, 127, 127, 127, 122, 122, 127, 123, 127, 122, 123, 127, 123, 123, 123, 122]}
>>> mode(result[122])
122
>>> mode(result[123])
127
>>> mode(result[127])
123
_______________________________________________________________________________________
SKEW:
{122: [127, 123, 123, 122, 122, 122, 122, 122, 122, 123, 122, 127, 122, 127, 122, 122, 123, 122, 122, 123, 123, 122, 122, 122, 122, 127, 122, 122, 127, 127, 127, 123, 123, 127, 122, 122, 127, 127, 123, 122, 122, 123, 127, 127, 122, 122, 122, 123, 123, 122, 123, 123, 127, 123, 123, 127, 122, 122, 123, 123, 122, 123, 127, 123, 122, 122, 122, 122, 123, 122, 122, 122, 123, 123, 127, 127, 123, 127, 127, 127, 123, 127, 122, 122, 127, 122, 123, 122, 127], 123: [123, 122, 123, 122, 122, 127, 127, 127, 127, 127, 123, 122, 123, 122, 123, 122, 122, 122, 123, 127, 127, 127, 123, 127, 123, 123, 123, 123, 123, 123, 127, 127, 122, 123, 127, 123, 123, 127, 123, 127, 122, 127, 122, 123, 122, 127, 123, 123, 122, 123, 122, 122, 127, 127, 127, 127, 123, 122, 127, 127, 123, 122, 122, 123, 122, 122, 127, 123, 127, 127, 123, 127, 127, 127, 122, 122, 122, 123, 127, 127, 127, 127, 123, 123, 123, 127, 127, 122, 122, 122, 122], 127: [123, 127, 122, 127, 123, 123, 123, 127, 123, 123, 123, 122, 122, 122, 123, 122, 127, 127, 127, 127, 127, 123, 127, 127, 123, 123, 123, 122, 122, 127, 122, 123, 122, 123, 123, 127, 123, 123, 123, 122, 122, 122, 127, 123, 123, 122, 123, 123, 123, 123, 127, 123, 123, 127, 127, 123, 122, 123, 122, 127, 127, 123, 127, 127, 127, 127, 122, 123, 122, 122, 123, 123, 123, 127, 127, 127, 127, 122, 122, 127, 123, 127, 122, 123, 127, 123, 123, 123, 122]}
>>> mode(result[122])
122
>>> mode(result[123])
127
>>> mode(result[127])
123
_______________________________________________________________________________________
KURT:
{122: [127, 123, 123, 122, 122, 122, 122, 122, 122, 123, 122, 127, 122, 127, 122, 122, 123, 122, 122, 123, 123, 122, 122, 122, 122, 127, 122, 122, 127, 127, 127, 123, 123, 127, 122, 122, 127, 127, 123, 122, 122, 123, 127, 127, 122, 122, 122, 123, 123, 122, 123, 123, 127, 123, 123, 127, 122, 122, 123, 123, 122, 123, 127, 123, 122, 122, 122, 122, 123, 122, 122, 122, 123, 123, 127, 127, 123, 127, 127, 127, 123, 127, 122, 122, 127, 122, 123, 122, 127], 123: [123, 122, 123, 122, 122, 127, 127, 127, 127, 127, 123, 122, 123, 122, 123, 122, 122, 122, 123, 127, 127, 127, 123, 127, 123, 123, 123, 123, 123, 123, 127, 127, 122, 123, 127, 123, 123, 127, 123, 127, 122, 127, 122, 123, 122, 127, 123, 123, 122, 123, 122, 122, 127, 127, 127, 127, 123, 122, 127, 127, 123, 122, 122, 123, 122, 122, 127, 123, 127, 127, 123, 127, 127, 127, 122, 122, 122, 123, 127, 127, 127, 127, 123, 123, 123, 127, 127, 122, 122, 122, 122], 127: [123, 127, 122, 127, 123, 123, 123, 127, 123, 123, 123, 122, 122, 122, 123, 122, 127, 127, 127, 127, 127, 123, 127, 127, 123, 123, 123, 122, 122, 127, 122, 123, 122, 123, 123, 127, 123, 123, 123, 122, 122, 122, 127, 123, 123, 122, 123, 123, 123, 123, 127, 123, 123, 127, 127, 123, 122, 123, 122, 127, 127, 123, 127, 127, 127, 127, 122, 123, 122, 122, 123, 123, 123, 127, 127, 127, 127, 122, 122, 127, 123, 127, 122, 123, 127, 123, 123, 123, 122]}
>>> mode(result[122])
122
>>> mode(result[123])
127
>>> mode(result[127])
123
_______________________________________________________________________________________
PercentageSubBand:
{122: [127, 123, 123, 122, 122, 122, 122, 122, 122, 123, 122, 127, 122, 127, 122, 122, 123, 122, 122, 123, 123, 122, 122, 122, 122, 127, 122, 122, 127, 127, 127, 123, 123, 127, 122, 122, 127, 127, 123, 122, 122, 123, 127, 127, 122, 122, 122, 123, 123, 122, 123, 123, 127, 123, 123, 127, 122, 122, 123, 123, 122, 123, 127, 123, 122, 122, 122, 122, 123, 122, 122, 122, 123, 123, 127, 127, 123, 127, 127, 127, 123, 127, 122, 122, 127, 122, 123, 122, 127], 123: [123, 122, 123, 122, 122, 127, 127, 127, 127, 127, 123, 122, 123, 122, 123, 122, 122, 122, 123, 127, 127, 127, 123, 127, 123, 123, 123, 123, 123, 123, 127, 127, 122, 123, 127, 123, 123, 127, 123, 127, 122, 127, 122, 123, 122, 127, 123, 123, 122, 123, 122, 122, 127, 127, 127, 127, 123, 122, 127, 127, 123, 122, 122, 123, 122, 122, 127, 123, 127, 127, 123, 127, 127, 127, 122, 122, 122, 123, 127, 127, 127, 127, 123, 123, 123, 127, 127, 122, 122, 122, 122], 127: [123, 127, 122, 127, 123, 123, 123, 127, 123, 123, 123, 122, 122, 122, 123, 122, 127, 127, 127, 127, 127, 123, 127, 127, 123, 123, 123, 122, 122, 127, 122, 123, 122, 123, 123, 127, 123, 123, 123, 122, 122, 122, 127, 123, 123, 122, 123, 123, 123, 123, 127, 123, 123, 127, 127, 123, 122, 123, 122, 127, 127, 123, 127, 127, 127, 127, 122, 123, 122, 122, 123, 123, 123, 127, 127, 127, 127, 122, 122, 127, 123, 127, 122, 123, 127, 123, 123, 123, 122]}
>>> mode(result[122])
122
>>> mode(result[123])
127
>>> mode(result[127])
123
_______________________________________________________________________________________
EnergySubBand:
{122: [127, 123, 123, 122, 122, 122, 122, 122, 122, 123, 122, 127, 122, 127, 122, 122, 123, 122, 122, 123, 123, 122, 122, 122, 122, 127, 122, 122, 127, 127, 127, 123, 123, 127, 122, 122, 127, 127, 123, 122, 122, 123, 127, 127, 122, 122, 122, 123, 123, 122, 123, 123, 127, 123, 123, 127, 122, 122, 123, 123, 122, 123, 127, 123, 122, 122, 122, 122, 123, 122, 122, 122, 123, 123, 127, 127, 123, 127, 127, 127, 123, 127, 122, 122, 127, 122, 123, 122, 127], 123: [123, 122, 123, 122, 122, 127, 127, 127, 127, 127, 123, 122, 123, 122, 123, 122, 122, 122, 123, 127, 127, 127, 123, 127, 123, 123, 123, 123, 123, 123, 127, 127, 122, 123, 127, 123, 123, 127, 123, 127, 122, 127, 122, 123, 122, 127, 123, 123, 122, 123, 122, 122, 127, 127, 127, 127, 123, 122, 127, 127, 123, 122, 122, 123, 122, 122, 127, 123, 127, 127, 123, 127, 127, 127, 122, 122, 122, 123, 127, 127, 127, 127, 123, 123, 123, 127, 127, 122, 122, 122, 122], 127: [123, 127, 122, 127, 123, 123, 123, 127, 123, 123, 123, 122, 122, 122, 123, 122, 127, 127, 127, 127, 127, 123, 127, 127, 123, 123, 123, 122, 122, 127, 122, 123, 122, 123, 123, 127, 123, 123, 123, 122, 122, 122, 127, 123, 123, 122, 123, 123, 123, 123, 127, 123, 123, 127, 127, 123, 122, 123, 122, 127, 127, 123, 127, 127, 127, 127, 122, 123, 122, 122, 123, 123, 123, 127, 127, 127, 127, 122, 122, 127, 123, 127, 122, 123, 127, 123, 123, 123, 122]}
>>> mode(result[122])
122
>>> mode(result[123])
127
>>> mode(result[127])
123


features = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': [],'EnergySubBand': [],'PercentageSubBand': [],'Energytot': []}

features = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': []}
for key, item in wpd_data.wpd_result.items():
    for fragment in item[1][122]:
        for feat in fragment:
            if( feat in features.keys() ):
                if( len( features[feat] ) ):
                    features[feat] = np.add( features[feat], fragment[feat] )
                else:
                    features[feat] = fragment[feat]
                    
                    
tot_wpd_results = {}
for electrode in wpd_data.wpd_result:
    tot_wpd_results.update({122: [], 123: [], 127: []})


wpd_data.wpd_result[electrode][1][task]

for electrode in wpd_data.electrodes:
    for task in [122, 123, 127]:
        for ix, fragment in enumerate( wpd_data.wpd_result[electrode][1][task] ):
            if( len( tot_wpd_results[task] ) and len(tot_wpd_results[task]) >= fragment ):
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['MAV'] )
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['AVP'] )
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['SD'] )
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['SKEW'] )
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['KURT'] )
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['EnergySubBand'] )
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['PercentageSubBand'] )
                tot_wpd_results[task][fragment] = np.add( tot_wpd_results[task][fragment], wpd_data.wpd_result[electrode][1][task][fragment]['Energytot'] )
            else:
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['MAV']
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['AVP']
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['SD']
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['SKEW']
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['KURT']
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['EnergySubBand']
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['PercentageSubBand']
                tot_wpd_results[task][fragment] = wpd_data.wpd_result[electrode][1][task][fragment]['Energytot']
                
                
                
                
for column in test[0].columns.values:
    print( pd.DataFrame(test[0][column].to_list(), index=test[0].index).add( 
                pd.DataFrame(test[1][column].to_list(), index=test[1].index),
                pd.DataFrame(test[2][column].to_list(), index=test[2].index),
                pd.DataFrame(test[3][column].to_list(), index=test[3].index),
                pd.DataFrame(test[4][column].to_list(), index=test[4].index),
                pd.DataFrame(test[5][column].to_list(), index=test[5].index),
                pd.DataFrame(test[6][column].to_list(), index=test[6].index),
                pd.DataFrame(test[7][column].to_list(), index=test[7].index)) )


'MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand','Energytot'

result = pd.DataFrame( columns=['MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand','Energytot'] )
for column in test[0].columns.values:
    for i in test:
        result[column].add( pd.DataFrame(i[column].to_list(), index=i.index) )

result = pd.DataFrame( columns=['MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand','Energytot'] )
for col in columns:
  aux = pd.DataFrame()
  print(col)
  for ix,i in enumerate(test):
    if( aux.empty ):
      aux = pd.DataFrame(i[col].to_list(), index=i.index)
    else:
      aux = aux.add( pd.DataFrame(i[col].to_list(), index=i.index) )
  result[col] = (aux/len(test)).values.tolist()




for task in self.tasks_intervals.keys():
    for electrode in self.electrodes:
        data_electrodes.append(
            pd.DataFrame(self.wpd_result_per_electrode[electrode][1][task])
        )



for task in self.tasks_intervals.keys():
    result = pd.DataFrame( columns=self.features )
    data_electrodes = []
    
    for electrode in self.electrodes:
        data_electrodes.append(
                    self.wpd_result_per_electrode[electrode][1][task]
        )
        
    for col in self.features:
        aux = pd.DataFrame()
        
        for ix, i in enumerate(data_electrodes):
            
            if( aux.empty ):
                aux = pd.DataFrame(i[col].to_list(), index=i.index)
            else:
                aux = aux.add( pd.DataFrame(i[col].to_list(), index=i.index) )
                
        result[col] = (aux/len(data_electrodes)).values.tolist()
    self.wpd_result_sum[task] = result
    
    
    
X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 30% test



CONF:
DWTcoeffs[8] = np.zeros_like(DWTcoeffs[8])
DWTcoeffs[7] = np.zeros_like(DWTcoeffs[7])
DWTcoeffs[6] = np.zeros_like(DWTcoeffs[6])
#DWTcoeffs[5] = np.zeros_like(DWTcoeffs[5])
#DWTcoeffs[0] = np.zeros_like(DWTcoeffs[0])

122 -127
>>> results['MAV']['rate']
66.66666666666667
>>> results['AVP']['rate']
66.66666666666667
>>> results['SD']['rate']
61.111111111111114
>>> results['SKEW']['rate']
66.66666666666667
>>> results['KURT']['rate']
61.111111111111114
>>> results['EnergySubBand']['rate']
77.77777777777777
>>> results['PercentageSubBand']['rate']
55.55555555555556
>>> results['Energytot']['rate']
77.77777777777777

122 - 123
>>> results['MAV']['rate']
50.0
>>> results['AVP']['rate']
61.111111111111114
>>> results['SD']['rate']
55.55555555555556
>>> results['SKEW']['rate']
44.44444444444444
>>> results['KURT']['rate']
44.44444444444444
>>> results['EnergySubBand']['rate']
77.77777777777777
>>> results['PercentageSubBand']['rate']
55.55555555555556
>>> results['Energytot']['rate']
77.77777777777777
