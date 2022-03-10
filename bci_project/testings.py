#27     122 - 123 | 127
#26     122 - 127 | 123
#76     122 - 127 | 123
#28     122 - 123 | 127
#199    127 - 123 | 127
#240    127 - 123 | 123
#127    123 - 122 | 122
#155    123 - 122 | 122
#258    127 - 123 | 127
#147    123 - 127 | 127
#5      122 - 123 | 123
#183    127 - 122 | 127
#74     122 - 123 | 123
#104    123 - 122 | 122
#63     122 - 127 | 127
125    123 - 123 | 122
148    123 - 123 | 122
185    127 - 127 | 127
47     122 - 122 | 122
55     122 - 122 | 122
201    127 - 127 | 123
220    127 - 127 | 123
70     122 - 122 | 122
231    127 - 127 | 127
130    123 - 123 | 123
45     122 - 122 | 123
33     122 - 122 | 122


#155    123 - 122 | 122
#258    127 - 123 | 127
#147    123 - 127 | 127
#5      122 - 123 | 123
#183    127 - 122 | 127
#74     122 - 123 | 123
#104    123 - 122 | 122
#63     122 - 127 | 127
#27     122 - 123 | 127
#26     122 - 127 | 123
#76     122 - 127 | 123
#28     122 - 123 | 127
125    123 - 123 | 122
148    123 - 123 | 122
185    127 - 127 | 127
47     122 - 122 | 122
55     122 - 122 | 122
199    127 - 123 | 127
201    127 - 127 | 123
220    127 - 127 | 123
240    127 - 123 | 123
127    123 - 122 | 122
70     122 - 122 | 122
231    127 - 127 | 127
130    123 - 123 | 123
45     122 - 122 | 123
33     122 - 122 | 122


test_target = y_test.tolist()
count = 0
for ix, data in enumerate(X_test):
    subtracted = data-aux
    mathed = np.sqrt(subtracted.pow(2).sum(axis=1))
    df_mathed = mathed.to_frame(name='data')
    df_mathed['task'] = y_train.values
    print( f" TARGET: {test_target[ix]} RESULT: {mode(df_mathed.sort_values(by='data').head(3)['task'].tolist())}" )
    if( test_target[ix] == mode(df_mathed.sort_values(by='data').head(3)['task'].tolist()) ):
        count += 1



df_test.MAV = results['MAV']['mode']
df_test.AVP = results['AVP']['mode']
df_test.SD  = results['SD']['mode']
df_test.SKEW  = results['SKEW']['mode']
df_test.KURT  = results['KURT']['mode']
df_test.EnergySubBand  = results['EnergySubBand']['mode']
df_test.PercentageSubBand  = results['PercentageSubBand']['mode']
df_test.Energytot  = results['Energytot']['mode']

results['MAV']['rate']
results['AVP']['rate']
results['SD']['rate']
results['SKEW']['rate']
results['KURT']['rate']
results['EnergySubBand']['rate']
results['PercentageSubBand']['rate']
results['Energytot']['rate']

df_test.target = results['Energytot']['target']

df1 = df_test.loc[:, df_test.columns != col]
df_test['mode'] = df1.mode(axis=1)[0].astype('int32')

CONF:
DWTcoeffs[8] = np.zeros_like(DWTcoeffs[8])
DWTcoeffs[7] = np.zeros_like(DWTcoeffs[7])
DWTcoeffs[6] = np.zeros_like(DWTcoeffs[6])
#DWTcoeffs[5] = np.zeros_like(DWTcoeffs[5])
#DWTcoeffs[0] = np.zeros_like(DWTcoeffs[0])

122 - 127
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

__ LAST __
>>> results['MAV']['rate']
61.111111111111114
>>> results['AVP']['rate']
72.22222222222223
>>> results['SD']['rate']
55.55555555555556
>>> results['SKEW']['rate']
55.55555555555556
>>> results['KURT']['rate']
44.44444444444444
>>> results['EnergySubBand']['rate']
77.77777777777777
>>> results['PercentageSubBand']['rate']
61.111111111111114
>>> results['Energytot']['rate']
77.77777777777777


___________________________________


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


BOTTOM
DWTcoeffs[8] = np.zeros_like(DWTcoeffs[8])
DWTcoeffs[7] = np.zeros_like(DWTcoeffs[7])
#DWTcoeffs[6] = np.zeros_like(DWTcoeffs[6])
#DWTcoeffs[5] = np.zeros_like(DWTcoeffs[5])
#DWTcoeffs[0] = np.zeros_like(DWTcoeffs[0])
target  mode
122     122 | -> w
122     123 | -> 
122     122 | -> w
122     122 | -> w
122     122 | -> w
123     123 | -> w
122     122 | -> w
122     122 | -> w
122     123 | -> 
122     123 | -> 
123     123 | -> w
123     123 | -> w
123     122 | -> 
123     123 | -> w
122     122 | -> w
123     122 | -> 
123     122 | -> 
122     123 | -> 


EXTREMES
DWTcoeffs[8] = np.zeros_like(DWTcoeffs[8])
DWTcoeffs[7] = np.zeros_like(DWTcoeffs[7])
#DWTcoeffs[6] = np.zeros_like(DWTcoeffs[6])
#DWTcoeffs[5] = np.zeros_like(DWTcoeffs[5])
#DWTcoeffs[0] = np.zeros_like(DWTcoeffs[0])
target  mode
122     122 | ->w
122     123 | ->
122     122 | ->w
122     122 | ->w
122     122 | ->w
123     123 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     123 | ->
123     123 | ->w
123     123 | ->w
123     123 | ->w
123     123 | ->w
122     122 | ->w
123     123 | ->w
123     122 | ->
122     123 | ->

17 ->   4
        13

SYM9
target  mode
122     122 | ->w
122     123 | ->
122     122 | ->w
122     122 | ->w
122     122 | ->w
123     123 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
123     122 | ->
123     122 | ->
123     123 | ->w
123     122 | ->
122     122 | ->w
123     123 | ->w
123     122 | ->
122     123 | ->


SYM9 + DB4
target  mode
122     122 | ->w
122     123 | ->
122     122 | ->w
122     122 | ->w
122     122 | ->w
123     123 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     123 | ->
123     123 | ->w
123     123 | ->w
123     123 | ->w
123     123 | ->w
122     122 | ->w
123     123 | ->w
123     122 | ->
122     123 | ->

?????
target  mode
122     122 | ->w
122     123 | ->
122     122 | ->w
122     123 | ->
122     122 | ->w
123     123 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     123 | ->
123     123 | ->w
123     123 | ->w
123     123 | ->w
123     123 | ->w
122     122 | ->w
123     122 | ->
123     122 | ->
122     122 | ->w


SMOOTH DENOISING
target  mode
122     122 | ->w
122     123 | ->
122     122 | ->w
122     123 | ->
122     122 | ->w
123     123 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     123 | ->
123     123 | ->w
123     123 | ->w
123     123 | ->w
123     123 | ->w
122     122 | ->w
123     122 | ->
123     122 | ->
122     122 | ->w


target  mode
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     123 | ->
122     122 | ->w
123     123 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     123 | ->
123     122 | ->
123     123 | ->w
123     123 | ->w
123     123 | ->w
122     122 | ->w
123     123 | ->w
123     122 | ->
122     123 | ->w


target  mode
122     122 | ->w
122     123 | ->
122     122 | ->w
122     122 | ->w
122     122 | ->w
123     123 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
123     122 | ->
123     123 | ->w
123     123 | ->w
123     122 | ->
122     122 | ->w
123     122 | ->
123     122 | ->
122     123 | ->


122 -127
target  mode
122     127 | ->
122     122 | ->w
122     122 | ->w
127     127 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     122 | ->w
122     127 | ->
122     122 | ->w
122     122 | ->w
127     127 | ->w
127     122 | ->
122     122 | ->w
127     127 | ->w
127     127 | ->w
122     127 | ->



_________________________________________________________________
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

PAIRS = [
    "[df_122, df_123]",
    "[df_122, df_127]",
    "[df_127, df_123]",
]

knn = [1, 3, 5, 15]


for k in knn:
    for pair in PAIRS:
        results = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': [],'EnergySubBand': [],'PercentageSubBand': [],'Energytot': []}
        for feature in features:
            df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
            df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
            df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
            df_tot_122.reset_index(drop=True, inplace=True)
            df_tot_123.reset_index(drop=True, inplace=True)
            df_tot_127.reset_index(drop=True, inplace=True)
            df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
            df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
            df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
            df = eval( f"pd.concat({pair})" )
            df.reset_index(drop=True, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.10,random_state=109) # 70% training and 10% test
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train.tolist())
            distances, indices = nbrs.kneighbors(X_test.tolist())
            target = y_test.tolist()
            count = 0
            results[feature] = {'mode': [], 'target': []}
            for ix, i in enumerate(indices):
                #if( ix != 0 ):
                results[feature]['mode'].append( mode( [ y_train.iloc[y] for ind, y in enumerate(i) ] ) )
                results[feature]['target'].append( target[ix] )
                if( mode( [ y_train.iloc[y] for ind, y in enumerate(i) ] ) == target[ix] ):
                    count += 1
            results[feature].update({'rate': (count*100/len(y_train)), 'y_test': y_test, 'X_test': X_test})
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
        count = 0
        for row in df_test[['target', 'mode']].to_records():
            if(row[1]==row[2]):
                count += 1   
        print( f" K: {k} | PAIR: {pair} | RATE: {(count/len(df_test['target']))*100}" )
        
        
features = {'MAV','AVP','SD','KURT','EnergySubBand','PercentageSubBand','Energytot'}
from sklearn import neighbors, datasets
from sklearn import metrics
for k in knn:
    for pair in PAIRS:
        print(pair)
        target =  []
        results = {'MAV': float(),'AVP': float(),'SD': float(),'KURT': float(),'EnergySubBand': float(),'PercentageSubBand': float(),'Energytot': float()}
        for feature in features:
            #print(feature)
            df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
            df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
            df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
            df_tot_122.reset_index(drop=True, inplace=True)
            df_tot_123.reset_index(drop=True, inplace=True)
            df_tot_127.reset_index(drop=True, inplace=True)
            df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
            df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
            df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
            df = eval( f"pd.concat({pair})" )
            df.reset_index(drop=True, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.1,random_state=109) # 70% training and 10% test
            target = y_test.tolist()
            nbrs = neighbors.KNeighborsClassifier(k, weights="distance").fit(X_train.tolist(), y_train.tolist())
            #distances, indices = nbrs.kneighbors(X_test.tolist())
            pred = nbrs.predict(X_test.tolist())
            #print(f"K: {k} | Feature: {feature} | Accuracy: {metrics.accuracy_score(y_test, pred)*100}")
            #print(pred)
            results[feature] = float(metrics.accuracy_score(y_test, pred)*100)
        print(f"MODE FOR k={k} AND PAIR {pair}: {mode([ acc for acc in results.values()])}")
        df_test = pd.DataFrame(columns=['MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand','Energytot','target'])
        df_test.MAV = results['MAV']['prediction']
        df_test.AVP = results['AVP']['prediction']
        df_test.SD  = results['SD']['prediction']
        df_test.SKEW  = results['SKEW']['prediction']
        df_test.KURT  = results['KURT']['prediction']
        df_test.EnergySubBand  = results['EnergySubBand']['prediction']
        df_test.PercentageSubBand  = results['PercentageSubBand']['prediction']
        df_test.Energytot  = results['Energytot']['prediction']
        df_test.target = target
        col = "target"
        df1 = df_test.loc[:, df_test.columns != col]
        df_test['mode'] = df1.mode(axis=1)[0].astype('int32')
        count = 0
        for row in df_test[['target', 'mode']].to_records():
            if(row[1]==row[2]):
                count += 1   
        print( f" K: {k} | PAIR: {pair} | RATE: {(count/len(df_test['target']))*100}" )
        
        
        
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


for pair in PAIRS:
    print(pair)
    target =  []
results = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': [],'EnergySubBand': [],'PercentageSubBand': [],'Energytot': []}
for feature in features:
    #print(feature)
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    #df = eval( f"pd.concat({pair})" )
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.1,random_state=109) # 70% training and 10% test
    
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train.tolist(),y_train.tolist())
    y_pred=clf.predict(X_test.tolist())
    print(f"Feature: {feature} | Accuracy: { int( metrics.accuracy_score(y_test, y_pred)*100 ) }")
        
  



df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122]['MAV'], wpd_data[1].wpd_result_sum[122]['MAV'], wpd_data[2].wpd_result_sum[122]['MAV'], wpd_data[3].wpd_result_sum[122]['MAV'] ] )
df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123]['MAV'], wpd_data[1].wpd_result_sum[123]['MAV'], wpd_data[2].wpd_result_sum[123]['MAV'], wpd_data[3].wpd_result_sum[123]['MAV'] ] )
df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127]['MAV'], wpd_data[1].wpd_result_sum[127]['MAV'], wpd_data[2].wpd_result_sum[127]['MAV'], wpd_data[3].wpd_result_sum[127]['MAV'] ] )
df_tot_122.reset_index(drop=True, inplace=True)
df_tot_123.reset_index(drop=True, inplace=True)
df_tot_127.reset_index(drop=True, inplace=True)
df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
df = pd.concat( [df_122, df_123, df_127] )
X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
X_train = scaling.transform(X_train.tolist())
X_test = scaling.transform(X_test.tolist())
  
  
  
|===|===|===|
| S | V | M |
|===|===|===|

from sklearn.model_selection import GridSearchCV
parameters =   [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],'gamma': ['auto', 1, 0.1, 0.02, 0.001]}]
from sklearn import svm
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
    X_train = scaling.transform(X_train.tolist())
    X_test = scaling.transform(X_test.tolist())
    svc = svm.SVC(verbose=False)
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train.tolist(), y_train.tolist())
    test_scores = cross_val_score(clf, X_test, y_test, cv=3)
    results_df = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
    print( f"{clf.best_estimator_}|{np.round(100*clf.best_score_,2)}|{np.round(100*results_df['std_test_score'].iloc[0],2)}|{np.round(1000*results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0],2)}" )
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
    #print(f" FEATURE: {feature} | Best score: {clf.best_score_} | Best estimator: {clf.best_estimator_}")
    
    
    


GRID SEARCH RESULT:
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: AVP | Best score: 0.5155049786628734 | Best estimator: SVC(C=1000, gamma=1)
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: SKEW | Best score: 0.4411095305832148 | Best estimator: SVC(C=1000, gamma=1, kernel='sigmoid')
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: Energytot | Best score: 0.36728307254623044 | Best estimator: SVC(C=10, gamma='auto', kernel='sigmoid')
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: SD | Best score: 0.5374110953058322 | Best estimator: SVC(C=1000, gamma='auto', kernel='poly')
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: MAV | Best score: 0.5110953058321479 | Best estimator: SVC(C=1, gamma=1)
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: KURT | Best score: 0.43598862019914647 | Best estimator: SVC(C=100, gamma=1)
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: PercentageSubBand | Best score: 0.4200568990042674 | Best estimator: SVC(C=1000, gamma=1, kernel='poly')
GridSearchCV(estimator=SVC(),
             param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                          'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
 FEATURE: EnergySubBand | Best score: 0.4894736842105264 | Best estimator: SVC(C=100, gamma=0.1)



from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
    X_train = scaling.transform(X_train.tolist())
    X_test = scaling.transform(X_test.tolist())
    
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train.tolist(), y_train.tolist())
    print(f" FEATURE: {feature} | Score: {gpc.score(X_train.tolist(), y_train.tolist())}")
    
    
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
    X_train = scaling.transform(X_train.tolist())
    X_test = scaling.transform(X_test.tolist())
    
    clf = DecisionTreeClassifier(random_state=0)
    cross_val_score(clf, X_train.tolist(), y_train.tolist(), cv=10)
    print(f" FEATURE: {feature} | Score: {mode(cross_val_score(clf, X_train.tolist(), y_train.tolist(), cv=10))}")
    
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
    X_train = scaling.transform(X_train.tolist())
    X_test = scaling.transform(X_test.tolist())
    
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf.fit(X_train.tolist(), y_train.tolist())
    y_predict = clf.predict( X_test.tolist() )
    test_scores = cross_val_score(clf, X_test.tolist(), y_test.tolist(), scoring='accuracy', n_jobs=-1, cv=3)
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
    print(f" FEATURE: {feature} | Score: {metrics.accuracy_score(y_test, y_predict)*100}")
    
    
    
df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122], wpd_data[1].wpd_result_sum[122], wpd_data[2].wpd_result_sum[122], wpd_data[3].wpd_result_sum[122] ] )
df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123], wpd_data[1].wpd_result_sum[123], wpd_data[2].wpd_result_sum[123], wpd_data[3].wpd_result_sum[123] ] )
df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127], wpd_data[1].wpd_result_sum[127], wpd_data[2].wpd_result_sum[127], wpd_data[3].wpd_result_sum[127] ] )
df_tot_122.reset_index(drop=True, inplace=True)
df_tot_123.reset_index(drop=True, inplace=True)
df_tot_127.reset_index(drop=True, inplace=True)
df_tot_122 = df_tot_122.assign(task=122)
df_tot_123 = df_tot_123.assign(task=123)
df_tot_127 = df_tot_127.assign(task=127)
df_tot = pd.concat( [df_tot_122, df_tot_123, df_tot_127] )

features = {'MAV','AVP','SD','EnergySubBand','Energytot'}
from sklearn.neural_network import MLPClassifier
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
    X_train = scaling.transform(X_train.tolist())
    X_test = scaling.transform(X_test.tolist())
    
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train.tolist(), y_train.tolist())
    y_predict = clf.predict( X_test.tolist() )
    
    print(f" FEATURE: {feature} | Score: {metrics.accuracy_score(y_test, y_predict)*100}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
from sklearn.preprocessing import MinMaxScaler
_________________________________________________________________
import code; code.interact(local=vars())
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from wavelet_processor import WaveletProcessor
import pandas as pd
import numpy as np
from statistics import mode, mean
from collections import Counter
wpd_data = WaveletProcessor.all
import os
from sklearn import neighbors, datasets
from sklearn import metrics
import statistics
os.system('clear')
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

PAIRS = [
    "[df_122, df_123]",
    "[df_122, df_127]",
    "[df_127, df_123]",
]

knn = [1, 3, 5, 15]
    
for k in knn:
    for pair in PAIRS:
        results = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': [],'EnergySubBand': [],'PercentageSubBand': [],'Energytot': []}
        for feature in features:
            # TRAIN
            df_train_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
            df_train_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
            df_train_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )            
            df_train_122.reset_index(drop=True, inplace=True)
            df_train_123.reset_index(drop=True, inplace=True)
            df_train_127.reset_index(drop=True, inplace=True)            
            df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
            df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
            df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
            df_train = eval( f"pd.concat({pair})" )
            df_train.reset_index(drop=True, inplace=True)
            #TEST
            df_122 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[122][feature], 'task':122})
            df_123 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[123][feature], 'task':123})
            df_127 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[127][feature], 'task':127})
            df_test = eval( f"pd.concat({pair})" )
            df_test.reset_index(drop=True, inplace=True)            
            X_train = df_train.data.tolist()
            y_train = df_train.task.tolist()            
            X_test = df_test.data.tolist()
            y_test = df_test.task.tolist()            
            #X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.10,random_state=109) # 70% training and 10% test
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
            distances, indices = nbrs.kneighbors(X_test)
            target = df_test.task.tolist()
            nbrs = neighbors.KNeighborsClassifier(k, weights="distance").fit(X_train, y_train)
            pred = nbrs.predict(X_test)
            results[feature] = float(metrics.accuracy_score(y_test, pred)*100)
            #print( f"FEATURE: {feature} | RESULT: {float(metrics.accuracy_score(y_test, pred)*100)}" )
        print(f"MODE FOR k={k} AND PAIR {pair}: {statistics.median([ acc for acc in results.values()])}")




results = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': [],'EnergySubBand': [],'PercentageSubBand': [],'Energytot': []}
for pair in PAIRS:
    for feature in features:
        # TRAIN
        df_train_122 = pd.concat( [ wpd_data[3].wpd_result_sum[122][feature], wpd_data[0].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature] ] )
        df_train_123 = pd.concat( [ wpd_data[3].wpd_result_sum[123][feature], wpd_data[0].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature] ] )
        df_train_127 = pd.concat( [ wpd_data[3].wpd_result_sum[127][feature], wpd_data[0].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature] ] )
        
        df_train_122.reset_index(drop=True, inplace=True)
        df_train_123.reset_index(drop=True, inplace=True)
        df_train_127.reset_index(drop=True, inplace=True)
        
        df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
        df_train = eval( f"pd.concat({pair})" )
        df_train.reset_index(drop=True, inplace=True)
        #TEST
        df_122 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[122][feature], 'task':122})
        df_123 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[123][feature], 'task':123})
        df_127 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[127][feature], 'task':127})
        df_test = eval( f"pd.concat({pair})" )
        df_test.reset_index(drop=True, inplace=True)
        X_train = df_train.data.tolist()
        y_train = df_train.task.tolist()
        X_test = df_test.data.tolist()
        y_test = df_test.task.tolist()
        clf=RandomForestClassifier(n_estimators=1000)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        results[feature].append( int( metrics.accuracy_score(y_test, y_pred)*100 ) )
        print(f"Feature: {feature} | Accuracy: { int( metrics.accuracy_score(y_test, y_pred)*100 ) }")
        
        
        
for feat, values in results.items():
    print(f"{feat}: {statistics.median(values)}")


results = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': [],'EnergySubBand': [],'PercentageSubBand': [],'Energytot': []}
    
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
for pair in PAIRS:
    print(f"PAIR: {pair}" )
    for feature in features:
        try:
            df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
            df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
            df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
            df_tot_122.reset_index(drop=True, inplace=True)
            df_tot_123.reset_index(drop=True, inplace=True)
            df_tot_127.reset_index(drop=True, inplace=True)
            df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
            df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
            df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
            df_train = eval( f"pd.concat({pair})" )
            df_train.reset_index(drop=True, inplace=True)
            X_train, X_test, y_train, y_test = train_test_split(df_train.data, df_train.task, test_size=0.3,random_state=109) # 70% training and 10% test
            scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
            X_train = scaling.transform(X_train.tolist())
            X_test = scaling.transform(X_test.tolist())
            gnb = GaussianNB()
            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            test_scores = cross_val_score(gnb, X_test.tolist(), y_test.tolist(), scoring='accuracy', cv=3, n_jobs=-1)
            #print('Mean train Accuracy: %.3f (%.3f)' % (np.mean(train_scores), np.std(train_scores)))
            print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
            #print(f"PAIR: {pair} | FEATURE: {feature} | Number of mislabeled points out of a total {X_test.shape[0]} points :  {(y_test != y_pred).sum()}")
        except Exception as e:
            print(e)
    print("_______________________________")





features = {'MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand','Energytot', 'ZC'}
from sklearn import preprocessing
for ix in range(len(wpd_data)-1):
    for task in [122, 123, 127]:
        for feature in features:
            wpd_data[ix].wpd_result_sum[task][feature] = pd.Series( preprocessing.normalize(wpd_data[ix].wpd_result_sum[task][feature].tolist()).tolist() )

parameters_KNN = {
    'n_neighbors': (1,3,5,6,7,8,9,10,11,13,15),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev')
}

grid_search_KNN = GridSearchCV(
    estimator=estimator_KNN,
    param_grid=parameters_KNN,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)
KNN_1=grid_search_KNN.fit(X_train, y_train)
y_pred_KNN1 =KNN_1.predict(X_test)
print(grid_search_KNN.best_params_ )
print('Best Score - KNN:', grid_search_KNN.best_score_ )


import code; code.interact(local=vars())
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from wavelet_processor import WaveletProcessor
import pandas as pd
import numpy as np
from statistics import mode, mean
from collections import Counter
wpd_data = WaveletProcessor.all
import os
from sklearn import neighbors, datasets
from sklearn import metrics
import statistics

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

import os
from sklearn import neighbors, datasets
from sklearn import metrics
import statistics

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

PAIRS = [
    "[df_122, df_123]",
    "[df_122, df_127]",
    "[df_127, df_123]",
]
os.system('clear')
for pair in PAIRS:
    estimator_KNN = KNeighborsClassifier(algorithm='auto')
    parameters_KNN = {
        'n_neighbors': (1,3,5,6,7,8,9,10,11,13,15),
        'leaf_size': (20,40,1),
        'p': (1,2),
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'chebyshev')
    }
    grid_search_KNN = GridSearchCV(
        estimator=estimator_KNN,
        param_grid=parameters_KNN,
        scoring = 'accuracy',
        n_jobs = -1,
        cv = 5
    )
    for feature in features:
        # TRAIN
        df_train_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
        df_train_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
        df_train_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
        df_train_122.reset_index(drop=True, inplace=True)
        df_train_123.reset_index(drop=True, inplace=True)
        df_train_127.reset_index(drop=True, inplace=True)            
        df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
        df_tot = eval( f"pd.concat({pair})" )
        df_tot.reset_index(drop=True, inplace=True)           
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df_tot.data.tolist(), df_tot.task.tolist(), test_size = 0.30, random_state = 109)
        
        KNN_model_default = KNeighborsClassifier()
        KNN_model_default.fit(X_train, y_train)
        y_pred_KNN_default =KNN_model_default.predict(X_test)
        pred = float(metrics.accuracy_score(y_test, y_pred_KNN_default)*100)
        if( pred > 80 ):
            print(f"{feature} | {pair}")
            print( f"DEFAULT prediction rate: {pred}" )
            print("_______________________END_________________________")
        
        KNN_1 = grid_search_KNN.fit(X_train, y_train)
        y_pred_KNN1 = KNN_1.predict(X_test)
        if( grid_search_KNN.best_score_*100 > 80 ):
            print(f"{feature} | {pair}")
            print(grid_search_KNN.best_params_ )
            print('Best Score - KNN:', grid_search_KNN.best_score_ )
            print("_______________________END_________________________")



#print(f"MODE FOR k={k} AND PAIR {pair}: {statistics.median([ acc for acc in results.values()])}")

features = {'MAV','AVP','SD','SKEW','KURT','EnergySubBand','PercentageSubBand'}

for test in range(len(wpd_data)):
    # TRAIN
    df_train_122 = pd.concat( [ wpd_data[i].wpd_result_sum[122][feature] for i in range(len(wpd_data)) if i != test ] )
    df_train_123 = pd.concat( [ wpd_data[i].wpd_result_sum[123][feature] for i in range(len(wpd_data)) if i != test ] )
    df_train_127 = pd.concat( [ wpd_data[i].wpd_result_sum[127][feature] for i in range(len(wpd_data)) if i != test ] )
    
    df_train_122.reset_index(drop=True, inplace=True)
    df_train_123.reset_index(drop=True, inplace=True)
    df_train_127.reset_index(drop=True, inplace=True)
    
    df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
    df_train = eval( f"pd.concat({pair})" )
    df_train.reset_index(drop=True, inplace=True)
    #TEST
    df_122 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[122][feature], 'task':122})
    df_123 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[123][feature], 'task':123})
    df_127 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[127][feature], 'task':127})
    df_test = eval( f"pd.concat({pair})" )
    df_test.reset_index(drop=True, inplace=True)
    X_train = df_train.data.tolist()
    y_train = df_train.task.tolist()
    X_test = df_test.data.tolist()
    y_test = df_test.task.tolist()
    print( f"{len(X_train)} | {len(y_train)} | {len(X_test)} | {len(y_test)}" )



    
for pair in PAIRS:
    estimator_KNN = KNeighborsClassifier(algorithm='auto')
    parameters_KNN = {
        'n_neighbors': (1,3,5,6,7,8,9,10,11,13,15),
        'leaf_size': (20,30,40,1),
        'p': (1,2),
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'chebyshev')
    }
    grid_search_KNN = GridSearchCV(
        estimator=estimator_KNN,
        param_grid=parameters_KNN,
        scoring = 'accuracy',
        n_jobs = -1,
        cv = 5
    )
    for test in range(len(wpd_data)):
        print(f"Index as test: {test}")
        for feature in features:
            # TRAIN
            df_train_122 = pd.concat( [ wpd_data[i].wpd_result_sum[122][feature] for i in range(len(wpd_data)) if i != test ] )
            df_train_122 = pd.concat( [ wpd_data[i].wpd_result_sum[123][feature] for i in range(len(wpd_data)) if i != test ] )
            df_train_122 = pd.concat( [ wpd_data[i].wpd_result_sum[127][feature] for i in range(len(wpd_data)) if i != test ] )
            
            df_train_122.reset_index(drop=True, inplace=True)
            df_train_123.reset_index(drop=True, inplace=True)
            df_train_127.reset_index(drop=True, inplace=True)
            
            df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
            df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
            df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
            df_train = eval( f"pd.concat({pair})" )
            df_train.reset_index(drop=True, inplace=True)
            #TEST
            df_122 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[122][feature], 'task':122})
            df_123 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[123][feature], 'task':123})
            df_127 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[127][feature], 'task':127})
            df_test = eval( f"pd.concat({pair})" )
            df_test.reset_index(drop=True, inplace=True)
            X_train = df_train.data.tolist()
            y_train = df_train.task.tolist()
            X_test = df_test.data.tolist()
            y_test = df_test.task.tolist()
            print(f"{feature} | {pair}")
            print( f"{len(X_train)} | {len(y_train)} | {len(X_test)} | {len(y_test)}" )
            KNN_model_default = KNeighborsClassifier()
            KNN_model_default.fit(X_train, y_train)
            y_pred_KNN_default =KNN_model_default.predict(X_test)
            pred = float(metrics.accuracy_score(y_test, y_pred_KNN_default)*100)
            if( pred > 80 ):
                print(f"{feature} | {pair}")
                print( f"DEFAULT prediction rate: {pred}" )
                print("_______________________END_________________________")
            KNN_1 = grid_search_KNN.fit(X_train, y_train)
            y_pred_KNN1 = KNN_1.predict(X_test)
            if( grid_search_KNN.best_score_*100 > 80 ):
                print(f"{feature} | {pair}")
                print(grid_search_KNN.best_params_ )
                print('Best Score - KNN:', grid_search_KNN.best_score_ )
                print("_______________________END_________________________")
                
                
                
                
            features = {'MAV','AVP','SD','EnergySubBand','Energytot', 'ZC'}    
_________________________________________________________________

import code; code.interact(local=vars())
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from wavelet_processor import WaveletProcessor
import pandas as pd
import numpy as np
from statistics import mode, mean
from collections import Counter
wpd_data = WaveletProcessor.all
import os
from sklearn import neighbors, datasets
from sklearn import metrics
import statistics

import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

import os
from sklearn import neighbors, datasets
from sklearn import metrics
import statistics

features = ['MAV','AVP','SD','SKEW','KURT','ZC','MC','entropy','n5','n25','n75','n95','median','mean','std','var','rms','EnergySubBand','PercentageSubBand','Energytot']


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

PAIRS = [
    "[df_122, df_123]",
    "[df_122, df_127]",
    "[df_127, df_123]",
]

knn = [1, 3, 5, 15]

results_per_file = { k: {pair: [] for pair in PAIRS } for k in knn }
os.system('clear')

estimator_KNN = KNeighborsClassifier(algorithm='auto')
parameters_KNN = {
    'n_neighbors': (1,3,5,6,7,8,9,10,11,13,15),
    'leaf_size': (20,40,1),
    'p': (1,2),
    'weights': ('uniform', 'distance'),
    'metric': ('minkowski', 'chebyshev')
}
grid_search_KNN = GridSearchCV(
    estimator=estimator_KNN,
    param_grid=parameters_KNN,
    scoring = 'accuracy',
    n_jobs = -1,
    cv = 5
)

results = {
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
    #'EnergySubBand': [],
    'PercentageSubBand': [],
    'Energytot': []
}

for k in knn:
    for test in range(len(wpd_data)):
        print(f"_________________________________________FILE {test} AS TEST_________________________________________")
        for pair in PAIRS:
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
            for feature in features:
                print(f"{feature} | {pair}")
                # TRAIN
                df_train_122 = pd.concat( [ wpd_data[i].wpd_result_sum[122][feature] for i in range(len(wpd_data)) if i != test ] )
                df_train_123 = pd.concat( [ wpd_data[i].wpd_result_sum[123][feature] for i in range(len(wpd_data)) if i != test ] )
                df_train_127 = pd.concat( [ wpd_data[i].wpd_result_sum[127][feature] for i in range(len(wpd_data)) if i != test ] )                
                df_train_122.reset_index(drop=True, inplace=True)
                df_train_123.reset_index(drop=True, inplace=True)
                df_train_127.reset_index(drop=True, inplace=True)
                df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
                df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
                df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
                df_train = eval( f"pd.concat({pair})" )
                df_train.reset_index(drop=True, inplace=True)
                #print(df_train)
                #TEST
                df_122 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[122][feature], 'task':122})
                df_123 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[123][feature], 'task':123})
                df_127 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[127][feature], 'task':127})
                df_test = eval( f"pd.concat({pair})" )
                df_test.reset_index(drop=True, inplace=True)
                #print(df_test)
                X_train = df_train.data.tolist()
                y_train = df_train.task.tolist()
                X_test = df_test.data.tolist()
                y_test = df_test.task.tolist()
                #X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.10,random_state=109) # 70% training and 10% test
                #nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
                #distances, indices = nbrs.kneighbors(X_test)
                #target = df_test.task.tolist()
                #nbrs = neighbors.KNeighborsClassifier(k, weights="uniform", p=1, leaf_size=20).fit(X_train, y_train)
                #pred = nbrs.predict(X_test)
                #results[feature] = float(metrics.accuracy_score(y_test, pred)*100)
                #print( f"DEFAULT ------FEATURE: {feature} | RESULT: {float(metrics.accuracy_score(y_test, pred)*100)}" )
                #___________________________________________
                KNN_1 = grid_search_KNN.fit(X_train, y_train)
                y_pred_KNN1 = KNN_1.predict(X_test)
                #print(f"{feature} | {pair}")
                print(grid_search_KNN.best_params_ )
                print('++++++++++++++++++++Best Score - KNN:', grid_search_KNN.best_score_ )
                #print(f"MEDIAN FOR k={k} AND PAIR {pair}: {statistics.median([ acc for acc in results.values()])}")
                #results_per_file[k][pair].append( statistics.median([ acc for acc in results.values()]) )


results_per_file = { k: {pair: [] for pair in PAIRS } for k in knn }
results = {'MAV': [],'AVP': [],'SD': [],'ZC': [],
            'n25': [],
            'rms': []} 
for k in knn:
    for test in range(len(wpd_data)):
        print(f"_________________________________________FILE {test} AS TEST_________________________________________")
        for pair in PAIRS:
            features = ['MAV','AVP','SD','SKEW','KURT','ZC','MC','entropy','n5','n25','n75','n95','median','mean','std','var','rms','EnergySubBand','PercentageSubBand','Energytot']
            features = {'MAV': [],'AVP': [],'SD': [], 'ZC': [],
            'n25': [],
            'rms': []} 
            for feature in features:
                # TRAIN
                df_train_122 = pd.concat( [ wpd_data[i].wpd_result_sum[122][feature] for i in range(len(wpd_data)) if i != test ] )
                df_train_123 = pd.concat( [ wpd_data[i].wpd_result_sum[123][feature] for i in range(len(wpd_data)) if i != test ] )
                df_train_127 = pd.concat( [ wpd_data[i].wpd_result_sum[127][feature] for i in range(len(wpd_data)) if i != test ] )
                df_train_122.reset_index(drop=True, inplace=True)
                df_train_123.reset_index(drop=True, inplace=True)
                df_train_127.reset_index(drop=True, inplace=True)
                df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
                df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
                df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
                df_train = eval( f"pd.concat({pair})" )
                df_train.reset_index(drop=True, inplace=True)
                #TEST
                df_122 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[122][feature], 'task':122})
                df_123 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[123][feature], 'task':123})
                df_127 = pd.DataFrame({'data': wpd_data[test].wpd_result_sum[127][feature], 'task':127})
                df_test = eval( f"pd.concat({pair})" )
                df_test.reset_index(drop=True, inplace=True)
                X_train = df_train.data.tolist()
                y_train = df_train.task.tolist()
                X_test = df_test.data.tolist()
                y_test = df_test.task.tolist()
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
                distances, indices = nbrs.kneighbors(X_test)
                target = df_test.task.tolist()
                nbrs = neighbors.KNeighborsClassifier(k, weights="uniform", p=1, leaf_size=20).fit(X_train, y_train)
                pred = nbrs.predict(X_test)
                results[feature] = float(metrics.accuracy_score(y_test, pred)*100)
            print(f"MEDIAN FOR k={k} AND PAIR {pair}: {statistics.median([ acc for acc in results.values()])}")
            results_per_file[k][pair].append( statistics.median([ acc for acc in results.values()]) )

from pprint import pprint
pprint( { k: {pair: statistics.median(result) for pair, result in results_per_file[k].items()} for k in results_per_file.keys() } )



from sklearn.model_selection import GridSearchCV
parameters =   [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],'gamma': ['auto', 1, 0.1, 0.02, 0.001]}]
from sklearn import svm
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
    X_train = scaling.transform(X_train.tolist())
    X_test = scaling.transform(X_test.tolist())
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train.tolist(), y_train.tolist())
    #if( clf.best_score_ > 0.7 ):
    results_df = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
    print(f"FEATURE: {feature}")
    #print(f"Best estimator: {clf.best_estimator_}")
    #print(f"Best score: {clf.best_score_}")
    #print(f"std: {results_df['std_test_score'].iloc[0]}")
    #print(f"time: {results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0]}")
    print( f"{clf.best_estimator_}|{np.round(100*clf.best_score_,2)}|{np.round(100*results_df['std_test_score'].iloc[0],2)}|{np.round(1000*results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0],2)}" )
    print("__________________________________________________________")
    


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
for pair in PAIRS:
    print(pair)
    for feature in features:
        df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
        df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
        df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
        df_tot_122.reset_index(drop=True, inplace=True)
        df_tot_123.reset_index(drop=True, inplace=True)
        df_tot_127.reset_index(drop=True, inplace=True)
        df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
        df = pd.concat([df_122, df_123, df_127])
        #df = eval( f"pd.concat({pair})" )
        df.reset_index(drop=True, inplace=True)
        #print(f"FEATURE: {feature}")
        X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        cls = GradientBoostingClassifier(n_estimators=2000)
        cls.fit(X_train, y_train)
        #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
        #train_scores = cross_val_score(cls, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        test_scores = cross_val_score(cls, X_test, y_test, scoring='accuracy', cv=3, n_jobs=-1)
        #print('Mean train Accuracy: %.3f (%.3f)' % (np.mean(train_scores), np.std(train_scores)))
        print('%.3f;%.3f'% (100*np.mean(test_scores), 100*np.std(test_scores)))

[df_122, df_123]
GradientBoostingClassifier(verbose=False)
Mean test Accuracy: 0.750 (0.068)




parameters = {
    "loss":["deviance"],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

parameters = {'learning_rate': [0.01,0.02,0.03,0.04],
                'subsample'    : [0.9, 0.5, 0.2, 0.1],
                'n_estimators' : [100,500,1000, 1500],
                'max_depth'    : [4,6,8,10]
                }
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
for pair in PAIRS:
    print(pair)
    for feature in features:
        df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
        df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
        df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
        df_tot_122.reset_index(drop=True, inplace=True)
        df_tot_123.reset_index(drop=True, inplace=True)
        df_tot_127.reset_index(drop=True, inplace=True)
        df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
        #df = pd.concat([df_122, df_123, df_127])
        df = eval( f"pd.concat({pair})" )
        df.reset_index(drop=True, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        #pipe = make_pipeline(GradientBoostingClassifier(verbose=False))
        GBC = GradientBoostingClassifier(verbose=False)
        #tuned_parameters = [{
        #    'max_depth': range(3, 5),
        #    'min_samples_split': range(4,6),
        #    'learning_rate':np.linspace(0.1, 1, 10),
        #    'n_estimators':[50, 600, 1150, 1700, 2000]
        #    }]
        #clf = GridSearchCV(GBC, tuned_parameters, cv=3, scoring='accuracy', n_jobs=-1)
        #clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)
        #clf.fit(X_train, y_train)
        GBC.fit(X_train, y_train)
        #results_df = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
        #print(f"FEATURE: {feature}")
        ##cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
        #print( f"{clf.best_estimator_}|{np.round(100*clf.best_score_,2)}|\
        #        {np.round(100*results_df['std_test_score'].iloc[0],2)}|\
        #        {np.round(1000*results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0],2)}" )
        #train_scores = cross_val_score(clf, X_train, y_train, scoring='f1_micro', cv=cv, n_jobs=-1)
        test_scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=3, n_jobs=-1)
        #print('Mean train Accuracy: %.3f (%.3f)' % (np.mean(train_scores), np.std(train_scores)))
        print('Mean test Accuracy: %.3f (%.3f)' % (np.mean(test_scores), np.std(test_scores)))
        print("__________________________________________________________")



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()
    param_grid = {'max_depth':[3,4,5,6,7,8,9,10],
              'max_features':[0.8,0.9,1],
              'learning_rate':[0.01,0.1,1],
              'n_estimators':[80,100,120,140,150],
              'subsample': [0.8,0.9,1]}
    clf = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, scoring=acc_score, cv=5)
    #clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)
    clf.fit(X_train, y_train)
    results_df = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
    print(f"FEATURE: {feature}")
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    print( f"{clf.best_estimator_}|{np.round(100*clf.best_score_,2)}|\
            {np.round(100*results_df['std_test_score'].iloc[0],2)}|\
            {np.round(1000*results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0],2)}" )
    #train_scores = cross_val_score(clf, X_train, y_train, scoring='f1_micro', cv=cv, n_jobs=-1)
    test_scores = cross_val_score(clf, X_test, y_test, cv=3, n_jobs=-1)
    #print('Mean train Accuracy: %.3f (%.3f)' % (np.mean(train_scores), np.std(train_scores)))
    print('Mean test Accuracy: %.3f (%.3f)' % (np.mean(test_scores), np.std(test_scores)))
    print("__________________________________________________________")





from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import time
for feature in features:
    try:
        df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
        df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
        df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
        df_tot_122.reset_index(drop=True, inplace=True)
        df_tot_123.reset_index(drop=True, inplace=True)
        df_tot_127.reset_index(drop=True, inplace=True)
        df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
        df = pd.concat([df_122, df_123, df_127])
        df.reset_index(drop=True, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        #print(y_pred)
        test_scores = cross_val_score(gnb, X_test, y_test, scoring='accuracy', cv=3, n_jobs=-1)
        results_df = pd.DataFrame(gnb.cv_results_).sort_values(by=['rank_test_score'])
        print( f"{np.round(1000*results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0],2)}" )
        #print('Mean train Accuracy: %.3f (%.3f)' % (np.mean(train_scores), np.std(train_scores)))
        #print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
        #print("_______________________________")
        print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
        #print(f"FEATURE: {feature} | Number of mislabeled points out of a total {X_test.shape[0]} points :  {(y_test != y_pred).sum()}")
    except Exception as e:
        print(e)
    #print("_______________________________")



for pair in PAIRS:
    print(pair)
    target =  []

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
for feature in features:
    #print(feature)
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    #df = eval( f"pd.concat({pair})" )
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.1,random_state=109) # 70% training and 10% test
    clf=RandomForestClassifier(n_estimators=10, max_depth=5, verbose=False)
    clf.fit(X_train.tolist(),y_train.tolist())
    y_pred=clf.predict(X_test.tolist())
    test_scores = cross_val_score(clf, X_test.tolist(), y_test.tolist(), scoring='accuracy', n_jobs=-1, cv=3)
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
    print(f"Feature: {feature} | Accuracy: { int( metrics.accuracy_score(y_test, y_pred)*100 ) }")





from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
for pair in PAIRS:
    print(pair)
    for feature in features:
        df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
        df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
        df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
        df_tot_122.reset_index(drop=True, inplace=True)
        df_tot_123.reset_index(drop=True, inplace=True)
        df_tot_127.reset_index(drop=True, inplace=True)
        df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
        df = pd.concat([df_122, df_123, df_127])
        df.reset_index(drop=True, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
        df = pd.concat([df_122, df_123, df_127])
        clf=RandomForestClassifier(n_estimators=100, verbose=False)
        clf.fit(X_train,y_train)
        #y_pred=clf.predict(X_test)
        test_scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=3, n_jobs=-1)
        print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
        #print(f"Feature: {feature} | Accuracy: { int( metrics.accuracy_score(y_test, y_pred)*100 ) }")


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    #warnings.filterwarnings('error',category=ConvergenceWarning, module='sklearn')    
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('error')
        maximum_iter = 7000
        while True:
            try:
                mlp_gs = MLPClassifier(max_iter=maximum_iter, verbose=False)
                clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
                print(w)
                print("hey jude")
                #break
            except Warning as e:
                print(e)
                maximum_iter += 500
                print(maximum_iter)
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels
    #y_predict = clf.predict( X_test.tolist() )
    test_scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=3, n_jobs=-1)
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
    #print(f" FEATURE: {feature} | Score: {metrics.accuracy_score(y_test, y_predict)*100}")



from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score
parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
for pair in PAIRS:
    print(pair)
for feature in features:
    df_tot_122 = pd.concat( [ 
        wpd_data[0].wpd_result_sum[122][feature],
        wpd_data[1].wpd_result_sum[122][feature],
        wpd_data[2].wpd_result_sum[122][feature],
        wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ 
        wpd_data[0].wpd_result_sum[123][feature],
        wpd_data[1].wpd_result_sum[123][feature],
        wpd_data[2].wpd_result_sum[123][feature],
        wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ 
        wpd_data[0].wpd_result_sum[127][feature],
        wpd_data[1].wpd_result_sum[127][feature],
        wpd_data[2].wpd_result_sum[127][feature],
        wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    #df = eval( f"pd.concat({pair})" )
    df.reset_index(drop=True, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    maximum_iter = 10000
    mlp_gs = MLPClassifier(max_iter=maximum_iter, verbose=False)
    clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    clf.fit(X_train, y_train) # X is train samples and y is the corresponding labels
    #y_predict = clf.predict( X_test.tolist() )
    test_scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=3, n_jobs=-1)
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))


    /python3.10/site-packages/sklearn/neural_network/_multilayer_perceptron.py:692: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (100) reached and the optimization hasn't converged yet.






from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
params = {'max_leaf_nodes': list(range(2, 100)),
            'min_samples_split': [2, 3, 4]}
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    df.reset_index(drop=True, inplace=True)    
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    clf = GridSearchCV(DecisionTreeClassifier(random_state=0), params, verbose=0, cv=3)
    clf.fit(X_train, y_train)
    test_scores = cross_val_score(clf, X_test, y_test, cv=3)
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
params = {'max_leaf_nodes': list(range(2, 100)),
            'min_samples_split': [2, 3, 4]}
for pair in PAIRS:
    print(pair)
    for feature in features:
        df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
        df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
        df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
        df_tot_122.reset_index(drop=True, inplace=True)
        df_tot_123.reset_index(drop=True, inplace=True)
        df_tot_127.reset_index(drop=True, inplace=True)
        df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
        df = eval( f"pd.concat({pair})" )
        df.reset_index(drop=True, inplace=True)    
        X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
        clf = GridSearchCV(DecisionTreeClassifier(random_state=0), params, verbose=0, cv=3)
        clf.fit(X_train, y_train)
        test_scores = cross_val_score(clf, X_test, y_test, cv=3)
        print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))







from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
for pair in PAIRS:
    print(pair)
    for feature in features:
        df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
        df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
        df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
        df_tot_122.reset_index(drop=True, inplace=True)
        df_tot_123.reset_index(drop=True, inplace=True)
        df_tot_127.reset_index(drop=True, inplace=True)
        df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
        df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
        df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
        #df = pd.concat([df_122, df_123, df_127])
        df = eval( f"pd.concat({pair})" )
        df.reset_index(drop=True, inplace=True)        
        X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
        X_train = X_train.tolist()
        X_test = X_test.tolist()
        y_train = y_train.tolist()
        y_test = y_test.tolist()
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
        X_train = scaling.transform(X_train)
        X_test = scaling.transform(X_test)
        kernel = 1.0 * RBF(1.0)
        gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train, y_train)
        test_scores = cross_val_score(gpc, X_test, y_test, cv=3)
        print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))



from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
for feature in features:
    df_tot_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[1].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
    df_tot_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[1].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
    df_tot_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[1].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )
    df_tot_122.reset_index(drop=True, inplace=True)
    df_tot_123.reset_index(drop=True, inplace=True)
    df_tot_127.reset_index(drop=True, inplace=True)
    df_122 = pd.DataFrame({'data': df_tot_122, 'task':122})
    df_123 = pd.DataFrame({'data': df_tot_123, 'task':123})
    df_127 = pd.DataFrame({'data': df_tot_127, 'task':127})
    df = pd.concat([df_122, df_123, df_127])
    #df = eval( f"pd.concat({pair})" )
    df.reset_index(drop=True, inplace=True)        
    X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.3,random_state=109) # 70% training and 10% test
    X_train = X_train.tolist()
    X_test = X_test.tolist()
    y_train = y_train.tolist()
    y_test = y_test.tolist()
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X_train, y_train)
    test_scores = cross_val_score(gpc, X_test, y_test, cv=3)
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))



for k in knn:
    for pair in PAIRS:
        results = {'MAV': [],'AVP': [],'SD': [],'SKEW': [],'KURT': [],'EnergySubBand': [],'PercentageSubBand': [],'Energytot': []}
        for feature in features:
            # TRAIN
            df_train_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122][feature], wpd_data[2].wpd_result_sum[122][feature], wpd_data[3].wpd_result_sum[122][feature] ] )
            df_train_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123][feature], wpd_data[2].wpd_result_sum[123][feature], wpd_data[3].wpd_result_sum[123][feature] ] )
            df_train_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127][feature], wpd_data[2].wpd_result_sum[127][feature], wpd_data[3].wpd_result_sum[127][feature] ] )            
            df_train_122.reset_index(drop=True, inplace=True)
            df_train_123.reset_index(drop=True, inplace=True)
            df_train_127.reset_index(drop=True, inplace=True)            
            df_122 = pd.DataFrame({'data': df_train_122, 'task':122})
            df_123 = pd.DataFrame({'data': df_train_123, 'task':123})
            df_127 = pd.DataFrame({'data': df_train_127, 'task':127})
            df_train = eval( f"pd.concat({pair})" )
            df_train.reset_index(drop=True, inplace=True)
            #TEST
            df_122 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[122][feature], 'task':122})
            df_123 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[123][feature], 'task':123})
            df_127 = pd.DataFrame({'data': wpd_data[1].wpd_result_sum[127][feature], 'task':127})
            df_test = eval( f"pd.concat({pair})" )
            df_test.reset_index(drop=True, inplace=True)            
            X_train = df_train.data.tolist()
            y_train = df_train.task.tolist()            
            X_test = df_test.data.tolist()
            y_test = df_test.task.tolist()            
            #X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.10,random_state=109) # 70% training and 10% test
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X_train)
            distances, indices = nbrs.kneighbors(X_test)
            target = df_test.task.tolist()
            nbrs = neighbors.KNeighborsClassifier(k, weights="distance").fit(X_train, y_train)
            pred = nbrs.predict(X_test)
            results[feature] = float(metrics.accuracy_score(y_test, pred)*100)
            #print( f"FEATURE: {feature} | RESULT: {float(metrics.accuracy_score(y_test, pred)*100)}" )
        print(f"MODE FOR k={k} AND PAIR {pair}: {statistics.median([ acc for acc in results.values()])}")