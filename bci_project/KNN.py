KNN
{1:
{'[df_122, df_123]': 59.25287356321839,
'[df_122, df_127]': 66.7624521072797,
'[df_127, df_123]': 86.9949494949495},
3:
{'[df_122, df_123]': 64.03225806451613,
'[df_122, df_127]': 72.31800766283524,
'[df_127, df_123]': 84.0909090909091},
5:
{'[df_122, df_123]': 68.9247311827957,
'[df_122, df_127]': 80.14705882352942,
'[df_127, df_123]': 85.47979797979798},
15:
{'[df_122, df_123]': 67.25806451612902,
'[df_122, df_127]': 74.50980392156862,
'[df_127, df_123]': 86.9949494949495}}


{1:
{'[df_122, df_123]': 58.45383759733036,
'[df_122, df_127]': 63.31417624521073,
'[df_127, df_123]': 86.54188948306594},
3:
{'[df_122, df_123]': 62.36559139784946,
'[df_122, df_127]': 74.37739463601532,
'[df_127, df_123]': 85.47979797979798},
5:
{'[df_122, df_123]': 68.9247311827957,
'[df_122, df_127]': 77.28758169934639,
'[df_127, df_123]': 84.0909090909091},
15:
{'[df_122, df_123]': 72.15053763440861,
'[df_122, df_127]': 74.50980392156862,
'[df_127, df_123]': 88.38383838383838}}


{1: {'[df_122, df_123]': 60.12235817575083,
     '[df_122, df_127]': 62.19958202716823,
     '[df_127, df_123]': 85.47979797979798},
 3: {'[df_122, df_123]': 62.33870967741936,
     '[df_122, df_127]': 73.51532567049807,
     '[df_127, df_123]': 84.78535353535354},
 5: {'[df_122, df_123]': 69.73118279569891,
     '[df_122, df_127]': 76.62835249042146,
     '[df_127, df_123]': 85.5429292929293},
 15: {'[df_122, df_123]': 71.31720430107526,
      '[df_122, df_127]': 73.85057471264368,
      '[df_127, df_123]': 89.07828282828282}}


SVM
FZ CZ FC5 FC6 FC1 FC2 -> 5s window | mode overlaping 2s | extremes 6 levels
________________________________________________________________________________________
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: EnergySubBand | Best score: 0.6764550264550264 | Best estimator: SVC(C=10, gamma='auto', kernel='linear')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: SD | Best score: 0.6542328042328042 | Best estimator: SVC(C=100, gamma='auto')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: MAV | Best score: 0.6833333333333333 | Best estimator: SVC(C=0.1, gamma=1, kernel='poly')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: Energytot | Best score: 0.6542328042328043 | Best estimator: SVC(C=100, gamma='auto', kernel='poly')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: AVP | Best score: 0.6468253968253969 | Best estimator: SVC(C=1000, gamma=0.02, kernel='sigmoid')
=====================================================================================================
FZ CZ FC5 FC6 FC1 FC2 -> 3s window | mode overlaping 2s | extremes 6 levels
_____________________________________________________________________________________________________
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: MAV | Best score: 0.6782828282828283 | Best estimator: SVC(C=100, gamma=0.02)
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: AVP | Best score: 0.6560606060606061 | Best estimator: SVC(C=100, gamma=0.1)
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: EnergySubBand | Best score: 0.6604040404040403 | Best estimator: SVC(C=100, gamma='auto')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: Energytot | Best score: 0.6205050505050506 | Best estimator: SVC(C=100, gamma='auto', kernel='poly')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: SD | Best score: 0.6603030303030304 | Best estimator: SVC(C=0.01, gamma=1, kernel='poly')
=====================================================================================================
FZ CZ FC5 FC6 FC1 FC2 -> 5s window | mode overlaping 1s | extremes 6 levels
_____________________________________________________________________________________________________
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: MAV | Best score: 0.6910052910052911 | Best estimator: SVC(C=10, gamma='auto', kernel='linear')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: EnergySubBand | Best score: 0.6835978835978836 | Best estimator: SVC(C=1000, gamma='auto', kernel='poly')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: SD | Best score: 0.6904761904761905 | Best estimator: SVC(C=10, gamma='auto', kernel='linear')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: Energytot | Best score: 0.6542328042328043 | Best estimator: SVC(C=1000, gamma='auto', kernel='poly')
    GridSearchCV(estimator=SVC(),
                param_grid=[{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'gamma': ['auto', 1, 0.1, 0.02, 0.001],
                            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}])
    FEATURE: AVP | Best score: 0.6978835978835979 | Best estimator: SVC(C=100, gamma='auto', kernel='linear')
    
    
    
plt.plot([data for data in wpd_data[0].wpd_result_sum[122].SD.tolist()], 's', color='r')
plt.plot([data for data in wpd_data[0].wpd_result_sum[123].SD.tolist()], 'x', color='b')
plt.plot([data for data in wpd_data[0].wpd_result_sum[127].SD.tolist()], '1', color='g')
plt.show()


plt.plot(df_train_122.tolist(), 's', color='r')
plt.plot(df_train_123.tolist(), 'x', color='b')
plt.plot(df_train_127.tolist(), '1', color='g')
plt.show()


df_train_122 = pd.concat( [ wpd_data[0].wpd_result_sum[122], wpd_data[1].wpd_result_sum[122], wpd_data[2].wpd_result_sum[122], wpd_data[3].wpd_result_sum[122] ] )
df_train_123 = pd.concat( [ wpd_data[0].wpd_result_sum[123], wpd_data[1].wpd_result_sum[123], wpd_data[2].wpd_result_sum[123], wpd_data[3].wpd_result_sum[123] ] )
df_train_127 = pd.concat( [ wpd_data[0].wpd_result_sum[127], wpd_data[1].wpd_result_sum[127], wpd_data[2].wpd_result_sum[127], wpd_data[3].wpd_result_sum[127] ] )

df_train_122.reset_index(drop=True, inplace=True)
df_train_123.reset_index(drop=True, inplace=True)
df_train_127.reset_index(drop=True, inplace=True)
df_train_122['task'] = 122
df_train_123['task'] = 123
df_train_127['task'] = 127
df_tot = pd.concat( [df_train_122, df_train_123, df_train_127] )
test = df_tot[['MAV','AVP']].values
X_train, X_test, y_train, y_test = train_test_split(test.data.tolist(), test.task.tolist(), test_size=0.30,random_state=109) # 70% training and 10% test
X_train, X_test, y_train, y_test = train_test_split(test.data, test.task.tolist(), test_size=0.30,random_state=109) # 70% training and 10% test


scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
print(f" FEATURE: {feature} | Best score: {clf.best_score_} | Best estimator: {clf.best_estimator_}")
