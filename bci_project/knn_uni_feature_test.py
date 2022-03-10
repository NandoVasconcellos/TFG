from pprint import pprint
features = {'MAV','AVP','SD','SKEW','KURT','ZC','MC','entropy','n5','n25','n75','n95','median','mean','std','var','rms','EnergySubBand','PercentageSubBand','Energytot'}
for feature in features:
    print(feature)
    results_per_file = { k: {pair: [] for pair in PAIRS } for k in knn }
    result = ''
    for test in range(len(wpd_data)):
        #print(f"_________________________________________FILE {test} AS TEST_________________________________________")
        for k in knn:
            for pair in PAIRS:
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
                result = float(metrics.accuracy_score(y_test, pred)*100)
                #print(f"MEDIAN FOR k={k} AND PAIR {pair}: {result}")
                results_per_file[k][pair].append( result )
    #print( results_per_file )
    pprint( { k: {pair: {'score': np.round( np.median(result), 2), 'std': np.round( np.std(result), 2)} for pair, result in results_per_file[k].items()} for k in results_per_file.keys() } )