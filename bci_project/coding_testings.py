67,98;8,02
72,89;10,22
69,47;8,20
40,61;3,36
30,53;4,15
57,81;9,69
57,54;8,87
32,37;7,18
68,07;11,41
69,39;7,54
69,65;6,57
73,07;8,14
40,70;4,20
37,37;3,35
74,56;4,13
61,05;4,35
67,98;8,02
76,14;9,82
57,72;9,03
60,79;9,71


import time
for k in knn:
    print(k)
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
        y_train = y_train.tolist()
        X_test = X_test.tolist()
        y_test = y_test.tolist()
        #X_train, X_test, y_train, y_test = train_test_split(df.data, df.task, test_size=0.10,random_state=109) # 70% training and 10% test
        start = time.time()
        nbrs = neighbors.KNeighborsClassifier(k, weights="distance").fit(X_train, y_train)
        end = time.time()
        print(end - start)
        test_scores = cross_val_score(nbrs, X_test, y_test, cv=3)
        #print('%.2f;%.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))






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
    results_df = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
    print( f"{clf.best_estimator_}|{np.round(100*clf.best_score_,2)}|{np.round(100*results_df['std_test_score'].iloc[0],2)}|{np.round(1000*results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0],2)}" )
    #print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))





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
    start = time.time()
    clf=RandomForestClassifier(n_estimators=10, max_depth=5, verbose=False)
    start = time.time()
    clf.fit(X_train.tolist(),y_train.tolist())
    end = time.time()
    print(end - start)
    y_pred=clf.predict(X_test.tolist())
    test_scores = cross_val_score(clf, X_test.tolist(), y_test.tolist(), scoring='accuracy', n_jobs=-1, cv=3)
    #print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
    #print(f"Feature: {feature} | Accuracy: { int( metrics.accuracy_score(y_test, y_pred)*100 ) }")






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
    #clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    start = time.time()
    mlp_gs.fit(X_train, y_train) # X is train samples and y is the corresponding labels
    end = time.time()
    print(end - start)






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
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0, n_jobs=-1)
    start = time.time()
    gpc.fit(X_train, y_train)
    end = time.time()
    print(end - start)
    test_scores = cross_val_score(gpc, X_test, y_test, cv=3)
    print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))


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
    start = time.time()
    clf.fit(X_train.tolist(), y_train.tolist())
    end = time.time()
    print(end - start)
    test_scores = cross_val_score(clf, X_test, y_test, cv=3)
    results_df = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])
    print( f"{clf.best_estimator_}|{np.round(100*clf.best_score_,2)}|{np.round(100*results_df['std_test_score'].iloc[0],2)}|{np.round(1000*results_df['mean_score_time'].iloc[0]+results_df['mean_fit_time'].iloc[0],2)}" )
    
    #print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))
    #print(f" FEATURE: {feature} | Best score: {clf.best_score_} | Best estimator: {clf.best_estimator_}")
    



from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
parameters =   [{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],'gamma': ['auto', 1, 0.1, 0.02, 0.001]}]
from sklearn import svm
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
        scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train.tolist())
        X_train = scaling.transform(X_train.tolist())
        X_test = scaling.transform(X_test.tolist())
        svc = svm.SVC(verbose=False)
        clf = GridSearchCV(svc, parameters)
        clf.fit(X_train.tolist(), y_train.tolist())
        test_scores = cross_val_score(clf, X_test, y_test, cv=3)
        print('%.2f, %.2f' % (100*np.mean(test_scores), 100*np.std(test_scores)))