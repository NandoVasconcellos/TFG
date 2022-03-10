GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MAV
GradientBoostingClassifier(learning_rate=0.9, max_depth=4, min_samples_split=5,
                           n_estimators=1150, verbose=False)|56.3|            4.57|            3.97
Mean test Accuracy: 0.396 (0.107)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: AVP
GradientBoostingClassifier(learning_rate=0.9, min_samples_split=5,
                           n_estimators=1700, verbose=False)|57.04|            7.55|            3.03
Mean test Accuracy: 0.294 (0.053)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SD
GradientBoostingClassifier(learning_rate=0.7000000000000001, max_depth=4,
                           min_samples_split=5, n_estimators=1700,
                           verbose=False)|57.78|            9.6|            4.32
Mean test Accuracy: 0.412 (0.106)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SKEW
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=4,
                           n_estimators=2000, verbose=False)|34.81|            3.78|            3.26
Mean test Accuracy: 0.327 (0.087)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: KURT
GradientBoostingClassifier(learning_rate=0.8, min_samples_split=4,
                           n_estimators=50, verbose=False)|45.19|            5.24|            0.95
Mean test Accuracy: 0.345 (0.022)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: ZC
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|54.07|            8.95|            1.19
Mean test Accuracy: 0.448 (0.022)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MC
GradientBoostingClassifier(learning_rate=1.0, min_samples_split=4,
                           n_estimators=1150, verbose=False)|51.85|            1.05|            3.29
Mean test Accuracy: 0.361 (0.035)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: entropy
GradientBoostingClassifier(min_samples_split=4, n_estimators=600, verbose=False)|27.41|            3.78|            3.69
Mean test Accuracy: 0.276 (0.028)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n5
GradientBoostingClassifier(learning_rate=0.8, max_depth=4, min_samples_split=4,
                           n_estimators=1700, verbose=False)|57.04|            5.24|            4.51
Mean test Accuracy: 0.346 (0.092)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n25
GradientBoostingClassifier(learning_rate=1.0, min_samples_split=4,
                           n_estimators=1700, verbose=False)|59.26|            2.1|            4.18
Mean test Accuracy: 0.449 (0.055)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n75
GradientBoostingClassifier(learning_rate=0.4, min_samples_split=5,
                           n_estimators=50, verbose=False)|58.52|            3.78|            0.91
Mean test Accuracy: 0.415 (0.079)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n95
GradientBoostingClassifier(learning_rate=0.4, min_samples_split=4,
                           n_estimators=2000, verbose=False)|65.19|            5.24|            4.71
Mean test Accuracy: 0.467 (0.052)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: median
GradientBoostingClassifier(learning_rate=0.4, min_samples_split=5,
                           n_estimators=1700, verbose=False)|36.3|            5.24|            0.95
Mean test Accuracy: 0.277 (0.055)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: mean
GradientBoostingClassifier(learning_rate=0.9, min_samples_split=5,
                           n_estimators=1150, verbose=False)|35.56|            7.26|            4.26
Mean test Accuracy: 0.328 (0.029)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: std
GradientBoostingClassifier(learning_rate=0.8, min_samples_split=4,
                           n_estimators=600, verbose=False)|57.04|            7.33|            2.18
Mean test Accuracy: 0.432 (0.068)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: var
GradientBoostingClassifier(learning_rate=0.9, min_samples_split=5,
                           n_estimators=50, verbose=False)|58.52|            9.99|            1.11
Mean test Accuracy: 0.378 (0.082)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: rms
GradientBoostingClassifier(learning_rate=0.9, max_depth=4, min_samples_split=5,
                           n_estimators=1700, verbose=False)|58.52|            5.54|            4.38
Mean test Accuracy: 0.413 (0.111)
__________________________________________________________
