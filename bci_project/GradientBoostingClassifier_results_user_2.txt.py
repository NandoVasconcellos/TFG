GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MAV
GradientBoostingClassifier(learning_rate=0.9, max_depth=4, min_samples_split=5,
                           n_estimators=1150, verbose=False)|51.85|            3.78|            3.39
Mean test Accuracy: 0.483 (0.032)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: AVP
GradientBoostingClassifier(learning_rate=0.30000000000000004,
                           min_samples_split=4, n_estimators=50, verbose=False)|52.59|            5.83|            0.95
Mean test Accuracy: 0.381 (0.103)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SD
GradientBoostingClassifier(learning_rate=0.30000000000000004,
                           min_samples_split=4, n_estimators=50, verbose=False)|51.85|            2.77|            4.47
Mean test Accuracy: 0.449 (0.055)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SKEW
GradientBoostingClassifier(learning_rate=0.9, min_samples_split=5,
                           n_estimators=2000, verbose=False)|39.26|            2.77|            4.95
Mean test Accuracy: 0.275 (0.046)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: KURT
GradientBoostingClassifier(min_samples_split=5, n_estimators=50, verbose=False)|36.3|            2.77|            0.9
Mean test Accuracy: 0.346 (0.094)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: ZC
GradientBoostingClassifier(learning_rate=0.4, max_depth=4, min_samples_split=5,
                           n_estimators=50, verbose=False)|54.81|            2.77|            1.04
Mean test Accuracy: 0.311 (0.116)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MC
GradientBoostingClassifier(learning_rate=0.7000000000000001,
                           min_samples_split=5, n_estimators=1700,
                           verbose=False)|49.63|            4.57|            4.6
Mean test Accuracy: 0.396 (0.038)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: entropy
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|41.48|            1.05|            0.7
Mean test Accuracy: 0.310 (0.036)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n5
GradientBoostingClassifier(learning_rate=0.30000000000000004,
                           min_samples_split=5, n_estimators=50, verbose=False)|55.56|            1.81|            0.89
Mean test Accuracy: 0.414 (0.044)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n25
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|56.3|            6.87|            0.76
Mean test Accuracy: 0.481 (0.084)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n75
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|57.04|            5.54|            0.77
Mean test Accuracy: 0.378 (0.055)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n95
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|56.3|            6.37|            0.77
Mean test Accuracy: 0.500 (0.107)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: median
GradientBoostingClassifier(learning_rate=0.6, min_samples_split=5,
                           n_estimators=1700, verbose=False)|47.41|            7.55|            4.34
Mean test Accuracy: 0.346 (0.127)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: mean
GradientBoostingClassifier(learning_rate=1.0, min_samples_split=5,
                           n_estimators=2000, verbose=False)|37.78|            3.63|            5.35
Mean test Accuracy: 0.363 (0.050)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: std
GradientBoostingClassifier(learning_rate=0.8, max_depth=4, min_samples_split=5,
                           n_estimators=1150, verbose=False)|52.59|            2.1|            3.14
Mean test Accuracy: 0.449 (0.055)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: var
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|54.81|            6.37|            0.8
Mean test Accuracy: 0.414 (0.044)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: rms
GradientBoostingClassifier(learning_rate=0.8, max_depth=4, min_samples_split=5,
                           n_estimators=50, verbose=False)|51.11|            3.14|            0.98
Mean test Accuracy: 0.501 (0.036)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: EnergySubBand
GradientBoostingClassifier(learning_rate=0.9, max_depth=4, min_samples_split=4,
                           n_estimators=2000, verbose=False)|51.85|            7.33|            4.93
Mean test Accuracy: 0.312 (0.079)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: PercentageSubBand
GradientBoostingClassifier(min_samples_split=5, n_estimators=50, verbose=False)|46.67|            1.81|            1.23
Mean test Accuracy: 0.308 (0.103)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: Energytot
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|45.19|            7.55|            0.95
Mean test Accuracy: 0.535 (0.069)
__________________________________________________________
>>> 
