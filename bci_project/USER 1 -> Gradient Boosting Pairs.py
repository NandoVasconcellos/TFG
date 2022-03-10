USER 1 -> Gradient Boosting Pairs
[df_122, df_123]
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MAV
GradientBoostingClassifier(learning_rate=1.0, max_depth=4, min_samples_split=4,
                           n_estimators=50, verbose=False)|67.59|                7.35|                0.56
Mean test Accuracy: 0.694 (0.157)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: AVP
GradientBoostingClassifier(learning_rate=0.30000000000000004, max_depth=4,
                           min_samples_split=5, n_estimators=50, verbose=False)|57.85|                1.0|                0.55
Mean test Accuracy: 0.583 (0.068)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SD
GradientBoostingClassifier(learning_rate=0.5, max_depth=4, min_samples_split=5,
                           n_estimators=2000, verbose=False)|59.17|                8.44|                1.52
Mean test Accuracy: 0.639 (0.039)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SKEW
GradientBoostingClassifier(learning_rate=0.8, min_samples_split=4,
                           n_estimators=2000, verbose=False)|59.04|                1.47|                1.55
Mean test Accuracy: 0.389 (0.039)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: KURT
GradientBoostingClassifier(max_depth=4, min_samples_split=4, n_estimators=50,
                           verbose=False)|60.19|                3.58|                0.63
Mean test Accuracy: 0.556 (0.104)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: ZC
GradientBoostingClassifier(learning_rate=1.0, min_samples_split=4,
                           n_estimators=50, verbose=False)|65.04|                4.61|                0.58
Mean test Accuracy: 0.722 (0.039)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MC
GradientBoostingClassifier(learning_rate=0.30000000000000004, max_depth=4,
                           min_samples_split=4, n_estimators=50, verbose=False)|69.93|                4.13|                0.55
Mean test Accuracy: 0.611 (0.104)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: entropy
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|50.57|                2.25|                0.49
Mean test Accuracy: 0.528 (0.039)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n5
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=5,
                           n_estimators=1150, verbose=False)|67.33|                8.45|                1.15
Mean test Accuracy: 0.639 (0.104)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n25
GradientBoostingClassifier(learning_rate=0.2, min_samples_split=5,
                           n_estimators=1150, verbose=False)|78.35|                4.9|                1.21
Mean test Accuracy: 0.667 (0.068)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n75
GradientBoostingClassifier(learning_rate=0.4, max_depth=4, min_samples_split=5,
                           n_estimators=1700, verbose=False)|72.27|                1.98|                1.62
Mean test Accuracy: 0.750 (0.068)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n95
GradientBoostingClassifier(learning_rate=0.8, min_samples_split=4,
                           n_estimators=50, verbose=False)|69.89|                1.5|                0.57
Mean test Accuracy: 0.639 (0.104)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: median
GradientBoostingClassifier(learning_rate=0.2, min_samples_split=5,
                           n_estimators=1150, verbose=False)|60.23|                0.69|                1.33
Mean test Accuracy: 0.694 (0.039)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: mean
GradientBoostingClassifier(learning_rate=0.5, min_samples_split=5,
                           n_estimators=600, verbose=False)|59.08|                6.45|                1.22
Mean test Accuracy: 0.333 (0.136)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: std
GradientBoostingClassifier(learning_rate=0.6, min_samples_split=4,
                           n_estimators=1150, verbose=False)|60.32|                5.35|                1.24
Mean test Accuracy: 0.639 (0.039)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: var
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=4,
                           n_estimators=1700, verbose=False)|55.42|                7.29|                1.46
Mean test Accuracy: 0.528 (0.104)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: rms
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=4,
                           n_estimators=600, verbose=False)|66.4|                8.04|                1.01
Mean test Accuracy: 0.694 (0.157)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: EnergySubBand
GradientBoostingClassifier(learning_rate=0.5, min_samples_split=4,
                           n_estimators=600, verbose=False)|61.38|                7.14|                0.97
Mean test Accuracy: 0.611 (0.079)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: PercentageSubBand
GradientBoostingClassifier(learning_rate=0.4, min_samples_split=4,
                           n_estimators=1700, verbose=False)|70.99|                6.38|                1.46
Mean test Accuracy: 0.528 (0.079)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: Energytot
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|51.76|                3.88|                0.51
Mean test Accuracy: 0.722 (0.171)
__________________________________________________________
[df_122, df_127]
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MAV
GradientBoostingClassifier(min_samples_split=4, n_estimators=1700,
                           verbose=False)|78.21|                4.34|                1.62
Mean test Accuracy: 0.599 (0.099)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: AVP
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|77.03|                10.03|                0.59
Mean test Accuracy: 0.498 (0.052)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SD
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=4,
                           n_estimators=1700, verbose=False)|78.17|                7.0|                1.45
Mean test Accuracy: 0.621 (0.117)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SKEW
GradientBoostingClassifier(learning_rate=1.0, max_depth=4, min_samples_split=4,
                           n_estimators=600, verbose=False)|52.08|                7.34|                0.95
Mean test Accuracy: 0.601 (0.021)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: KURT
GradientBoostingClassifier(learning_rate=0.4, max_depth=4, min_samples_split=5,
                           n_estimators=50, verbose=False)|58.71|                2.79|                0.6
Mean test Accuracy: 0.548 (0.074)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: ZC
GradientBoostingClassifier(learning_rate=0.8, min_samples_split=5,
                           n_estimators=2000, verbose=False)|75.05|                3.74|                1.59
Mean test Accuracy: 0.553 (0.153)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MC
GradientBoostingClassifier(learning_rate=1.0, max_depth=4, min_samples_split=5,
                           n_estimators=1700, verbose=False)|78.28|                5.41|                1.32
Mean test Accuracy: 0.451 (0.016)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: entropy
GradientBoostingClassifier(learning_rate=0.2, max_depth=4, min_samples_split=5,
                           n_estimators=600, verbose=False)|54.34|                0.71|                1.2
Mean test Accuracy: 0.601 (0.021)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n5
GradientBoostingClassifier(learning_rate=0.7000000000000001,
                           min_samples_split=4, n_estimators=50, verbose=False)|82.54|                5.73|                0.58
Mean test Accuracy: 0.623 (0.072)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n25
GradientBoostingClassifier(learning_rate=0.8, min_samples_split=4,
                           n_estimators=600, verbose=False)|73.84|                5.71|                0.94
Mean test Accuracy: 0.597 (0.104)
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
                           n_estimators=1700, verbose=False)|80.32|                7.42|                1.5
Mean test Accuracy: 0.498 (0.052)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n95
GradientBoostingClassifier(learning_rate=0.5, max_depth=4, min_samples_split=5,
                           n_estimators=2000, verbose=False)|84.8|                2.95|                1.76
Mean test Accuracy: 0.696 (0.130)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: median
GradientBoostingClassifier(learning_rate=0.5, max_depth=4, min_samples_split=4,
                           n_estimators=1700, verbose=False)|51.08|                0.76|                1.45
Mean test Accuracy: 0.601 (0.066)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: mean
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=5,
                           n_estimators=1700, verbose=False)|53.33|                4.89|                1.52
Mean test Accuracy: 0.526 (0.018)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: std
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=4,
                           n_estimators=600, verbose=False)|77.1|                7.28|                0.93
Mean test Accuracy: 0.571 (0.106)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: var
GradientBoostingClassifier(learning_rate=0.9, max_depth=4, min_samples_split=4,
                           n_estimators=1700, verbose=False)|76.02|                4.46|                1.43
Mean test Accuracy: 0.474 (0.018)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: rms
GradientBoostingClassifier(learning_rate=0.7000000000000001,
                           min_samples_split=4, n_estimators=50, verbose=False)|79.28|                5.77|                0.59
Mean test Accuracy: 0.599 (0.099)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: EnergySubBand
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|77.03|                10.03|                1.16
Mean test Accuracy: 0.447 (0.108)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: PercentageSubBand
GradientBoostingClassifier(learning_rate=0.6, min_samples_split=5,
                           n_estimators=600, verbose=False)|67.31|                5.8|                1.03
Mean test Accuracy: 0.425 (0.094)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: Energytot
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|75.05|                3.74|                0.52
Mean test Accuracy: 0.474 (0.018)
__________________________________________________________
[df_127, df_123]
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MAV
GradientBoostingClassifier(learning_rate=0.4, min_samples_split=4,
                           n_estimators=50, verbose=False)|88.73|                5.05|                0.93
Mean test Accuracy: 0.929 (0.058)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: AVP
GradientBoostingClassifier(min_samples_split=4, n_estimators=600, verbose=False)|85.61|                3.7|                1.23
Mean test Accuracy: 0.905 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SD
GradientBoostingClassifier(learning_rate=0.7000000000000001,
                           min_samples_split=4, n_estimators=1150,
                           verbose=False)|87.69|                4.15|                1.18
Mean test Accuracy: 0.929 (0.058)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SKEW
GradientBoostingClassifier(learning_rate=0.2, max_depth=4, min_samples_split=4,
                           n_estimators=50, verbose=False)|61.77|                5.75|                0.58
Mean test Accuracy: 0.524 (0.121)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: KURT
GradientBoostingClassifier(min_samples_split=5, n_estimators=50, verbose=False)|61.93|                5.81|                0.6
Mean test Accuracy: 0.667 (0.067)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: ZC
GradientBoostingClassifier(learning_rate=1.0, min_samples_split=5,
                           n_estimators=1700, verbose=False)|82.51|                2.63|                1.5
Mean test Accuracy: 0.881 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MC
GradientBoostingClassifier(learning_rate=0.9, min_samples_split=4,
                           n_estimators=1700, verbose=False)|84.66|                8.81|                1.46
Mean test Accuracy: 0.833 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: entropy
GradientBoostingClassifier(min_samples_split=4, n_estimators=50, verbose=False)|56.69|                0.62|                0.52
Mean test Accuracy: 0.452 (0.067)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n5
GradientBoostingClassifier(learning_rate=0.4, min_samples_split=4,
                           n_estimators=50, verbose=False)|82.51|                5.74|                0.6
Mean test Accuracy: 0.976 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n25
GradientBoostingClassifier(learning_rate=0.7000000000000001,
                           min_samples_split=4, n_estimators=1150,
                           verbose=False)|85.61|                2.68|                1.21
Mean test Accuracy: 0.881 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n75
GradientBoostingClassifier(learning_rate=0.8, min_samples_split=5,
                           n_estimators=50, verbose=False)|91.82|                5.1|                0.57
Mean test Accuracy: 0.952 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: n95
GradientBoostingClassifier(learning_rate=1.0, max_depth=4, min_samples_split=4,
                           n_estimators=1700, verbose=False)|86.68|                7.56|                1.05
Mean test Accuracy: 0.952 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: median
GradientBoostingClassifier(learning_rate=0.30000000000000004, max_depth=4,
                           min_samples_split=4, n_estimators=50, verbose=False)|71.05|                5.62|                0.67
Mean test Accuracy: 0.500 (0.101)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: mean
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=5,
                           n_estimators=2000, verbose=False)|67.08|                5.97|                1.54
Mean test Accuracy: 0.643 (0.000)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: std
GradientBoostingClassifier(learning_rate=1.0, min_samples_split=4,
                           n_estimators=600, verbose=False)|88.7|                2.72|                0.98
Mean test Accuracy: 0.905 (0.034)
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
                           n_estimators=600, verbose=False)|83.55|                5.1|                1.42
Mean test Accuracy: 0.905 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: rms
GradientBoostingClassifier(learning_rate=0.30000000000000004,
                           min_samples_split=5, n_estimators=50, verbose=False)|88.73|                5.05|                0.59
Mean test Accuracy: 0.929 (0.058)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: EnergySubBand
GradientBoostingClassifier(learning_rate=0.30000000000000004,
                           min_samples_split=5, n_estimators=50, verbose=False)|87.66|                2.36|                1.26
Mean test Accuracy: 0.905 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: PercentageSubBand
GradientBoostingClassifier(learning_rate=0.2, min_samples_split=5,
                           n_estimators=50, verbose=False)|64.05|                8.95|                0.57
Mean test Accuracy: 0.690 (0.034)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: Energytot
GradientBoostingClassifier(min_samples_split=5, n_estimators=50, verbose=False)|73.17|                1.83|                0.53
Mean test Accuracy: 0.905 (0.034)
__________________________________________________________