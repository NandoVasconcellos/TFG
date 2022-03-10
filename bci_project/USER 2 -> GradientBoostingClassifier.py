USER 2 -> GradientBoostingClassifier
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MAV
GradientBoostingClassifier(learning_rate=0.2, min_samples_split=4,
                           n_estimators=50, verbose=False)|51.85|            4.19|            0.88
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
GradientBoostingClassifier(learning_rate=0.8, max_depth=4, min_samples_split=4,
                           n_estimators=600, verbose=False)|51.85|            3.78|            2.19
Mean test Accuracy: 0.363 (0.050)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: SD
GradientBoostingClassifier(max_depth=4, min_samples_split=5, n_estimators=600,
                           verbose=False)|52.59|            3.78|            9.21
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
GradientBoostingClassifier(learning_rate=0.6, max_depth=4, min_samples_split=4,
                           n_estimators=50, verbose=False)|40.0|            1.81|            1.65
Mean test Accuracy: 0.257 (0.079)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: KURT
GradientBoostingClassifier(max_depth=4, min_samples_split=5, n_estimators=50,
                           verbose=False)|37.78|            3.63|            1.16
Mean test Accuracy: 0.329 (0.070)
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
                           n_estimators=50, verbose=False)|54.07|            1.05|            1.09
Mean test Accuracy: 0.294 (0.053)
__________________________________________________________
GridSearchCV(cv=3, estimator=GradientBoostingClassifier(verbose=False),
             n_jobs=-1,
             param_grid=[{'learning_rate': array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]),
                          'max_depth': range(3, 5),
                          'min_samples_split': range(4, 6),
                          'n_estimators': [50, 600, 1150, 1700, 2000]}],
             scoring='accuracy')
FEATURE: MC
GradientBoostingClassifier(learning_rate=1.0, min_samples_split=5,
                           n_estimators=50, verbose=False)|50.37|            4.57|            1.68
Mean test Accuracy: 0.414 (0.044)
Mean test Accuracy: 0.310 (0.036)
Mean test Accuracy: 0.381 (0.071)
Mean test Accuracy: 0.430 (0.054)
Mean test Accuracy: 0.327 (0.046)
Mean test Accuracy: 0.483 (0.110)
Mean test Accuracy: 0.363 (0.157)
Mean test Accuracy: 0.346 (0.094)
Mean test Accuracy: 0.484 (0.060)
Mean test Accuracy: 0.414 (0.044)
Mean test Accuracy: 0.483 (0.032)
Mean test Accuracy: 0.415 (0.051)
Mean test Accuracy: 0.325 (0.098)
Mean test Accuracy: 0.535 (0.069)

