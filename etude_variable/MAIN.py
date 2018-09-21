import sys

sys.path.insert(0, "C:\projets_python\diabolo")


import warnings



import sklearn
import etude_variable.jouer as jouer

import etude_variable.lecture_data as ld
import pandas as pd
import numpy as np
import matplotlib
import scipy
import platform

# make sure to use position 1


print('Operating system version....', platform.platform())
print("Python version is........... %s.%s.%s" % sys.version_info[:3])
print('scikit-learn version is.....', sklearn.__version__)
print('pandas version is...........', pd.__version__)
print('numpy version is............', np.__version__)
print('matplotlib version is.......', matplotlib.__version__)
print('scipy version is.......', scipy.__version__)


def jouerlescourse(nb_itera, allure, mode_debug=0, actualise=0):
    global best_model_1, best_model_2, best_model_3, best_model_4, best_model_5, \
        feature_columns1, feature_columns2, feature_columns3, \
        feature_columns4, feature_columns5, response_column

    # ENTRAINE_ALLURE
    best_model, feature_columns, response_column = jouer.entraine_allure(allure=allure,
                                                                         nb_iter=nb_itera,
                                                                         mode_debug=mode_debug,
                                                                         actualise=actualise)

    if allure == 1:
        best_model_1 = best_model
        feature_columns1 = feature_columns

    if allure == 2:
        best_model_2 = best_model
        feature_columns2 = feature_columns

    if allure == 3:
        best_model_3 = best_model
        feature_columns3 = feature_columns

    if allure == 4:
        best_model_4 = best_model
        feature_columns4 = feature_columns

    if allure == 5:
        best_model_5 = best_model
        feature_columns5 = feature_columns

    jouer.construireFichierCSV(allure=allure, best_model=best_model,
                               feature_columns=feature_columns,
                               response_column=response_column,
                               mode_debug=mode_debug)
