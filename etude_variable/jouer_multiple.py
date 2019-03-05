
import sys

#LOCALISATION DES DONNEES
sys.path.insert(0, "C:/projets_python/diabolo")


import warnings

warnings.filterwarnings("ignore")

from math import *
from math import *





# LIBRAIRIE PYHTON CLASSIQUES
import pandas as pd
import numpy as np
import matplotlib
import scipy
import platform
from collections import Counter
#ESTIMATEUR
import sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.externals import joblib

from sklearn.ensemble import  RandomForestClassifier
from sklearn import neighbors, datasets
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform
from scipy.stats import randint

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier


#TRAINING
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


#Evaluateur
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


#outils
from dask.diagnostics import ProgressBar
import logging
from datetime import datetime


#Metriques
from sklearn.model_selection import ShuffleSplit

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve
from sklearn import  metrics
from sklearn.metrics import  roc_auc_score
from sklearn.metrics import  classification_report

from sklearn.preprocessing import StandardScaler

#Outils

import time

import numpy as np




#Graphique
from matplotlib import pyplot
from matplotlib import pyplot as plt
import seaborn as sns






print('Operating system version....', platform.platform())
print("Python version is........... %s.%s.%s" % sys.version_info[:3])
print('scikit-learn version is.....', sklearn.__version__)
print('pandas version is...........', pd.__version__)
print('numpy version is............', np.__version__)
print('matplotlib version is.......', matplotlib.__version__)
print('scipy version is.......', scipy.__version__)







def jouer_aux_courses(allure_etudier, recalculer, autostart=0):

    print("Jouer_aux_courses")
    pourcentage_test_size = 0.25




    from sklearn.model_selection import train_test_split

    df2, index_col = charger_data(allure=allure_etudier,autostart=autostart)





    df_gagnant, feature_columns, response_column = creation_data_df_donnes(df2)



    #display_corr_with_col(df2, 'SELECTION')

    print("SPLIT DES DONNEES µµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµµ")
    train_x, test_x, train_y, test_y = split_des_data(df_gagnant, feature_columns, response_column, pourcentage_test_size)

    histo_feature(df_gagnant)

    fig, ax = plt.subplots(figsize=(22, 22))
    sns.heatmap(df_gagnant.corr(), annot=True, fmt=".2f", linewidths=.0, ax=ax, annot_kws={"size": 10}, xticklabels=1)

    # Train and Test dataset size details
    print("--------------------------------")
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)
    print("--------------------------------")

    model = fit_du_model(allure_etudier, train_x, train_y, test_x, test_y, recalculer,autostart)


    if recalculer==1:
        print(model)
        importances = pd.DataFrame({'feature': feature_columns, 'importance': np.round(model.feature_importances_, 3)})
        importances = importances.sort_values('importance', ascending=False).set_index('feature')

        print(importances)
        importances.plot.bar()
        y_pred = model.predict(test_x)
        predictions = [round(value) for value in y_pred]
        PROBA = model.predict_proba(test_x)
        # evaluate predictions
        accuracy = accuracy_score(test_y, predictions)
        print(">>>>>>>>>>  Accuracy: %.3f%%" % (accuracy * 100.0))
        print(classification_report(test_y, predictions))

    df_numero_a_predire = lecture_data('d:\data_jour2.csv', get_names(), ['IDPARTCIPANT', 'IDCOURSE'], allure_etudier,
                                          avec_index=True)

    df_numero_a_predire = transformationPrediction(df_numero_a_predire, allure_etudier,autostart=autostart)

    print('Fichier lu d:\data_jour2.csv')

    df_numero_a_predire['SELECTION'] = 0

    if allure_etudier == 1:
        df_numero_a_predire = suppression_pour_allure_1(df_numero_a_predire,autostart=autostart)

    if allure_etudier == 2:
        df_numero_a_predire = suppression_pour_allure_2(df_numero_a_predire)

    if allure_etudier == 3:
        df_numero_a_predire = suppression_pour_allure_3(df_numero_a_predire)

    if allure_etudier == 4:
        df_numero_a_predire = suppression_pour_allure_4(df_numero_a_predire)

    if allure_etudier == 5:
        df_numero_a_predire = suppression_pour_allure_5(df_numero_a_predire)

    print(" df_numero_a_predire SHAPE", df_numero_a_predire.shape)
    train_x, test_x, train_y, test_y = split_des_data(df_numero_a_predire, feature_columns, response_column, pourcentage_test_size=1)

    y_pred = model.predict(test_x)
    print(y_pred)

    df_pred = pd.DataFrame.from_dict(y_pred)
    test_copy = test_x.copy()  #################

    PROBA = model.predict_proba(test_x)

    df_proba = pd.DataFrame.from_dict(PROBA)


    df_final = pd.concat([df_proba, df_pred], axis=1)
    #print(df_final.head(10))

    test_x = drop_test(test_copy)

    test_y = copie_data(test_x=test_x, df_proba=df_proba, allure_etudier=allure_etudier)


    # print(" ")
    print(" Ecritures des fichiers ...")

    print(test_y.head(100))

    ecrire_pour_diabolo(test_x=test_y, allure_etudier=allure_etudier,autstart=autostart)
    print(" FIN...")


    #return df_gagnant, feature_columns, response_column,  train_x, test_x, train_y, test_y, model


def lecture_data(Fichier, xnames, xindex_col, allure=1,avec_index=True):
    import dask.dataframe as dd

    print("Lecture du fichier ==============> ",Fichier)
    print("")

    if avec_index==True:
        print("avec index ==============> ", Fichier)
        df = pd.read_csv(Fichier,  index_col=xindex_col,     sep=';',     names=xnames,header=None,skipinitialspace=True,    encoding='Latin-1',
                         dtype={
                             'ALLURE': np.int16,
                             'CO_DISTANCE': np.int16,
                             'CO_PRIX': str,
                             'HIPPO': str,
                             'IDCOURSE': np.int32,
                             'IDPARTCIPANT': np.int32,
                             'PAR_AGE': np.int8,
                             'PAR_ARRIVE': np.int8,
                             'PAR_CARRIERE': np.int32,
                             'PAR_CARRIERE_Q': np.int8,
                             'PAR_CLASSE_AGE': np.int32,
                             'PAR_COTEDER': np.int16,
                             'PAR_ENT_ECART_GAGNANT': np.int8,
                             'PAR_ENT_RAPPORT_GAGNANT_M': np.int16,
                             'PAR_ENT_REU_PLACE': np.int16,
                             'PAR_ENT_REUSSITE_GAGNE': np.int16,
                             'PAR_ENT_VICTOIRE': np.int16,
                             'PAR_GAIN': np.float,
                             'pAR_JOC_ECART_GAGNANT': np.int16,
                             'PAR_JOC_ECART_PLACE': np.int16,
                             'PAR_JOC_NB_COURSE': np.int16,
                             'PAR_JOC_PLACE_3P': np.int16,
                             'pAR_JOC_RAPPORT_GAGNANT_M': np.int16,
                             'PAR_JOC_REU_PLACE': np.int16,
                             'pAR_JOC_REUSSITE_GAGNE': np.int16,
                             'pAR_JOC_VICTOIRE': np.int16,
                             'PAR_NP': np.int16,
                             'PAR_NUM': np.int16,
                             'PAR_PLACE': np.int16,
                             'PAR_PLACE_Q': np.int16,
                             'PAR_REUSSITE_3P': np.int16,
                             'PAR_REUSSITE_GAGNE': np.int16,
                             'PAR_REUSSITE_QUINTE': np.int16,
                             'PAR_RUESSITE_PLACE': np.int16,
                             'autostart': np.int8,
                             'cendre': np.int8,
                             'grande_piste': np.int8,
                             'Point': np.int32,
                             'Nb_partant': str,
                             'PAR_PROPRIO': str,
                             'NOM_JOC': str,
                             'NOM_ENTR': str,
                             'POIDS': np.int16,
                             'CORDE': np.int16,
                             'CHEVAL': str,
                             'MUSIC_CHEVAL': np.int16,
                             'MUSIC_ENT': np.int16,
                             'MUSIC_JOC': np.int16,
                             'PAR_VALEUR': np.int16,
                             'PAR_ENT_ECART_PLACE': np.int8,
                             'PAR_VICTOIRE': np.int8,
                             'PAR_VICTOIRE_Q': np.int16,
                             'PAR_ENT_NB_COURSE': np.int16,
                             'POINTS_MUSIC': np.int16,
                             'MOIS_COURSE': np.int32,
                             'DATE_COURSE': object,
                             'PAR_SEXE':np.int16,
                             'FIN_ligne': str}


                         )


        df = df.groupby("ALLURE")
        df = df.get_group(allure)
        print("Slection des course du jour pour allure ",allure)
        print(df.shape)





    else:
        print("sans index ==============> ", Fichier)
        df = pd.read_csv(Fichier,    sep=';',  names=xnames, header=None, skipinitialspace=True,    encoding='Latin-1' ,
              dtype={
               'ALLURE':np.int16,
               'CO_DISTANCE':np.int16,
              'CO_PRIX':str,
              'HIPPO':str,
              'IDCOURSE':np.int32,
              'IDPARTCIPANT':np.int32,
              'PAR_AGE':np.int8,
              'PAR_ARRIVE':np.int8,
              'PAR_CARRIERE':np.int32,
              'PAR_CARRIERE_Q':np.int8,
              'PAR_CLASSE_AGE':np.int32,
              'PAR_COTEDER':np.int16,
              'PAR_ENT_ECART_GAGNANT':np.int8,
              'PAR_ENT_RAPPORT_GAGNANT_M':np.int16,
              'PAR_ENT_REU_PLACE':np.int16,
              'PAR_ENT_REUSSITE_GAGNE':np.int16,
              'PAR_ENT_VICTOIRE':np.int16,
              'PAR_GAIN':np.float,
              'pAR_JOC_ECART_GAGNANT':np.int16,
              'PAR_JOC_ECART_PLACE':np.int16,
              'PAR_JOC_NB_COURSE':np.int16,
              'PAR_JOC_PLACE_3P':np.int16,
              'pAR_JOC_RAPPORT_GAGNANT_M':np.int16,
              'PAR_JOC_REU_PLACE':np.int16,
              'pAR_JOC_REUSSITE_GAGNE':np.int16,
              'pAR_JOC_VICTOIRE':np.int16,
              'PAR_NP':np.int16,
              'PAR_NUM':np.int16,
              'PAR_PLACE':np.int16,
              'PAR_PLACE_Q':np.int16,
              'PAR_REUSSITE_3P':np.int16,
              'PAR_REUSSITE_GAGNE':np.int16,
              'PAR_REUSSITE_QUINTE':np.int16,
              'PAR_RUESSITE_PLACE':np.int16,
              'autostart':np.int8,
              'cendre':np.int8,
              'grande_piste':np.int8,
              'Point':np.int32,
              'Nb_partant':str,
              'PAR_PROPRIO':str,
              'NOM_JOC':str,
              'NOM_ENTR':str,
              'POIDS':np.int16,
              'CORDE':np.int16,
              'CHEVAL':str,
              'MUSIC_CHEVAL':np.int16,
              'MUSIC_ENT':np.int16,
              'MUSIC_JOC':np.int16,
              'PAR_VALEUR':np.int16,
              'PAR_ENT_ECART_PLACE':np.int8,
              'PAR_VICTOIRE':np.int8,
              'PAR_VICTOIRE_Q':np.int16,
              'PAR_ENT_NB_COURSE':np.int16,
              'POINTS_MUSIC':np.int16,
              'MOIS_COURSE':np.int32,
              'DATE_COURSE':object,
              'PAR_SEXE':np.int16,
              'FIN_ligne':str}
                     )
        df = df.groupby("ALLURE")
        df = df.get_group(allure)
        print("shape ETUDE :::: ")
        print(df.shape)








    print(" ****************** Fin de lecture data   :  ", Fichier)

    return df

# Creating bins for the win column
def assign_selection(W):
    if W==3:
        return 3
    if W==2:
        return 2
    if W==1:
        return 1

    return 0



def transformation(df2, allure_etudier,autostart=0):

    # Filtre des données¶
    print('\n\n----------- TRANSFORMATION --------- AJOUT DE COLONNE------------\n')
    print(df2.info())
    df2['SELECTION'] = 0



    start_time = timer()
    #Print selection des cote >0
    df2 = df2[df2.PAR_COTEDER > 0]
    df2 = df2[df2.PAR_ARRIVE > 0]  # On garde la ligne qui possede information arrivée 99 1 2 3 4 5

#    if allure_etudier == 1:
        #df2 = my_drop(df2, "PAR_COTEDER")

        # if autostart==1:
        #     #df2 = df2[df2.autostart==1]
        #     df2 = my_drop(df2, "autostart")
        # else:
        #     df2 = my_drop(df2, "autostart")

 #   if allure_etudier == 3:
  #      df2 = my_drop(df2, "PAR_COTEDER")


    if allure_etudier == 2:
        df2 = df2[df2.POIDS > 20]
        df2 = df2[df2.POIDS < 80]
        df2 = df2[df2.CORDE >  0]
        df2 = df2[df2.CORDE < 25]

        #df2 = my_drop(df2, "PAR_COTEDER")
        df2 = my_drop(df2, "CORDE")
        print('Gestion des poids et cordes')

    print('\n\n Définition de SELECTION ************')
    df2['SELECTION'] = df2['PAR_ARRIVE'].apply(assign_selection)
    print('')

    print("\nPAR ARRIVE est supprimé")

    df2 = my_drop(df2, 'PAR_ARRIVE')




    timer(start_time)


    print("DATA : -------------------------------------------------\n")
    print(df2.shape)



    return df2



def transformationPrediction(df2, allure_etudier,autostart=0):

    # Filtre des données¶
    print('\n\n----------- TRANSFORMATION --------- AJOUT DE COLONNE------------\n')
    print(df2.info())
    #df2['SELECTION'] = 0


    #df2 = df2[df2.CO_PRIX <= 500000]



    #df2 = my_drop(df2, "PAR_COTEDER")


#    df2 = df2[df2.PAR_ARRIVE > 0]  # On garde la ligne qui possede information arrivée 99 1 2 3 4 5
    if allure_etudier == 2:
        df2 = df2[df2.POIDS > 20]
        df2 = df2[df2.POIDS < 80]
        #df2 = df2[df2.CORDE >  0]
        #df2 = df2[df2.CORDE < 25]
        print('Gestion des poids et cordes')

    #if allure_etudier == 1:
    #    df2 = my_drop(df2, "PAR_COTEDER")
    #     #df2 = my_drop(df2, "PAR_COTEDER")
    #     if autostart == 1:
    #         #df2 = df2[df2.autostart == 1]
    #         df2 = my_drop(df2, "autostart")
    #     else:
    #         df2 = my_drop(df2, "autostart")




    return df2


#----------------------------------------------------------------------------------------
def charger_data(allure,autostart=0):

    index_col = ['IDPARTCIPANT', 'IDCOURSE']

    # lecture des données brutes contruite par l'application Windev
    # GetNames donne les champs a lire
    if (allure==1):
        nomfic='d:\data_jour_1.csv'
    if (allure==2):
        nomfic='d:\data_jour_2.csv'
    if (allure==3):
        nomfic='d:\data_jour_3.csv'
    if (allure==4):
        nomfic='d:\data_jour_4.csv'
    if (allure==5):
        nomfic='d:\data_jour_5.csv'





    df2 = lecture_data(nomfic, get_names(), xindex_col=index_col, allure=allure, avec_index=False)

    df2 = df2.groupby("ALLURE")
    df2 = df2.get_group(allure)
    df2 = my_drop(df2, 'ALLURE')

    print('Transformation------\n')


    if allure==1:
        if autostart==1:
            print("Valeur de autostart = ",autostart)
        else:
            print("Pas de d'autostart ")


        df2 = suppression_pour_allure_1(df2,autostart=autostart)




    if allure==2:
        df2 = suppression_pour_allure_2(df2)

    if allure == 3:
        df2 = suppression_pour_allure_3(df2)

    if allure==4:
        df2 = suppression_pour_allure_4(df2)
    if allure==5:
        df2 = suppression_pour_allure_5(df2)




    df2=transformation(df2, allure,autostart=autostart)





    print(df2.shape)


    return df2, index_col


def suppression_pour_allure_2(df2):

    print('\n suppression_pour_allure_2')
    df2 = my_drop(df2, "FIN_ligne")
    df2 = my_drop(df2, "PAR_NP")
    df2 = my_drop(df2, "PAR_PROPRIO")
    df2 = my_drop(df2, "HIPPO")
    df2 = my_drop(df2, "NOM_JOC")
    df2 = my_drop(df2, "CO_DISTANCE")
    df2 = my_drop(df2, "NOM_ENTR")
    df2 = my_drop(df2, "PAR_GAIN")
    df2 = my_drop(df2, "PAR_NOM")
    df2 = my_drop(df2, "CHEVAL")
    #df2 = my_drop(df2, "PAR_VALEUR")
    df2 = my_drop(df2, "grande_piste")
    #df2 = my_drop(df2, "PAR_POIDS")
    df2 = my_drop(df2, "CO_PRIX")
    df2 = my_drop(df2, "PAR_CLASSE_AGE")
    df2 = my_drop(df2, "autostart")
    df2 = my_drop(df2, "cendre")
    df2 = my_drop(df2, "PAR_VICTOIRE_Q")
    #df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
    df2 = my_drop(df2, "PAR_JOC_VICTOIRE")
    df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
    # #df2 = my_drop(df2, "MOIS_COURSE")
    df2 = my_drop(df2, "DATE_COURSE")
    #
    #df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
    #df2 = my_drop(df2, "CORDE")
    df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT")
    df2 = my_drop(df2, "PAR_PLACE_Q")
    # df2 = my_drop(df2, "MOIS_COURSE")
    df2 = my_drop(df2, "PAR_CARRIERE_Q")
    #df2 = my_drop(df2, "Nb_partant")
    df2 = my_drop(df2, "MUSIC_ENT")
    #df2 = my_drop(df2, "MUSIC_JOC")
    #df2 = my_drop(df2, "PAR_JOC_PLACE_3P")
    df2 = my_drop(df2, "PAR_ENT_VICTOIRE")
    df2 = my_drop(df2, "PAR_ENT_RAPPORT_GAGNANT_M")
    df2 = my_drop(df2, "PAR_JOC_RAPPORT_GAGNANT_M")
    # df2 = my_drop(df2, "PAR_ENT_NB_COURSE")
    df2 = my_drop(df2, "PAR_ENT_REUSSITE_GAGNE")
    #df2 = my_drop(df2, "PAR_PLACE")
    df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
    #df2 = my_drop(df2, "PAR_VICTOIRE")
    #df2 = my_drop(df2, "PAR_REUSSITE_QUINTE")
    df2 = my_drop(df2, "PAR_JOC_REUSSITE_GAGNE")
    df2 = my_drop(df2, "Nb_partant")
    #





    return df2



def suppression_pour_allure_4(df2):

    print('\n suppression_pour_allure_2')
    df2 = my_drop(df2, "FIN_ligne")
    df2 = my_drop(df2, "PAR_NP")
    df2 = my_drop(df2, "PAR_PROPRIO")
    df2 = my_drop(df2, "HIPPO")
    df2 = my_drop(df2, "NOM_JOC")
    df2 = my_drop(df2, "CO_DISTANCE")
    df2 = my_drop(df2, "NOM_ENTR")
    df2 = my_drop(df2, "PAR_GAIN")
    df2 = my_drop(df2, "PAR_NOM")
    df2 = my_drop(df2, "CHEVAL")
    df2 = my_drop(df2, "CO_PRIX")

    #df2 = my_drop(df2, "PAR_VALEUR")
    df2 = my_drop(df2, "grande_piste")
    df2 = my_drop(df2, "PAR_NUM")
    df2 = my_drop(df2, "PAR_CLASSE_AGE")
    df2 = my_drop(df2, "autostart")
    df2 = my_drop(df2, "cendre")

    df2 = my_drop(df2, "MUSIC_JOC")
    df2 = my_drop(df2, "DATE_COURSE")
    df2 = my_drop(df2, "Nb_partant")


    df2 = my_drop(df2, "CORDE")



    return df2


def suppression_pour_allure_5(df2):

    print('\n suppression_pour_allure_2')
    df2 = my_drop(df2, "FIN_ligne")
    df2 = my_drop(df2, "PAR_NP")
    df2 = my_drop(df2, "PAR_PROPRIO")
    df2 = my_drop(df2, "HIPPO")
    df2 = my_drop(df2, "NOM_JOC")
    df2 = my_drop(df2, "CO_DISTANCE")
    df2 = my_drop(df2, "NOM_ENTR")
    df2 = my_drop(df2, "PAR_GAIN")
    df2 = my_drop(df2, "PAR_NOM")
    df2 = my_drop(df2, "CHEVAL")
    df2 = my_drop(df2, "CO_PRIX")

    df2 = my_drop(df2, "PAR_VALEUR")
    df2 = my_drop(df2, "grande_piste")
    df2 = my_drop(df2, "PAR_NUM")
    df2 = my_drop(df2, "PAR_CLASSE_AGE")
    df2 = my_drop(df2, "autostart")
    df2 = my_drop(df2, "cendre")
    #df2 = my_drop(df2, "PAR_VICTOIRE_Q")
    df2 = my_drop(df2, "MUSIC_JOC")
    df2 = my_drop(df2, "DATE_COURSE")


    df2 = my_drop(df2, "CORDE")

    return df2





def suppression_pour_allure_1(df2,autostart=0):
    print('')


    print ('suppression des collones vraiments inutiles')

    df2 = my_drop(df2, "FIN_ligne")
    df2 = my_drop(df2, "PAR_NP")
    df2 = my_drop(df2, "PAR_PROPRIO")
    df2 = my_drop(df2, "HIPPO") #NON
    df2 = my_drop(df2, "NOM_JOC")

    #if autostart!=1:
    #    df2 = my_drop(df2, "CO_DISTANCE") # trop d' irrégularité

    df2 = my_drop(df2, "NOM_ENTR") # NON
    #df2 = my_drop(df2, "PAR_GAIN")
    df2 = my_drop(df2, "PAR_NOM")
    df2 = my_drop(df2, "CHEVAL")
    df2 = my_drop(df2, "PAR_VALEUR")
    #df2 = my_drop(df2, "grande_piste")
    df2 = my_drop(df2, "PAR_NUM")
    df2 = my_drop(df2, "PAR_CLASSE_AGE")

    #df2 = my_drop(df2, "IDCOURSE")
    #df2 = my_drop(df2, "IDPARTCIPANT")

    df2 = my_drop(df2, "cendre")  # suppression des collones vraiments inutiles
    df2 = my_drop(df2, "PAR_AGE")
    df2 = my_drop(df2, "Nb_partant")
    df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
    # #df2 = my_drop(df2, "MUSIC_ENT")
    df2 = my_drop(df2, "PAR_VICTOIRE_Q")
    # #df2 = my_drop(df2, "MUSIC_JOC")
    df2 = my_drop(df2, "PAR_VALEUR")
    #df2 = my_drop(df2, "autostart")
    #
    # # pour 1
    df2 = my_drop(df2, "POIDS")
    df2 = my_drop(df2, "CORDE")
    #df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
    df2 = my_drop(df2, "PAR_CARRIERE_Q")
    #

    #df2 = my_drop(df2, "CO_DISTANCE")
    df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT")
    df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
    #df2 = my_drop(df2, "MOIS_COURSE")
    #
    df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT_M")
    df2 = my_drop(df2, "PAR_ENT_RAPPORT_GAGNANT_M")
    #
    # df2 = my_drop(df2, "PAR_CARRIERE")
    # df2 = my_drop(df2, "PAR_PLACE")
    #
    # df2 = my_drop(df2, "PAR_PLACE")
    df2 = my_drop(df2, "CO_PRIX")
    #df2 = my_drop(df2, "PAR_REUSSITE_QUINTE")
    # #df2 = my_drop(df2, "POINTS_MUSIC")
    #
    # df2 = my_drop(df2, "pAR_JOC_VICTOIRE")
    #
    # df2 = my_drop(df2, "MUSIC_JOC")
    #
    # df2 = my_drop(df2, "MUSIC_ENT")
    #
    #
    df2 = my_drop(df2, "PAR_PLACE_Q")
    #
    df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
    #
    df2 = my_drop(df2, "pAR_JOC_REUSSITE_GAGNE")
    df2 = my_drop(df2, "PAR_ENT_REUSSITE_GAGNE")
    df2 = my_drop(df2, "PAR_REUSSITE_QUINTE")
    #
    df2 = my_drop(df2, "pAR_JOC_RAPPORT_GAGNANT_M")
    df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
    df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
    df2 = my_drop(df2, "PAR_ENT_NB_COURSE")
    #
    df2 = my_drop(df2, "PAR_VICTOIRE")
    df2 = my_drop(df2, "pAR_JOC_VICTOIRE")
    df2 = my_drop(df2, "pAR_JOC_REUSSITE_GAGNE")
    #df2 = my_drop(df2, "Point")
    #

    df2 = my_drop(df2, "DATE_COURSE")
    df2 = my_drop(df2, "PAR_AGE")


    df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
    # df2 = my_drop(df2, "PAR_ENT_VICTOIRE")
    #
    df2 = my_drop(df2, "MOIS_COURSE")


    #
    #
    #
    # #df2 = my_drop(df2, "pAR_JOC_REUSSITE_GAGNE")
    #
    #


    return df2

def suppression_pour_allure_3(df2):
    print('')


    df2 = my_drop(df2, "autostart")
    print('')


    print ('suppression des collones vraiments inutiles')

    df2 = my_drop(df2, "FIN_ligne")
    df2 = my_drop(df2, "PAR_NP")
    df2 = my_drop(df2, "PAR_PROPRIO")
    df2 = my_drop(df2, "HIPPO") #NON
    df2 = my_drop(df2, "NOM_JOC")
    df2 = my_drop(df2, "CO_DISTANCE") # trop d' irrégularité
    df2 = my_drop(df2, "NOM_ENTR") # NON
    df2 = my_drop(df2, "PAR_GAIN")
    df2 = my_drop(df2, "PAR_NOM")
    df2 = my_drop(df2, "CHEVAL")
    df2 = my_drop(df2, "PAR_VALEUR")
    df2 = my_drop(df2, "grande_piste")
    df2 = my_drop(df2, "PAR_NUM")
    df2 = my_drop(df2, "PAR_CLASSE_AGE")
    #df2 = my_drop(df2, "PAR_VALEUR")



    #df2 = my_drop(df2, "IDCOURSE")
    #df2 = my_drop(df2, "IDPARTCIPANT")

    df2 = my_drop(df2, "cendre")  # suppression des collones vraiments inutiles
    # df2 = my_drop(df2, "PAR_AGE")
    df2 = my_drop(df2, "Nb_partant")
    df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
    # #df2 = my_drop(df2, "MUSIC_ENT")
    df2 = my_drop(df2, "PAR_VICTOIRE_Q")
    # #df2 = my_drop(df2, "MUSIC_JOC")
    # #df2 = my_drop(df2, "PAR_PLACE_Q")
    #df2 = my_drop(df2, "autostart")
    #
    # # pour 1
    df2 = my_drop(df2, "POIDS")
    df2 = my_drop(df2, "CORDE")
    #df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
    df2 = my_drop(df2, "PAR_CARRIERE_Q")
    #
    df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT")
    df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
    #df2 = my_drop(df2, "MOIS_COURSE")
    #
    df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT_M")
    df2 = my_drop(df2, "PAR_ENT_RAPPORT_GAGNANT_M")
    #
    # df2 = my_drop(df2, "PAR_CARRIERE")
    # df2 = my_drop(df2, "PAR_PLACE")
    #
    # df2 = my_drop(df2, "PAR_PLACE")
    df2 = my_drop(df2, "CO_PRIX")
    df2 = my_drop(df2, "PAR_REUSSITE_QUINTE")
    # #df2 = my_drop(df2, "POINTS_MUSIC")
    #
    # df2 = my_drop(df2, "pAR_JOC_VICTOIRE")
    #
    # df2 = my_drop(df2, "MUSIC_JOC")
    #
    # df2 = my_drop(df2, "MUSIC_ENT")
    #
    #
    df2 = my_drop(df2, "PAR_PLACE_Q")
    #
    df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
    #
    # df2 = my_drop(df2, "pAR_JOC_REUSSITE_GAGNE")
    # df2 = my_drop(df2, "PAR_ENT_REUSSITE_GAGNE")
    # df2 = my_drop(df2, "PAR_REUSSITE_QUINTE")
    #df2 = my_drop(df2, "pAR_JOC_RAPPORT_GAGNANT_M")
    df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
    df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
    # #df2 = my_drop(df2, "PAR_ENT_NB_COURSE")
    # df2 = my_drop(df2, "PAR_VICTOIRE_Q")
    #df2 = my_drop(df2, "Point")
    #df2 = my_drop(df2, "autostart")
    df2 = my_drop(df2, "DATE_COURSE")
    df2 = my_drop(df2, "PAR_AGE")
    df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
    # df2 = my_drop(df2, "PAR_ENT_VICTOIRE")

    df2 = my_drop(df2, "MOIS_COURSE")
    # #df2 = my_drop(df2, "pAR_JOC_REUSSITE_GAGNE")
    #
    #



    return df2

def assign_selection2(W):
    if W==1:
        return 1
    if W==0:
        return 0


def creation_data_df_donnes(df_gagnant):
    #df_gagnant['SELECTION'] = df_gagnant['SELECTION2'].apply(assign_selection2)
    #df_gagnant.drop(["SELECTION2"], axis=1, inplace=True)

    index_col = ['IDPARTCIPANT', 'IDCOURSE']
    df_gagnant = df_gagnant.set_index(index_col)
    print("DEFINITION DE LA STRUCTURE ---------------------------------------------------------")
    print("(1) Shape df_gagnant  ", df_gagnant.shape, "\n")

    df_gagnant_len = len(df_gagnant.columns) - 1
    Lib_features = df_gagnant.columns[:df_gagnant_len]

    feature_columns = Lib_features  ##<<<<<<<<<<<<<<<<
    response_column = ['SELECTION']  ##<<<<<<<<<<<<<<<<


    print("(2) FEATURES ", Lib_features)
    print("(3) response column ", response_column)

    print('\n Détection de valeurs NULLEs \n ---------------------')
    print('')

    print(df_gagnant.isnull().sum())

    print('\n\n Proportions ------------\n')

    target_count = df_gagnant['SELECTION'].value_counts()

    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
    target_count.plot(kind='bar', title='Count (target)')
    normal_trans_perc0 = sum(df_gagnant['SELECTION'] == 0) / (sum(df_gagnant['SELECTION'] == 0) + sum(df_gagnant['SELECTION'] == 1) + sum(df_gagnant['SELECTION'] == 2)+ sum(df_gagnant['SELECTION'] == 3))
    normal_trans_perc1 = sum(df_gagnant['SELECTION'] == 1) / (sum(df_gagnant['SELECTION'] == 0) + sum(df_gagnant['SELECTION'] == 1) + sum(df_gagnant['SELECTION'] == 2)+ sum(df_gagnant['SELECTION'] == 3))
    normal_trans_perc2 = sum(df_gagnant['SELECTION'] == 2) / (sum(df_gagnant['SELECTION'] == 0) + sum(df_gagnant['SELECTION'] == 1) + sum(df_gagnant['SELECTION'] == 2)+ sum(df_gagnant['SELECTION'] == 3))
    normal_trans_perc3 = sum(df_gagnant['SELECTION'] == 3) / (sum(df_gagnant['SELECTION'] == 0) + sum(df_gagnant['SELECTION'] == 1) + sum(df_gagnant['SELECTION'] == 2)+ sum(df_gagnant['SELECTION'] == 3))

    fraud_trans_perc = 1 - normal_trans_perc1 - normal_trans_perc2-normal_trans_perc3


    print('')
    print('')

    print('Total number of records : {} '.format(len(df_gagnant)))
    print('Nombre de participations avec SELECTION = 0 : {}'.format(sum(df_gagnant['SELECTION'] == 0)))
    print('Nombre de participations avec SELECTION = 1  : {}'.format(sum(df_gagnant['SELECTION'] == 1)))
    print('Nombre de participations avec SELECTION = 2  : {}'.format(sum(df_gagnant['SELECTION'] == 2)))
    print('Nombre de participations avec SELECTION = 3  : {}'.format(sum(df_gagnant['SELECTION'] == 3)))
    print('')
    #print('Pourcentage 0: {:.4f}%,  pourcentage 1 : {:.4f}% ,  pourcentage 2 : {:.4f}% ,  pourcentage 3 : {:.4f}%'.
    #      format(normal_trans_perc0 * 100),format(normal_trans_perc0 * 100),format(normal_trans_perc1 * 100),format(normal_trans_perc2 * 100),format(normal_trans_perc3 * 100))




    return  df_gagnant, feature_columns, response_column










def split_des_data_matrix(df_gagnant,feature_columns,response_column,pourcentage_test_size=0.3):
    # train_x, test_x, train_y, test_y = jj.split_data2(df_gagnant, feature_columns,response_column)
    from sklearn.model_selection import train_test_split

    print('\n')
    print("SPLIT des données ....\n")

    train_x, test_x, train_y, test_y = train_test_split(df_gagnant[feature_columns], df_gagnant[response_column], test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(train_x, label=train_y)

    dtest = xgb.DMatrix(test_x, label=test_y)

    return  dtrain, dtest, train_y, test_y





def split_des_data(df_gagnant,feature_columns,response_column,pourcentage_test_size=0.3):
    # train_x, test_x, train_y, test_y = jj.split_data2(df_gagnant, feature_columns,response_column)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    from sklearn.preprocessing import QuantileTransformer


    print('\n')
    print("SPLIT DES DATA .....")

    train_x, test_x, train_y, test_y = train_test_split(df_gagnant[feature_columns], df_gagnant[response_column], train_size=1-pourcentage_test_size)


    #if pourcentage_test_size < 1:
        #train_y = train_y['SELECTION'].ravel()
     #   train_x, train_y = smot2(train_x, train_y, feature_columns)


    test_y = test_y['SELECTION'].ravel()
    train_y = train_y['SELECTION'].ravel()



    #train_test_split(dataset[feature_headers], dataset[target_header],
     #                train_size=train_percentage, test_size=None, random_state=42)


    if pourcentage_test_size<1:
        print('Fit de Train X')
        print("")
        #scaler = RobustScaler(quantile_range=(30,70)).fit(train_x)
        #scaler = RobustScaler(quantile_range=(20, 80)).fit(train_x)
        scaler =StandardScaler().fit(train_x)

        #test_x, test_y = smot2(test_x, test_y, feature_columns)
        X_train_scaled = pd.DataFrame(scaler.transform(train_x), index=train_x.index.values, columns=train_x.columns.values)
        train_x = X_train_scaled

        print("sCALER Train_x === ", train_x.shape)
        print("SCALER Train y === ", train_y.shape)

        scaler = StandardScaler().fit(test_x)
        X_test_scaled = pd.DataFrame(scaler.transform(test_x), index=test_x.index.values, columns=test_x.columns.values)
        test_x = X_test_scaled


        print("")
        print("")
        print("Test x", test_x.shape)
        print("Test y", test_y.shape)
        print("")
        print("")




    if pourcentage_test_size==1:

        print('Fit de Test X')
        scaler = StandardScaler().fit(test_x)
        #scaler = RobustScaler(quantile_range=(20, 80)).fit(test_x)
        print("Scaler ok")

        X_test_scaled = pd.DataFrame(scaler.transform(test_x), index=test_x.index.values, columns=test_x.columns.values)
        test_x = X_test_scaled

        print("")
        print("Test x", test_x.shape)
        print("Test y", test_y.shape)









    print("Number of training records: ", len(train_x))
    print("Number of test records: ", len(test_x))



    return  train_x, test_x, train_y, test_y









def fit_du_model(allure_etudier,train_x, train_y, test_x, test_y, recalcule=1,autostart=0):
    print("Fit du model ....")
    if  autostart==0:
        print("Sans autoStart")
    else:
        print("Avec autoStart")



    if recalcule==1:
        print("RECALCULER ......................")
        print("RECALCULER ......................")
        print("RECALCULER ......................")
        print("")

        model = get_estimator(allure_etudier,autostart)
        model = fit_estimator(model, train_x, train_y, test_x, test_y)
        save_mymodel(clf=model,allure=allure_etudier,autostart=autostart)


        return  model

    print("MODELE CHARGE PAR FICHIER")
    print("")

    model = load_mymodel(allure_etudier, autostart)
    return model







def evaluation_du_model_matrix(model, train_x, train_y,test_x,test_y,feature_columns):
    from sklearn.metrics import precision_score


    y_pred = model.predict(test_x)

    # extracting most confident predictions

    best_preds = np.asarray([np.argmax(line) for line in y_pred])

    print('Numpy array precision:', precision_score(test_y, best_preds, average='macro'))



    predictions = [round(value) for value in y_pred]
    #PROBA = model.predict_proba(test_x)
    # evaluate predictions
    accuracy = accuracy_score(test_y, predictions)
    print(">>>>>>>>>>  Accuracy: %.3f%%" % (accuracy * 100.0))
    print(classification_report(test_y, predictions))

    print("Roc auc score ")
    y_true = test_y
    print(roc_auc_score(test_y, y_pred))



def evaluation_du_model(model, train_x, train_y,test_x,test_y,feature_columns):

    y_pred = model.predict(test_x)
    predictions = [round(value) for value in y_pred]
    PROBA = model.predict_proba(test_x)
    # evaluate predictions
    accuracy = accuracy_score(test_y, predictions)
    print(">>>>>>>>>>  Accuracy: %.3f%%" % (accuracy * 100.0))
    print(classification_report(test_y, predictions))

    from sklearn.metrics import f1_score

    print("Roc auc score ")
    y_true = test_y
    print(roc_auc_score(test_y, y_pred))
    print("\nf1 score")
    print(f1_score(y_true, y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='weighted'))

    print(f1_score(y_true, y_pred, average=None))
    print("\nconfusion matrix")
    print(confusion_matrix(y_true, y_pred))
    print("\nprecision score *******************************")
    print(metrics.precision_score(y_true, y_pred))
    print("\nrecall score")
    print(metrics.recall_score(y_true, y_pred))
    print("\nf1 score")
    print(metrics.f1_score(y_true, y_pred))
    print("\nbeta score")
    print(metrics.fbeta_score(y_true, y_pred, beta=0.5))
    print(metrics.fbeta_score(y_true, y_pred, beta=1))
    print(metrics.fbeta_score(y_true, y_pred, beta=2))
    print("\nprecision recall")
    print(metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5))


    print("")
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)

    # plot log loss
    # fig, ax = pyplot.subplots()
    # ax.plot(x_axis, results['validation_0']['error'], label='Train')
    # ax.plot(x_axis, results['validation_1']['error'], label='Test')
    # ax.legend()
    # pyplot.ylabel('auc')
    # pyplot.title('XGBoost Error')
    # pyplot.show()
    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    pyplot.ylabel('AUC')
    pyplot.title('XGBoost Classification Auc')
    pyplot.show()

    affiche_evalSet(model, train_x, train_y, test_x, test_y, 1)
    affiche_evalSet(mybest_model=model, set_train=train_x, set_train_cible=train_y, set_test=test_x, set_test_cible=test_y, type_eval=2)

    importances = pd.DataFrame({'feature': feature_columns, 'importance': np.round(model.feature_importances_, 3)})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')

    print(importances)
    importances.plot.bar()





def prediction_course_du_jour(model,allure,feature_columns,response_column,autstart=0):

    df_numero_a_predire = lecture_data(Fichier='d:\data_jour2.csv',xnames=get_names(),xindex_col=['IDPARTCIPANT', 'IDCOURSE'], allure=allure,
                                          avec_index=True)



    print("Lecture des données du jour ok  ----------------------------------------------------------------")

    #df_numero_a_predire = scalerise(df_numero_a_predire, allure=allure)
    df_numero_a_predire= transformationPrediction(df_numero_a_predire, allure,autstart=autstart)
    print("*** Fichier lu d:\data_jour2.csv")



    df_numero_a_predire['SELECTION'] = 0

    print("Extrait fichier")
    print(df_numero_a_predire.head(10))



    if allure == 1:
        df_numero_a_predire = suppression_pour_allure_1(df_numero_a_predire)
    if allure == 2:
        df_numero_a_predire = suppression_pour_allure_2(df_numero_a_predire)

    if allure == 3:
        df_numero_a_predire = suppression_pour_allure_3(df_numero_a_predire)

    if allure == 4:
        df_numero_a_predire = suppression_pour_allure_4(df_numero_a_predire)

    if allure == 5:
        df_numero_a_predire = suppression_pour_allure_5(df_numero_a_predire)


    #train_x, test_x, train_y, test_y = split_dataset(df2_journee, 0, feature_columns, response_column)


    print(" df_numero_a_predire SHAPE", df_numero_a_predire.shape)


    train_x, test_x, train_y, test_y =split_des_data(df_numero_a_predire, feature_columns, response_column, pourcentage_test_size=1)

    #print("SHAPE apres SPLIT ", test_x.shape())

#    test_x=train_x

    #print(test_x)

    #test_y = test_y['SELECTION'].ravel()
    print("PREDICTIONS ....")
    y_pred = model.predict(test_x)
    print(y_pred)


    df_pred = pd.DataFrame.from_dict(y_pred)
    test_copy = test_x.copy()  #################

    PROBA = model.predict_proba(test_x)

    df_proba = pd.DataFrame.from_dict(PROBA)
    df_final = pd.concat([df_proba, df_pred], axis=1)
    print(df_final.head(5))


    test_x = drop_test(test_copy)

    test_y = copie_data(test_x=test_x, df_proba=df_proba, allure_etudier=allure)
    #print(" ")
    print(" ECRIRE POUR DIABOLO ...")

    print(test_y.head(100))



    ecrire_pour_diabolo(test_x=test_y, allure_etudier=allure,autstart=autstart)
    print(" FIN...")









def importance_features_model(model, feature_columns):
    importances = pd.DataFrame({'feature': feature_columns, 'importance': np.round(model.feature_importances_, 3)})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')

    print(importances)
    importances.plot.bar()



def histo_feature(df2):
    import matplotlib.pyplot as plt

    df2[df2.dtypes[(df2.dtypes == "float64") | (df2.dtypes == "int64")| (df2.dtypes == "int8")| (df2.dtypes == "int32")| (df2.dtypes == "int")]
        .index.values].hist(figsize=[20, 20])


def matrice_correlation(df_gagnant):
    fig, ax = plt.subplots(figsize=(22, 22))
    sns.heatmap(df_gagnant.corr(), annot=True, fmt=".2f", linewidths=.0, ax=ax, annot_kws={"size": 13}, xticklabels=1)

    display_corr_with_col(df_gagnant, 'SELECTION')


def evaluate_model(alg, train_x, target, feature_columns, cv_folds=5, early_stopping_rounds=1):
    import time

    start_time = timer(None)

    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(train_x[feature_columns].values, target['SELECTION'].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                      metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
    alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(train_x[feature_columns], target['SELECTION'], eval_metric='auc')
    print('')



    timer(start_time)




    # Predict training set:
    dtrain_predictions = alg.predict(train_x[feature_columns])
    dtrain_predprob = alg.predict_proba(train_x[feature_columns])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("\nprécision : %.4g" % metrics.accuracy_score(target['SELECTION'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(target['SELECTION'], dtrain_predprob))

    print()
    print()



    importances = pd.DataFrame({'feature': feature_columns, 'importance': np.round(alg.feature_importances_, 3)})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')

    print(importances)
    importances.plot.bar()



    # feat_imp = pd.Series(alg.booster().get_fscore())
    #
    # feat_imp.plot(kind='bar', title='Feature Importance', color='g')
    # plt.ylabel('Feature Importance Score')










def affiche_evalSet(mybest_model,set_train,set_train_cible,set_test,set_test_cible,type_eval):

    set_test_cible_predicted = mybest_model.predict(set_test)

    set_train_cible_predicted = mybest_model.predict(set_train)

    print('la précision est :  ', mybest_model.score(set_test, set_test_cible))  # accuracy
               # print("TRAINING roc_auc_score :  %s" % auc)



    for x, y in [(set_train, set_train_cible), (set_test, set_test_cible)]:
             yp = mybest_model.predict(x)
             cm = confusion_matrix(y, yp.ravel())
             print(cm)

#plt.matshow(cm)
#   plt.title('Confusion matrix')
    #    plt.colorbar()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')

    if type_eval==1:
        ntotal = len(set_test)
        correct = set_test_cible == set_test_cible_predicted
        numCorrect = sum(correct)
        percent = round((100.0 * numCorrect) / ntotal, 3)
        print()
        print("\ Classification Correcte des données de test : {0:d}/{1:d}  {2:8.3f}%".format(numCorrect, ntotal, percent))
        prediction_score = 100.0 * mybest_model.score(set_test, set_test_cible)
        print('\n***************** Score  TEST  : %8.3f  ************************' % prediction_score)
    else:
        ntotal = len(set_train)
        correct = set_train_cible== set_train_cible_predicted
        numCorrect = sum(correct)
        percent = round((100.0 * numCorrect) / ntotal, 3)
        print()
        print("CLASSIFICATION CORRECTE DES DONNEES DE Train  : {0:d}/{1:d}  {2:8.3f}%".format(numCorrect, ntotal, percent))
        prediction_score = 100.0 * mybest_model.score(set_train, set_train_cible)
        print('\n ************* Score  TRAINING  : %8.3f  ************************' % prediction_score)



def display_corr_with_col(df, col):

    print(df.head(10))

    correlation_matrix = df.corr()
    correlation_type = correlation_matrix[col].copy()
    abs_correlation_type = correlation_type.apply(lambda x: abs(x))
    desc_corr_values = abs_correlation_type.sort_values(ascending=False)
    y_values = list(desc_corr_values.values)[1:]
    x_values = range(0,len(y_values))
    xlabels = list(desc_corr_values.keys())[1:]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x_values, y_values)
    ax.set_title('The correlation of all features with {}'.format(col), fontsize=20)
    ax.set_ylabel('Pearson correlatie coefficient [abs waarde]', fontsize=16)
    plt.xticks(x_values, xlabels, rotation='vertical')
    plt.show()



def save_mymodel(clf, allure,autostart=0):
    print("Sauvegarde modele ...")

    joblib.dump(clf, 'diabolo' + str(allure) +str(autostart)+ '.pkl')
    print("Sauvegarde modele ...ok")


def load_mymodel(allure, autostart=0):
    print("Restauration modele ...")
    clf = joblib.load('diabolo' + str(allure) +str(autostart)+ '.pkl')
    print("Restauration modele ...ok")

    return clf




def my_randomSearch(model, train_x, train_y, param_dist, n_iter_search,allure_etudier,scoring):
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        verbose=20,
        n_jobs=2,
        cv=10,
        refit=True,
        n_iter=n_iter_search,
        scoring=scoring
    )

    model_search = random_search.fit(train_x, train_y)
    model_search = model_search.best_estimator_

    save_mymodel(model_search, allure_etudier)

    return model_search


def get_estimator4 (allure_etudier):
    from sklearn.neural_network import MLPClassifier

    estimator = MLPClassifier(hidden_layer_sizes=(30, 30, 30,30,30,30),solver='adam',learning_rate='adaptive',learning_rate_init=0.001,max_iter=300,verbose=True , momentum =0.95)
    return estimator

## Helper functions for evaluating classifiers
def cross_val_predictions(est, X, y, cv):

	y_preds = np.zeros(y.shape)
	y_probas = np.zeros(y.shape)
	for train_idx, valid_idx in cv:
		print (X[train_idx].shape, y[train_idx].shape)
		est.fit(X[train_idx], y[train_idx])
		y_preds[valid_idx] = est.predict(X[valid_idx])
		y_probas[valid_idx] = est.predict_proba(X[valid_idx])[:,1]
	return y_preds, y_probas





def fit_estimator212(model,train_x, train_y, test_x,test_y):
    print("fit")
    model.fit(train_x, train_y)


    print(model)

    predictions = model.predict(test_x)
    print(confusion_matrix(test_y, predictions))

    # evaluate predictions
    accuracy = accuracy_score(test_y, predictions)
    print(">>>>>>>>>>  Accuracy: %.3f%%" % (accuracy * 100.0))

    print(classification_report(test_y, predictions))


    return model



def fit_estimator(model,train_x, train_y, test_x, test_y):

    print("recherche par modele")
    start_time = timer()

    eval_set = [(train_x, train_y), (test_x, test_y)]
    print("fit")
    #eval_metric = ["rmse"], eval_set = eval_set,
    model.fit(train_x, train_y, verbose=True )

    timer(start_time)

    y_pred = model.predict(test_x)
    predictions = [round(value) for value in y_pred]
    PROBA = model.predict_proba(test_x)
    # evaluate predictions
    accuracy = accuracy_score(test_y, predictions)
    print("")

    print(">>>>>>>>>>  Accuracy: %.3f%%" % (accuracy * 100.0))
    print(classification_report(test_y, predictions))
    return model


def colorie(X, model, ax, fig):
    import numpy

    xmin, xmax = numpy.min(X[:, 0]), numpy.max(X[:, 0])
    ymin, ymax = numpy.min(X[:, 1]), numpy.max(X[:, 1])
    hx = (xmax - xmin) / 100
    hy = (ymax - ymin) / 100
    xx, yy = numpy.mgrid[xmin:xmax:hx, ymin:ymax:hy]
    grid = numpy.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])



def resultat_baseTest(model, X_test, y_test):

    fig = plt.figure(figsize=(7, 5))
    ax = plt.subplot()
    colorie(X_test, model, ax, fig)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    ax.set_title("Résultats sur la base de test")





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):




        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt



#estimator =     XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
 #                 colsample_bytree=0.0001, gamma=0.9, learning_rate=0.3,
  #                max_delta_step=1, max_depth=3, min_child_weight=1, missing=nan,
   #               n_estimators=300, n_jobs=3, nthread=None,
    #              objective='binary:logistic', random_state=0, reg_alpha=7,
     #             reg_lambda=1, scale_pos_weight=1, seed=27, silent=True,
      #            subsample=0.9)


def get_estimator (allure_etudier, autostart=0):
    print("GET ESTIMATOR du model ....",allure_etudier)
    print("")


    if allure_etudier == 1:
        if autostart==0:
            estimator =XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1.0,
       colsample_bytree=0.7000000000000001, gamma=0.310000001,
       learning_rate=0.054999999999999986, max_delta_step=5, max_depth=10,
       min_child_weight=0, missing=None, n_estimators=123, n_jobs=3,
       nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0.210000001, reg_lambda=617, scale_pos_weight=1371,
       seed=943, silent=1, subsample=0.52)

        else:
            estimator =XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1.0,
       colsample_bytree=0.7000000000000001, gamma=0.310000001,
       learning_rate=0.054999999999999986, max_delta_step=5, max_depth=10,
       min_child_weight=0, missing=None, n_estimators=123, n_jobs=3,
       nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0.210000001, reg_lambda=617, scale_pos_weight=1371,
       seed=943, silent=1, subsample=0.52)


     # estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
     #               colsample_bytree=0.1, gamma=0.07, learning_rate=0.06,
     #               max_delta_step=15, max_depth=7, min_child_weight=4, missing=None,
     #               n_estimators=500, n_jobs=1, nthread=None,
     #               objective='binary:logistic', random_state=0, reg_alpha=0,
     #               reg_lambda=1, scale_pos_weight=0.09, seed=200, silent=True,
     #               subsample=0.6)
     # estimator=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
     #              colsample_bytree=0.1, gamma=0.07, learning_rate=0.025,
     #              max_delta_step=15, max_depth=8, min_child_weight=7, missing=None,
     #              n_estimators=1000, n_jobs=1, nthread=None,
     #              objective='binary:logistic', random_state=0, reg_alpha=0,
     #              reg_lambda=1, scale_pos_weight=0.09, seed=50, silent=True,
     #              subsample=0.6)

    #estimator = XGBClassifier()



    if allure_etudier == 3:
        estimator =XGBClassifier(base_score=0.5, booster='gbtree',
       colsample_bylevel=0.9734840756213539,
       colsample_bytree=0.8884730697680153, gamma=1.5120321064827247e-09,
       learning_rate=0.03450688894339603, max_delta_step=19, max_depth=4,
       min_child_weight=4, missing=None, n_estimators=919, n_jobs=3,
       nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0.003918268678531666, reg_lambda=1.5364727285778806e-09,
       scale_pos_weight=7.039252554215881e-06, seed=560, silent=1,
       subsample=0.2763022159415584)

    #estimator = XGBClassifier()



    if allure_etudier == 2:
        estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1.0,
       colsample_bytree=0.79, gamma=0.020000001,
       learning_rate=0.04499999999999999, max_delta_step=8, max_depth=6,
       min_child_weight=4, missing=None, n_estimators=357, n_jobs=3,
       nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0.060000001, reg_lambda=621, scale_pos_weight=410,
       seed=513, silent=1, subsample=0.42000000000000004)





    #estimator = XGBClassifier()




    if allure_etudier == 4:
        estimator = XGBClassifier(base_score=0.5, booster='gbtree',
                                    colsample_bylevel=0.9940987636114856,
                               colsample_bytree=0.9947781087274736,
                                  gamma=0,
                                  learning_rate=0.030457733797784645,
                                  max_delta_step=19,
                                  max_depth=9,
                                  min_child_weight=4,
                                  missing=None,
                                  n_estimators=476,
                                n_jobs=3,
                                  objective='multi:softprob',
                                  random_state=0,
                                       reg_alpha=01.3899635404083775e-09,
                                  reg_lambda=7.012515421940738e-09,
                                  scale_pos_weight=0.5859400371427048,
                                 silent=True,
                                  subsample=0.591831499173718)
   # estimator = XGBClassifier()

    if allure_etudier == 5:
        estimator = XGBClassifier(base_score=0.5, booster='gbtree',
                                    colsample_bylevel=0.9940987636114856,
                               colsample_bytree=0.9947781087274736,
                                  gamma=0,
                                  learning_rate=0.030457733797784645,
                                  max_delta_step=19,
                                  max_depth=9,
                                  min_child_weight=4,
                                  missing=None,
                                  n_estimators=476,
                                n_jobs=3,
                                  objective='multi:softprob',
                                  random_state=0,
                                       reg_alpha=01.3899635404083775e-09,
                                  reg_lambda=7.012515421940738e-09,
                                  scale_pos_weight=0.5859400371427048,
                                 silent=True,
                                  subsample=0.591831499173718)
    #estimator = XGBClassifier()


    print(estimator)

    return estimator





def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        print(start_time)
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def split_dataset(dataset, train_percentage, feature_headers,
                                  target_header,random_state=42,mode_debug=0):

    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage, test_size=None, random_state=42)




    if mode_debug==1:
        # Train and Test dataset size details
        print("--------------------------------")
        print("Train_x Shape :: ", train_x.shape)
        print("Train_y Shape :: ", train_y.shape)
        print("Test_x Shape :: ", test_x.shape)
        print("Test_y Shape :: ", test_y.shape)
        print("--------------------------------")

    return train_x, test_x, train_y, test_y


def my_drop(df, col):
    if col in df:
        df.drop([col], axis=1, inplace=True)
        print(col,"  Supprimé")
    return df






def suppression_colonne(df2, allure):
    # df=my_drop(df, "PAR_AGE")
    print('Suppression colonnne ', allure)

    if allure == 0:
        df2 = my_drop(df2,"FIN_ligne")
        df2 = my_drop(df2, "PAR_NP")
        df2 = my_drop(df2, "cendre")

        #df2 = my_drop(df2, "MUSIC_ENT")
        #df2 = my_drop(df2, "MUSIC_JOC")
        #df2 = my_drop(df2, "PAR_COTEDER")

        df2 = my_drop(df2, "grande_piste")
        df2 = my_drop(df2, "PAR_VALEUR")

        df2 = my_drop(df2, "FIN_ligne")
        df2 = my_drop(df2, "HIPPO")
        df2 = my_drop(df2, "CO_PRIX")

        df2 = my_drop(df2, "FIN_ligne")
        df2 = my_drop(df2, 'PAR_GAIN')
        df2 = my_drop(df2, 'PAR_CLASSE_AGE')


        #df2 = my_drop(df2, 'Point')
        df2 = my_drop(df2, 'PAR_CARRIERE_Q')
        df2 = my_drop(df2, 'PAR_PLACE_Q')
        df2 = my_drop(df2, "PAR_ENT_RAPPORT_GAGNANT_M")
        df2 = my_drop(df2, "pAR_JOC_RAPPORT_GAGNANT_M")
        df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT_M")
        df2 = my_drop(df2, 'PAR_PROPRIO')
        df2 = my_drop(df2, 'CHEVAL')
        df2 = my_drop(df2, 'NOM_ENTR')
        df2 = my_drop(df2, 'NOM_JOC')
        df2 = my_drop(df2, 'PAR_PLACE')
        #df2 = my_drop(df2, 'POINTS_MUSIC')








        df2 = my_drop(df2, 'aux')


    if allure == 1 or allure == 3:
        df2 = my_drop(df2, "POIDS")
        df2 = my_drop(df2, "CORDE")
        df2 = my_drop(df2, "CO_DISTANCE")
        #if allure == 3:
        df2 = my_drop(df2, "autostart")

        df2 = my_drop(df2, "MUSIC_CHEVAL")
        df2 = my_drop(df2, "MUSIC_ENT")
        df2 = my_drop(df2, "MUSIC_JOC")
        #df2 = my_drop(df2, 'POINTS_MUSIC')

        # df2 = my_drop(df2, "pAR_JOC_REUSSITE_GAGNE")
        # df2 = my_drop(df2, "pAR_JOC_VICTOIRE")
        if allure == 1:
            df2 = my_drop(df2, "PAR_VICTOIRE")
            df2 = my_drop(df2, "Nb_partant")
            df2 = my_drop(df2, "PAR_ENT_REU_PLACE")


        #
        # df2 = my_drop(df2, "PAR_REUSSITE_3P")
        # df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
        # df2 = my_drop(df2, "PAR_ENT_REU_PLACE")
        # df2 = my_drop(df2, "PAR_REUSSITE_GAGNE")
        # df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
        #
        df2 = my_drop(df2, "PAR_REUSSITE_QUINTE")

        # #
        df2 = my_drop(df2, "PAR_NUM")
        # df2 = my_drop(df2, "PAR_ENT_VICTOIRE")
        # #
        # df2 = my_drop(df2, "Nb_partant")
        # #
        # df2 = my_drop(df2, "PAR_JOC_REU_PLACE")
        #
        # df2 = my_drop(df2, "PAR_ENT_REUSSITE_GAGNE")
        #
        df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT")
        df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
        df2 = my_drop(df2, 'PAR_CARRIERE')
        # df2 = my_drop(df2, 'PAR_REUSSITE_QUINTE')
        df2 = my_drop(df2, 'PAR_VICTOIRE_Q')
        #df2 = my_drop(df2, 'PAR_ENT_NB_COURSE')


        # #
        #df2 = my_drop(df2, 'pAR_JOC_VICTOIRE')
        # #
        # # df2 = my_drop(df2, 'Nb_partant')
        # df2 = my_drop(df2, 'PAR_NUM')
        # #
        df2 = my_drop(df2, "PAR_AGE")
        #df2 = my_drop(df2, "PAR_CARRIERE")
        df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
        # # # df2 = my_drop(df2, "Nb_partant")
        df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
        #
        #
        # #
        # df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT")
        # df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
        # #
        # # df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT_M")
        #df2 = my_drop(df2, "PAR_VICTOIRE")
        #df2 = my_drop(df2, "PAR_ENT_VICTOIRE")
        #df2 = my_drop(df2, "pAR_JOC_REUSSITE_GAGNE")
        #df2 = my_drop(df2, "PAR_ENT_REUSSITE_GAGNE")




        #df2 = my_drop(df2, "PAR_PLACE")

    if allure == 2:
        df2 = my_drop(df2, "MY_auto_start")
        df2 = my_drop(df2, "autostart")
        df2 = my_drop(df2, 'PAR_NUM')
        df2 = my_drop(df2, 'CO_DISTANCE')
        #df2 = my_drop(df2, "PAR_COTEDER")
        df2 = my_drop(df2, "autostart")
        df2 = my_drop(df2, "PAR_VICTOIRE_Q")

        df2 = my_drop(df2, "MUSIC_CHEVAL")
        df2 = my_drop(df2, "MUSIC_ENT")
        df2 = my_drop(df2, "MUSIC_JOC")
        # df2 = my_drop(df2, 'POINTS_MUSIC')

        #df2 = my_drop(df2, 'Nb_partant')
        # df2 = my_drop(df2, 'PAR_JOC_ECART_PLACE')
        df2 = my_drop(df2, 'PAR_VICTOIRE')

        df2 = my_drop(df2, 'PAR_REUSSITE_QUINTE')
        # df2 = my_drop(df2, 'PAR_VICTOIRE_Q')
        # df2 = my_drop(df2, 'PAR_ENT_NB_COURSE')
        #
        # df2 = my_drop(df2, 'PAR_JOC_REU_PLACE')
        # df2 = my_drop(df2, 'pAR_JOC_REUSSITE_GAGNE')
        #
        # df2 = my_drop(df2, 'PAR_ENT_ECART_GAGNANT')
        #df2 = my_drop(df2, 'PAR_ENT_VICTOIRE')
        # df2 = my_drop(df2, 'PAR_ENT_REU_PLACE')
        df2 = my_drop(df2, 'PAR_CARRIERE')
        df2 = my_drop(df2, 'PAR_ENT_REUSSITE_GAGNE')
        df2 = my_drop(df2, 'PAR_ENT_ECART_PLACE')
        #df2 = my_drop(df2, 'pAR_JOC_VICTOIRE')

        # df2 = my_drop(df2, 'pAR_JOC_ECART_GAGNANT')
        # df2 = my_drop(df2, "CHEVAL")
        # df2 = my_drop(df2, 'PAR_PROPRIO')
        # df2 = my_drop(df2, 'NOM_ENTR')
        # df2 = my_drop(df2, 'NOM_JOC')
        #df2 = my_drop(df2, 'PAR_REUSSITE_QUINTE')
        #df2 = my_drop(df2, "PAR_NUM")
        # df2 = my_drop(df2, "CORDE")
        df2 = my_drop(df2, "Nb_partant")
        # f2 = my_drop(df2, 'PAR_VICTOIRE_Q')

        df2 = my_drop(df2, "PAR_AGE")
        #df2 = my_drop(df2, "PAR_CARRIERE")
        df2 = my_drop(df2, "pAR_JOC_ECART_GAGNANT")
        df2 = my_drop(df2, "PAR_ENT_ECART_GAGNANT")
        df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
        # df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
        # df2 = my_drop(df2, "PAR_PLACE")
        # df2 = my_drop(df2, "PAR_REUSSITE_3P")
        # df2 = my_drop(df2, "PAR_ENT_NB_COURSE")
        # df2 = my_drop(df2, "PAR_JOC_NB_COURSE")

        df2 = my_drop(df2, "PAR_VICTOIRE_Q")


    if allure == 4:
        df2 = my_drop(df2, "CORDE")
        df2 = my_drop(df2, "POIDS")
        df2 = my_drop(df2, "MY_auto_start")
        df2 = my_drop(df2, "MY_auto_start")
        df2 = my_drop(df2, "autostart")
        df2 = my_drop(df2, 'PAR_NUM')
        df2 = my_drop(df2, 'CO_DISTANCE')
        df2 = my_drop(df2, 'PAR_REUSSITE_QUINTE')
        df2 = my_drop(df2, "PAR_JOC_ECART_PLACE")
        df2 = my_drop(df2, "PAR_ENT_ECART_PLACE")
        df2 = my_drop(df2, "PAR_VICTOIRE_Q")

    if allure == 5:
        df2 = my_drop(df2, "CORDE")
        df2 = my_drop(df2, "POIDS")
        df2 = my_drop(df2, "MY_auto_start")

    return df2




def somme_note(df, num_course, cri):
    df.loc[df['IDCOURSE'] == num_course, 'p2018'] = df.loc[df['IDCOURSE'] == num_course, 'p2018'] + df.loc[
        df['IDCOURSE'] == num_course, cri]

    return df


def smot2(train_x, train_y, feature_columns):
    start_time = timer()
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    from imblearn.over_sampling import ADASYN



    # print('\nOriginal dataset shape {}'.format(Counter(train_y)))
    sm = SMOTEENN(ratio='minority', n_jobs=3, random_state=42, smote=SMOTE(k_neighbors=20))




    sm = SMOTE(ratio='minority', n_jobs=3, random_state=42,k_neighbors=20)
    sm = SMOTE(random_state=2)



    X_res, y_res = sm.fit_sample(train_x, train_y)

    train_x = pd.DataFrame(X_res, columns=feature_columns)
    train_y = pd.Series(y_res)
    print("Fin SMOT")
    timer(start_time)

    return train_x, train_y




def scalerise(df_gagnant, allure):
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler, QuantileTransformer
    scaled_features = df_gagnant.copy()
    features = scaled_features[get_critere_scale(allure)]

    # scaler = MinMaxScaler(feature_range=(0, 400)).fit(features.values)
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))

    features = scaler.fit_transform(features.values)
    scaled_features[get_critere_scale(allure)] = features


    df_gagnant = scaled_features
    return df_gagnant



def get_critere_scale(allure):


    if (allure==1 or allure==3 ):


        return [


            'PAR_ENT_REU_PLACE',
            'PAR_ENT_REUSSITE_GAGNE',
            'PAR_ENT_VICTOIRE',

            'PAR_JOC_NB_COURSE',
            'PAR_JOC_PLACE_3P',
            'PAR_JOC_REU_PLACE',
            'pAR_JOC_REUSSITE_GAGNE',
            'pAR_JOC_VICTOIRE',

            'PAR_REUSSITE_3P',
            'PAR_REUSSITE_GAGNE',
          #  'PAR_REUSSITE_QUINTE',
            'PAR_RUESSITE_PLACE',
            'Point',
            'Nb_partant',
           # 'MUSIC_ENT',
           # 'MUSIC_JOC',
            'PAR_VICTOIRE',

            'PAR_ENT_NB_COURSE'


        ]



    if (allure==2):
        return [


            'PAR_ENT_REU_PLACE',
            'PAR_ENT_VICTOIRE',
            'PAR_JOC_NB_COURSE',
            'PAR_JOC_PLACE_3P',
            'PAR_JOC_REU_PLACE',
            'pAR_JOC_REUSSITE_GAGNE',
            'pAR_JOC_VICTOIRE',
            'PAR_REUSSITE_3P',
            'PAR_REUSSITE_GAGNE' ,

            'PAR_RUESSITE_PLACE',
            'Point',
            'POIDS',
            'CORDE',
            'PAR_ENT_NB_COURSE',
            'POINTS_MUSIC'



        ]


    return     [
                   'PAR_REUSSITE_GAGNE',
                   'PAR_CARRIERE',
                   'PAR_RUESSITE_PLACE',
                   'PAR_REUSSITE_3P',
                   'PAR_ENT_ECART_GAGNANT',
                   'PAR_ENT_REU_PLACE',
                   'pAR_JOC_ECART_GAGNANT',
                   'PAR_JOC_REU_PLACE',
                   'PAR_ENT_VICTOIRE',
                   'PAR_JOC_PLACE_3P',
                   'pAR_JOC_REUSSITE_GAGNE',
                   'PAR_ENT_REUSSITE_GAGNE',
                   'pAR_JOC_VICTOIRE',
                   'PAR_JOC_ECART_PLACE',
                   'PAR_ENT_ECART_PLACE',
                   'PAR_VICTOIRE',
                   'PAR_REUSSITE_QUINTE',
                    'PAR_ENT_NB_COURSE',
                    'PAR_JOC_NB_COURSE',
                    'PAR_AGE',
                    'Nb_partant',
        'PAR_NUM',

                   'PAR_VICTOIRE_Q']



def encodage(df_gagnant,scaler):


    #from sklearn.preprocessing import RobustScaler
    #from sklearn.preprocessing import quantile_transform
    #scaler = RobustScaler()



    critere_scale=[
                   'PAR_REUSSITE_GAGNE',
                   'PAR_CARRIERE',
                   'PAR_RUESSITE_PLACE',
                   'PAR_REUSSITE_3P',
                   'PAR_ENT_ECART_GAGNANT',
                   'PAR_ENT_REU_PLACE',
                   'pAR_JOC_ECART_GAGNANT',
                   'PAR_JOC_REU_PLACE',
                   'PAR_ENT_VICTOIRE',
                   'PAR_JOC_PLACE_3P',
                   'pAR_JOC_REUSSITE_GAGNE',
                   'PAR_ENT_REUSSITE_GAGNE',
                   'pAR_JOC_VICTOIRE',
                   'PAR_JOC_ECART_PLACE',
                   'PAR_ENT_ECART_PLACE',
                   'PAR_VICTOIRE',
                   'PAR_REUSSITE_QUINTE',
                    'PAR_ENT_NB_COURSE',
                    'PAR_JOC_NB_COURSE',
                    'PAR_AGE',
                    'Nb_partant',
        'PAR_NUM',

                   'PAR_VICTOIRE_Q']

    # copie du DATAFRAME
    #scaled_features = df_gagnant.copy()
    #features = scaled_features[critere_scale]
    #scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)

    scaled_features[critere_scale] = features


    #df_scale1 = df_gagnant[critere_scale]
    #df_scale2 = StandardScaler().fit_transform(df_scale1.values)

    #scaled_features_df = pd.DataFrame(df_scale2, index=df_gagnant.index, columns=df_gagnant.columns)



    #df_gagnant[critere_scale] = df_gagnant[critere_scale].apply(lambda x: StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(reshape(-1, 1)))


    return scaled_features

















    #df_gagnant[critere_scale] = quantile_transform(df_gagnant[critere_scale], n_quantiles=4)

    #df_gagnant[critere_scale] = qt.fit_transform(df_gagnant[critere_scale])
    #df_gagnant[critere_scale] = quantile_transform(df_gagnant[critere_scale], n_quantiles=5, random_state=42,subsample=300000)

    df_gagnant[['PAR_ENT_REU_PLACE', 'PAR_ENT_REUSSITE_GAGNE',
                'PAR_JOC_NB_COURSE', 'PAR_JOC_PLACE_3P','PAR_JOC_REU_PLACE',
                'pAR_JOC_REUSSITE_GAGNE', 'PAR_REUSSITE_3P', 'PAR_REUSSITE_GAGNE','PAR_RUESSITE_PLACE',
                'PAR_VICTOIRE', 'PAR_ENT_NB_COURSE', 'p2018']] = quantile_transform(df_gagnant[['PAR_ENT_REU_PLACE',
                    'PAR_ENT_REUSSITE_GAGNE', 'PAR_ENT_VICTOIRE','PAR_JOC_NB_COURSE', 'PAR_JOC_PLACE_3P','PAR_JOC_REU_PLACE',
                                    'pAR_JOC_REUSSITE_GAGNE', 'PAR_REUSSITE_3P', 'PAR_REUSSITE_GAGNE','PAR_RUESSITE_PLACE','PAR_VICTOIRE',
                                                                                    'PAR_ENT_NB_COURSE', 'p2018']],n_quantiles=10)


    return df_gagnant



def calcul_note(df, num_course, cri):
    df.loc[df['IDCOURSE'] == num_course, 'aux'] = df.loc[df['IDCOURSE'] == num_course, cri[0]]
    valeur_critere = df.loc[df['IDCOURSE'] == num_course, cri]  # selection critere
    maxx = valeur_critere.max()  # MAXI

    if ((maxx[0]) > 0):
        df.loc[df['IDCOURSE'] == num_course, cri] = (df.loc[df['IDCOURSE'] == num_course, cri] / maxx[0]) * 20
    else:
        df.loc[df['IDCOURSE'] == num_course, cri] = 0

    df = somme_note(df, num_course, cri[0])
    df.loc[df['IDCOURSE'] == num_course, cri[0]] = df.loc[df['IDCOURSE'] == num_course, 'aux']

    return df


def calcul_les_notes(df, allure):
    i = 0
    start_time = timer()
    df['p2018'] = 0
    df['aux'] = 0

    for n in df.groupby(['IDCOURSE'], axis=0):
        nc = n[0]


        if math.fmod(i, 800) == 0:
            print("    print(N° ", i, timer(start_time))

        if (allure == 1):
            df = calcul_note(df, nc, ['PAR_ENT_REU_PLACE'])
            df = calcul_note(df, nc, ['PAR_ENT_REUSSITE_GAGNE'])
            df = calcul_note(df, nc, ['PAR_ENT_VICTOIRE'])
            df = calcul_note(df, nc, ['PAR_ENT_NB_COURSE'])
            df = calcul_note(df, nc, ['PAR_JOC_REU_PLACE'])
            df = calcul_note(df, nc, ['pAR_JOC_REUSSITE_GAGNE'])
            df = calcul_note(df, nc, ['PAR_JOC_NB_COURSE'])
            df = calcul_note(df, nc, ['PAR_JOC_PLACE_3P'])
            df = calcul_note(df, nc, ['PAR_REUSSITE_GAGNE'])
            df = calcul_note(df, nc, ['PAR_RUESSITE_PLACE'])
            df = calcul_note(df, nc, ['PAR_REUSSITE_3P'])
            df = calcul_note(df, nc, ['PAR_CARRIERE_Q'])
            df = calcul_note(df, nc, ['PAR_PLACE_Q'])
            df = calcul_note(df, nc, ['PAR_CLASSE_AGE'])
            df = calcul_note(df, nc, ['PAR_VICTOIRE'])

        if (allure == 2):
            df = calcul_note(df, nc, ['PAR_ENT_REU_PLACE'])
            df = calcul_note(df, nc, ['PAR_ENT_REUSSITE_GAGNE'])
            df = calcul_note(df, nc, ['PAR_ENT_VICTOIRE'])
            df = calcul_note(df, nc, ['PAR_ENT_NB_COURSE'])
            df = calcul_note(df, nc, ['PAR_JOC_REU_PLACE'])
            df = calcul_note(df, nc, ['pAR_JOC_REUSSITE_GAGNE'])
            df = calcul_note(df, nc, ['pAR_JOC_VICTOIRE'])
            df = calcul_note(df, nc, ['PAR_JOC_NB_COURSE'])
            df = calcul_note(df, nc, ['PAR_JOC_PLACE_3P'])
            df = calcul_note(df, nc, ['PAR_REUSSITE_GAGNE'])
            df = calcul_note(df, nc, ['PAR_RUESSITE_PLACE'])
            df = calcul_note(df, nc, ['PAR_REUSSITE_3P'])
            df = calcul_note(df, nc, ['PAR_CARRIERE'])
            df = calcul_note(df, nc, ['PAR_CARRIERE_Q'])
            df = calcul_note(df, nc, ['PAR_PLACE_Q'])
            df = calcul_note(df, nc, ['PAR_CLASSE_AGE'])
            df = calcul_note(df, nc, ['PAR_VICTOIRE'])

            # df = calcul_note(df, nc, ['PAR_REUSSITE_QUINTE'])
            # df = calcul_note(df, nc, ['PAR_VICTOIRE_Q'])


        i = i + 1

    return df


def nettoyer_Nan(df):
    df[['PAR_ENT_ECART_GAGNANT']] = df[['PAR_ENT_ECART_GAGNANT']].fillna(df[['PAR_ENT_ECART_GAGNANT']].mean())
    df[['PAR_ENT_RAPPORT_GAGNANT_M']] = df[['PAR_ENT_RAPPORT_GAGNANT_M']].fillna(df[['PAR_ENT_RAPPORT_GAGNANT_M']].mean())
    df[['PAR_ENT_REU_PLACE']] = df[['PAR_ENT_REU_PLACE']].fillna(df[['PAR_ENT_REU_PLACE']].mean())
    df[['PAR_ENT_REUSSITE_GAGNE']] = df[['PAR_ENT_REUSSITE_GAGNE']].fillna(df[['PAR_ENT_REUSSITE_GAGNE']].mean())
    df[['PAR_ENT_VICTOIRE']] = df[['PAR_ENT_VICTOIRE']].fillna(df[['PAR_ENT_VICTOIRE']].mean())
    df[['PAR_ENT_NB_COURSE']] = df[['PAR_ENT_NB_COURSE']].fillna(df[['PAR_ENT_NB_COURSE']].mean())

    df[['pAR_JOC_ECART_GAGNANT']] = df[['pAR_JOC_ECART_GAGNANT']].fillna(df[['pAR_JOC_ECART_GAGNANT']].mean())
    df[['pAR_JOC_RAPPORT_GAGNANT_M']] = df[['pAR_JOC_RAPPORT_GAGNANT_M']].fillna(df[['pAR_JOC_RAPPORT_GAGNANT_M']].mean())
    df[['PAR_JOC_REU_PLACE']] = df[['PAR_JOC_REU_PLACE']].fillna(df[['PAR_JOC_REU_PLACE']].mean())
    df[['pAR_JOC_REUSSITE_GAGNE']] = df[['pAR_JOC_REUSSITE_GAGNE']].fillna(df[['pAR_JOC_REUSSITE_GAGNE']].mean())
    df[['pAR_JOC_VICTOIRE']] = df[['pAR_JOC_VICTOIRE']].fillna(df[['pAR_JOC_VICTOIRE']].mean())
    df[['PAR_JOC_ECART_PLACE']] = df[['PAR_JOC_ECART_PLACE']].fillna(df[['PAR_JOC_ECART_PLACE']].mean())
    df[['PAR_JOC_NB_COURSE']] = df[['PAR_JOC_NB_COURSE']].fillna(df[['PAR_JOC_NB_COURSE']].mean())
    df[['PAR_JOC_PLACE_3P']] = df[['PAR_JOC_PLACE_3P']].fillna(df[['PAR_JOC_PLACE_3P']].mean())
    df['PAR_RUESSITE_PLACE'] = df['PAR_RUESSITE_PLACE'].fillna(df['PAR_RUESSITE_PLACE'].mean())
    df[['PAR_REUSSITE_GAGNE']] = df[['PAR_REUSSITE_GAGNE']].fillna(0)
    df[['PAR_REUSSITE_3P']] = df[['PAR_REUSSITE_3P']].fillna(df[['PAR_REUSSITE_3P']].mean())
    df[['PAR_CARRIERE']] = df[['PAR_CARRIERE']].fillna(df[['PAR_CARRIERE']].mean())
    df[['PAR_CARRIERE_Q']] = df[['PAR_CARRIERE_Q']].fillna(df[['PAR_CARRIERE_Q']].mean())
    df[['PAR_GAIN']] = df[['PAR_GAIN']].fillna(df[['PAR_GAIN']].mean())
    df[['PAR_PLACE']] = df[['PAR_PLACE']].fillna(df[['PAR_PLACE']].mean())
    df[['PAR_PLACE_Q']] = df[['PAR_PLACE_Q']].fillna(df[['PAR_PLACE_Q']].mean())
    df[['PAR_CLASSE_AGE']] = df[['PAR_CLASSE_AGE']].fillna(df[['PAR_CLASSE_AGE']].mean())
    df[['PAR_POINT']] = df[['PAR_POINT']].fillna(df[['PAR_POINT']].mean())
    df[['PAR_REUSSITE_QUINTE']] = df[['PAR_REUSSITE_QUINTE']].fillna(df[['PAR_REUSSITE_QUINTE']].mean())
    df[['PAR_VICTOIRE_Q']] = df[['PAR_VICTOIRE_Q']].fillna(df[['PAR_VICTOIRE_Q']].mean())
    df[['PAR_ENT_ECART_PLACE']] = df[['PAR_ENT_ECART_PLACE']].fillna(df[['PAR_ENT_ECART_PLACE']].mean())

    return df


import math


def relation_1(a, b):
    if b == 0:
        return 0
    else:
        return (a / b) * 20


def plot_importance(feature_columns, model):
    importances = pd.DataFrame({'feature': feature_columns, 'importance': np.round(model.feature_importances_, 3)})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    print("")
    print(importances)
    importances.plot.bar()


def metrique_classe(y_pred,y_true,xclass):
    from imblearn.metrics import specificity_score
    from imblearn.metrics import sensitivity_score
    from imblearn.metrics import geometric_mean_score
    # La sensibilité est le rapport où est le nombre de vrais positifs et le nombre de faux négatifs.
    # La sensibilité quantifie la capacité à éviter les faux négatifs.tp
    # estimator issu de quelques FIT
    print("Sensibilité  du re-equilibrage des données sur le TEST")
    #log.traceLogInfo("Binary ",sensitivity_score(y_true, y_pred, average='binary', pos_label=xclass))

    print("La spécificité est intuitivement la capacité du classificateur à trouver tous les échantillons positifs")
    print("Binary ")
    print(specificity_score(y_true, y_pred, labels=None, pos_label=xclass, average='binary', sample_weight=None))
    print("\nCalculer la moyenne géométrique")
    print(geometric_mean_score(y_true, y_pred,labels=None, pos_label=xclass))
    print("\n Calculer  sensitivity score")
    print("La sensibilité est le rapport où est le nombre de vrais positifs et le nombre de faux négatifs.")
    print("La sensibilité quantifie la capacité à éviter les faux négatifs.")
    print(sensitivity_score(y_true, y_pred, labels=None, pos_label=xclass,average='binary'))




def sauvegarde(df2,allure,xnames2):
    start_time = timer(start_time)
    if allure == 1:
        df2.to_csv('d:\diabolo_1_note.csv', sep=';', columns=xnames2, header=False)
    if allure == 2:
        df2.to_csv('d:\diabolo_2_note.csv', sep=';', columns=xnames2, header=False)
    if allure == 3:
        df2.to_csv('d:\diabolo_3_note.csv', sep=';', columns=xnames2, header=False)

    timer()
    return df2



def ecrire_pour_diabolo(test_x,allure_etudier,autstart=0):


    print("\n\n fin de copie sur Test_x\n\n", test_x)
    nom_fichier=""


    start_time = timer()

    if (allure_etudier == 1):
        nom_fichier = "d:\py_resultat_trot.csv"
        ##if autstart==0:

        #else:
        #    nom_fichier = "d:\py_resultat_trot1.csv"





    if (allure_etudier == 2):
        nom_fichier = "d:\py_resultat_galop.csv"


    if (allure_etudier == 3):
        nom_fichier = "d:\py_resultat_trot_monte.csv"



    if (allure_etudier == 4):
        nom_fichier="d:\py_resultat_haie.csv"



    if (allure_etudier == 5):
        nom_fichier = "d:\py_resultat_steeple.csv"



    nom_fichier = nom_fichier




    print("\n ecrire ", nom_fichier,"********************************************************************************")
    test_x.to_csv(nom_fichier)


    timer(start_time)








def copie_data(test_x, df_proba,allure_etudier):
    start_time = timer()

    cumul = 1
    nb_rows = len(df_proba.index)
    print("Nomnre de lignes =", nb_rows)
    print(" FORMATION DU FICHIER pour ", allure_etudier)
    for i in range(0, nb_rows):
        n = df_proba[0][i]
        test_x['v0'][i] = n

        n = df_proba[1][i]
        test_x['v1'][i] = n

        n = df_proba[2][i]
        test_x['v2'][i] = n

        n = df_proba[3][i]
        test_x['v3'][i] = n

    timer(start_time)
    print("Fin copie")


    return test_x







def restauration(allure_etudier,xnames2):
    start_time = timer()

    if allure_etudier == 1:
        df2 = pd.read_csv('d:\diabolo_1_note.csv', sep=';', names=xnames2, skipinitialspace=True, encoding='utf-8')

        df2 = df2.drop(df2.index[0])
        df2.head(10)

    if allure_etudier == 2:
        df2 = pd.read_csv('d:\diabolo_2_note.csv', sep=';', names=xnames2, skipinitialspace=True, encoding='utf-8')

        df2 = df2.drop(df2.index[0])
        df2.head(10)

    if allure_etudier == 3:
        df2 = pd.read_csv('d:\diabolo_3_note.csv', sep=';', names=xnames2, skipinitialspace=True, encoding='utf-8')

        df2 = df2.drop(df2.index[0])
        df2.head(10)

    timer(start_time)

    return df2



def importance_features( train_x, train_y, test_x, test_y):



    from numpy import sort
    from sklearn.metrics import accuracy_score
    from sklearn.feature_selection import SelectFromModel

    model = XGBClassifier()
    model.fit(train_x, train_y)

    # make predictions for test data and evaluate
    pred_y = model.predict(test_x)
    predictions = [round(value) for value in pred_y]
    accuracy = metrics.accuracy_score(test_y, predictions)
    print("RFC Accuracy: %.2f%%" % (accuracy * 100.0))

    # fit model using each importance as a threshold
    thresholds = sort(model.feature_importances_)
    for thresh in thresholds:
        # selecting features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_train_x = selection.transform(train_x)

        # training model
        selection_model = XGBClassifier()

        selection_model.fit(select_train_x, train_y)

        # evaluating model
        select_test_x = selection.transform(test_x)
        pred_y = selection_model.predict(select_test_x)
        predictions = [round(value) for value in pred_y]
        accuracy = metrics.accuracy_score(test_y, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_train_x.shape[1], accuracy * 100.0))



def conversion(df2):
    df2['CO_PRIX'] = df2['CO_PRIX'].astype('int')
    df2['HIPPO'] = df2['HIPPO'].astype('int')

    df2['IDCOURSE'] = df2['IDCOURSE'].astype('int')
    df2['IDPARTCIPANT'] = df2['IDPARTCIPANT'].astype('int')
    df2['PAR_AGE'] = df2['PAR_AGE'].astype('int')
    df2['PAR_CARRIERE'] = df2['PAR_CARRIERE'].astype('float')
    df2['PAR_CARRIERE'] = df2['PAR_CARRIERE'].astype('int')
    df2['PAR_CARRIERE_Q'] = df2['PAR_CARRIERE_Q'].astype('float')
    df2['PAR_CARRIERE_Q'] = df2['PAR_CARRIERE_Q'].astype('int')
    df2['PAR_CLASSE_AGE'] = df2['PAR_CLASSE_AGE'].astype('float')
    df2['PAR_CLASSE_AGE'] = df2['PAR_CLASSE_AGE'].astype('int')
    df2['PAR_COTEDER'] = df2['PAR_COTEDER'].astype('float')
    df2['PAR_ENT_ECART_GAGNANT'] = df2['PAR_ENT_ECART_GAGNANT'].astype('float')
    df2['PAR_ENT_ECART_GAGNANT'] = df2['PAR_ENT_ECART_GAGNANT'].astype('int')
    df2['PAR_ENT_RAPPORT_GAGNANT_M'] = df2['PAR_ENT_RAPPORT_GAGNANT_M'].astype('float')
    df2['PAR_ENT_RAPPORT_GAGNANT_M'] = df2['PAR_ENT_RAPPORT_GAGNANT_M'].astype('int')
    df2['PAR_ENT_REU_PLACE'] = df2['PAR_ENT_REU_PLACE'].astype('float')
    df2['PAR_ENT_REUSSITE_GAGNE'] = df2['PAR_ENT_REUSSITE_GAGNE'].astype('float')
    df2['PAR_ENT_VICTOIRE'] = df2['PAR_ENT_VICTOIRE'].astype('float')
    df2['PAR_ENT_VICTOIRE'] = df2['PAR_ENT_VICTOIRE'].astype('int')
    df2['PAR_GAIN'] = df2['PAR_GAIN'].astype('float')
    df2['PAR_GAIN'] = df2['PAR_GAIN'].astype('int')
    df2['pAR_JOC_ECART_GAGNANT'] = df2['pAR_JOC_ECART_GAGNANT'].astype('float')
    df2['pAR_JOC_ECART_GAGNANT'] = df2['pAR_JOC_ECART_GAGNANT'].astype('int')
    df2['PAR_JOC_ECART_PLACE'] = df2['PAR_JOC_ECART_PLACE'].astype('float')
    df2['PAR_JOC_ECART_PLACE'] = df2['PAR_JOC_ECART_PLACE'].astype('int')
    df2['PAR_JOC_NB_COURSE'] = df2['PAR_JOC_NB_COURSE'].astype('float')
    df2['PAR_JOC_NB_COURSE'] = df2['PAR_JOC_NB_COURSE'].astype('int')
    df2['PAR_JOC_PLACE_3P'] = df2['PAR_JOC_PLACE_3P'].astype('float')
    df2['PAR_JOC_PLACE_3P'] = df2['PAR_JOC_PLACE_3P'].astype('int')
    df2['pAR_JOC_RAPPORT_GAGNANT_M'] = df2['pAR_JOC_RAPPORT_GAGNANT_M'].astype('float')
    df2['pAR_JOC_RAPPORT_GAGNANT_M'] = df2['pAR_JOC_RAPPORT_GAGNANT_M'].astype('int')
    df2['PAR_JOC_REU_PLACE'] = df2['PAR_JOC_REU_PLACE'].astype('float')
    df2['pAR_JOC_REUSSITE_GAGNE'] = df2['pAR_JOC_REUSSITE_GAGNE'].astype('float')
    df2['pAR_JOC_VICTOIRE'] = df2['pAR_JOC_VICTOIRE'].astype('float')
    df2['pAR_JOC_VICTOIRE'] = df2['pAR_JOC_VICTOIRE'].astype('int')
    df2['PAR_NP'] = df2['PAR_NP'].astype('float')
    df2['PAR_NP'] = df2['PAR_NP'].astype('int')
    df2['PAR_NUM'] = df2['PAR_NUM'].astype('float')
    df2['PAR_NUM'] = df2['PAR_NUM'].astype('int')
    df2['PAR_PLACE'] = df2['PAR_PLACE'].astype('float')
    df2['PAR_PLACE'] = df2['PAR_PLACE'].astype('int')
    df2['PAR_PLACE_Q'] = df2['PAR_PLACE_Q'].astype('float')
    df2['PAR_PLACE_Q'] = df2['PAR_PLACE_Q'].astype('int')
    df2['PAR_REUSSITE_3P'] = df2['PAR_REUSSITE_3P'].astype('float')
    df2['PAR_REUSSITE_GAGNE'] = df2['PAR_REUSSITE_GAGNE'].astype('float')
    df2['PAR_REUSSITE_QUINTE'] = df2['PAR_REUSSITE_QUINTE'].astype('float')
    df2['PAR_RUESSITE_PLACE'] = df2['PAR_RUESSITE_PLACE'].astype('float')
    df2['autostart'] = df2['autostart'].astype('float')
    df2['autostart'] = df2['autostart'].astype('int')
    df2['cendre'] = df2['cendre'].astype('float')
    df2['cendre'] = df2['cendre'].astype('int')
    df2['grande_piste'] = df2['grande_piste'].astype('float')
    df2['grande_piste'] = df2['grande_piste'].astype('int')
    df2['Point'] = df2['Point'].astype('float')
    df2['Point'] = df2['Point'].astype('int')
    df2['Nb_partant'] = df2['Nb_partant'].astype('float')
    df2['Nb_partant'] = df2['Nb_partant'].astype('int')
    df2['PAR_PROPRIO'] = df2['PAR_PROPRIO'].astype('float')
    df2['PAR_PROPRIO'] = df2['PAR_PROPRIO'].astype('int')
    df2['NOM_JOC'] = df2['NOM_JOC'].astype('float')
    df2['NOM_JOC'] = df2['NOM_JOC'].astype('int')
    df2['NOM_ENTR'] = df2['NOM_ENTR'].astype('float')
    df2['NOM_ENTR'] = df2['NOM_ENTR'].astype('int')
    df2['POIDS'] = df2['POIDS'].astype('float')
    df2['POIDS'] = df2['POIDS'].astype('int')
    df2['CORDE'] = df2['CORDE'].astype('float')
    df2['CORDE'] = df2['CORDE'].astype('int')
    df2['CHEVAL'] = df2['CHEVAL'].astype('float')
    df2['CHEVAL'] = df2['CHEVAL'].astype('int')
    df2['MUSIC_CHEVAL'] = df2['MUSIC_CHEVAL'].astype('str')
    df2['MUSIC_ENT'] = df2['MUSIC_ENT'].astype('str')
    df2['MUSIC_JOC'] = df2['MUSIC_JOC'].astype('str')
    df2['PAR_VALEUR'] = df2['PAR_VALEUR'].astype('float')
    df2['PAR_VALEUR'] = df2['PAR_VALEUR'].astype('int')
    df2['PAR_ENT_ECART_PLACE'] = df2['PAR_ENT_ECART_PLACE'].astype('float')
    df2['PAR_ENT_ECART_PLACE'] = df2['PAR_ENT_ECART_PLACE'].astype('int')
    df2['PAR_VICTOIRE'] = df2['PAR_VICTOIRE'].astype('float')
    df2['PAR_VICTOIRE'] = df2['PAR_VICTOIRE'].astype('int')
    df2['PAR_VICTOIRE_Q'] = df2['PAR_VICTOIRE_Q'].astype('float')
    df2['PAR_VICTOIRE_Q'] = df2['PAR_VICTOIRE_Q'].astype('int')
    df2['PAR_ENT_NB_COURSE'] = df2['PAR_ENT_NB_COURSE'].astype('float')
    df2['PAR_ENT_NB_COURSE'] = df2['PAR_ENT_NB_COURSE'].astype('int')
    df2['FIN_ligne'] = df2['FIN_ligne'].astype('str')

    df2['p2018'] = df2['p2018'].astype('float')
    df2['p2018'] = df2['p2018'].astype('int')
    df2['PAR_REUSSITE_QUINTE'] = df2['PAR_REUSSITE_QUINTE'].astype('float')
    df2['PAR_VICTOIRE_Q'] = df2['PAR_VICTOIRE_Q'].astype('float')
    df2['PAR_VICTOIRE_Q'] = df2['PAR_VICTOIRE_Q'].astype('int')

    return df2


def courbe_de_roc(model, test_x, test_y):
    from sklearn.metrics import roc_curve, auc

    probas = model.predict_proba(test_x)
    # probas est une matrice de deux colonnes avec la proabilités d'appartenance à chaque classe

    fpr, tpr, thresholds = roc_curve(test_y, probas[:, 1])
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %f" % roc_auc)

    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")


def get_names():
    xnames = ['ALLURE', 'CO_DISTANCE',
              'CO_PRIX', 'HIPPO',
              'IDCOURSE', 'IDPARTCIPANT',
              'PAR_AGE', 'PAR_ARRIVE',
              'PAR_CARRIERE',
              'PAR_CARRIERE_Q',
              'PAR_CLASSE_AGE',
              'PAR_COTEDER',
              'PAR_ENT_ECART_GAGNANT',
              'PAR_ENT_RAPPORT_GAGNANT_M',
              'PAR_ENT_REU_PLACE',
              'PAR_ENT_REUSSITE_GAGNE',
              'PAR_ENT_VICTOIRE',
              'PAR_GAIN',
              'pAR_JOC_ECART_GAGNANT',
              'PAR_JOC_ECART_PLACE',
              'PAR_JOC_NB_COURSE',
              'PAR_JOC_PLACE_3P',
              'pAR_JOC_RAPPORT_GAGNANT_M',
              'PAR_JOC_REU_PLACE',
              'pAR_JOC_REUSSITE_GAGNE',
              'pAR_JOC_VICTOIRE',
              'PAR_NP',
              'PAR_NUM',
              'PAR_PLACE',
              'PAR_PLACE_Q',
              'PAR_REUSSITE_3P',
              'PAR_REUSSITE_GAGNE',
              'PAR_REUSSITE_QUINTE',
              'PAR_RUESSITE_PLACE',
              'autostart',
              'cendre',
              'grande_piste',
              'Point',
              'Nb_partant',
              'PAR_PROPRIO',
              'NOM_JOC',
              'NOM_ENTR',
              'POIDS',
              'CORDE', 'CHEVAL',
              'MUSIC_CHEVAL',
              'MUSIC_ENT',
              'MUSIC_JOC',
              'PAR_VALEUR',
              'PAR_ENT_ECART_PLACE',
              'PAR_VICTOIRE',
              'PAR_VICTOIRE_Q',
              'PAR_ENT_NB_COURSE',
              'POINTS_MUSIC',
    'MOIS_COURSE',
              'DATE_COURSE',
              'PAR_SEXE',
              'FIN_ligne']

    return xnames


def get_names2():
    xnames2 = [
        'CO_PRIX', 'HIPPO',
        'IDCOURSE', 'IDPARTCIPANT',
        'PAR_AGE',
        'PAR_CARRIERE',
        'PAR_CARRIERE_Q',
        'PAR_CLASSE_AGE',
        'PAR_COTEDER',
        'PAR_ENT_ECART_GAGNANT',
        'PAR_ENT_RAPPORT_GAGNANT_M',
        'PAR_ENT_REU_PLACE',
        'PAR_ENT_REUSSITE_GAGNE',
        'PAR_ENT_VICTOIRE',
        'PAR_GAIN',
        'pAR_JOC_ECART_GAGNANT',
        'PAR_JOC_ECART_PLACE',
        'PAR_JOC_NB_COURSE',
        'PAR_JOC_PLACE_3P',
        'pAR_JOC_RAPPORT_GAGNANT_M',
        'PAR_JOC_REU_PLACE',
        'pAR_JOC_REUSSITE_GAGNE',
        'pAR_JOC_VICTOIRE',
        'PAR_NP',
        'PAR_NUM',
        'PAR_PLACE',
        'PAR_PLACE_Q',
        'PAR_REUSSITE_3P',
        'PAR_REUSSITE_GAGNE',
        'PAR_REUSSITE_QUINTE',
        'PAR_RUESSITE_PLACE',
        'autostart',
        'cendre',
        'grande_piste',
        'Point',
        'Nb_partant',
        'PAR_PROPRIO',
        'NOM_JOC',
        'NOM_ENTR',
        'POIDS',
        'CORDE', 'CHEVAL',
        'MUSIC_CHEVAL',
        'MUSIC_ENT',
        'MUSIC_JOC',
        'PAR_VALEUR',
        'PAR_ENT_ECART_PLACE',
        'PAR_VICTOIRE',
        'PAR_VICTOIRE_Q',
        'PAR_ENT_NB_COURSE',
        'POINTS_MUSIC',
        'FIN_ligne',
        'SELECTION2',
        'p2018']


    return xnames2

def drop_test(test_copy):
    test_x = test_copy.copy()

    test_x =my_drop(test_x, 'PAR_CLASSE_AGE')

    test_x =my_drop(test_x, 'PAR_ENT_ECART_GAGNANT')
    test_x =my_drop(test_x, 'PAR_ENT_RAPPORT_GAGNANT_M')
    test_x =my_drop(test_x, 'PAR_ENT_REU_PLACE')
    test_x =my_drop(test_x, 'PAR_ENT_REUSSITE_GAGNE')
    test_x =my_drop(test_x, 'PAR_ENT_VICTOIRE')
    test_x =my_drop(test_x, 'PAR_GAIN')
    test_x =my_drop(test_x, 'pAR_JOC_ECART_GAGNANT')
    test_x =my_drop(test_x, 'PAR_JOC_ECART_PLACE')
    test_x =my_drop(test_x, 'PAR_JOC_NB_COURSE')
    test_x =my_drop(test_x, 'PAR_JOC_PLACE_3P')
    test_x =my_drop(test_x, 'PAR_REUSSITE_GAGNE')
    test_x =my_drop(test_x, 'PAR_REUSSITE_QUINTE')
    test_x =my_drop(test_x, 'PAR_RUESSITE_PLACE')
    test_x =my_drop(test_x, 'CO_PRIX')
    test_x =my_drop(test_x, 'PAR_CARRIERE')
    test_x =my_drop(test_x, 'PAR_CARRIERE_Q')
    test_x =my_drop(test_x, 'pAR_JOC_RAPPORT_GAGNANT_M')
    test_x =my_drop(test_x, 'pAR_JOC_REUSSITE_GAGNE')
    test_x =my_drop(test_x, 'pAR_JOC_VICTOIRE')
    test_x =my_drop(test_x, 'PAR_PLACE')
    test_x =my_drop(test_x, 'PAR_PLACE_Q')
    test_x =my_drop(test_x, 'PAR_CLASSE_AGE')
    test_x =my_drop(test_x, 'pAR_JOC_VICTOIRE')
    test_x =my_drop(test_x, 'PAR_REUSSITE_3P')
    test_x =my_drop(test_x, 'PAR_JOC_REU_PLACE')
    test_x =my_drop(test_x, 'PAR_NUM')
    test_x =my_drop(test_x, 'PAR_COTEDER')
    test_x =my_drop(test_x, 'CORDE')
    test_x =my_drop(test_x, 'musique')
    test_x =my_drop(test_x, 'CHEVAL')
    test_x =my_drop(test_x, 'Nb_partant')

    test_x =my_drop(test_x, 'autostart')
    test_x =my_drop(test_x, 'grande_piste')
    test_x =my_drop(test_x, 'cendre')

    test_x =my_drop(test_x, 'PAR_PROPRIO')
    test_x =my_drop(test_x, 'NOM_JOC')
    test_x =my_drop(test_x, 'NOM_ENTR')

    test_x =my_drop(test_x, 'HIPPO')
    test_x = my_drop(test_x, 'POINTS_MUSIC')
    test_x =my_drop(test_x, 'PAR_AGE')
    test_x =my_drop(test_x, 'POIDS')
    test_x =my_drop(test_x, 'CO_DISTANCE')
    test_x =my_drop(test_x, 'CO_PRIX')
    test_x =my_drop(test_x, 'PAR_GAIN_NORMA')
    test_x =my_drop(test_x, 'CHEVAL2')
    test_x =my_drop(test_x, 'PAR_REUSSITE_3P2')
    test_x =my_drop(test_x, 'PAR_REUSSITE_QUINTE2')
    test_x =my_drop(test_x, 'PAR_CLASSE_AGE2')
    test_x =my_drop(test_x, 'PAR_COTEDER2')
    test_x =my_drop(test_x, 'Point')

    test_x =my_drop(test_x, 'MUSIC_CHEVAL')
    test_x =my_drop(test_x, 'MUSIC_ENT')
    test_x =my_drop(test_x, 'MUSIC_JOC')
    test_x =my_drop(test_x, 'PAR_VALEUR')

    test_x =my_drop(test_x, 'MY_REUSSITE_CHEVAL')
    test_x =my_drop(test_x, 'MY_REUSSITE_JOC')

    test_x =my_drop(test_x, 'MY_REUSSITE_ENT')
    test_x =my_drop(test_x, 'MY_ECART_JOC')
    test_x =my_drop(test_x, 'CLA_AGE_PRIX')
    test_x =my_drop(test_x, 'MY_auto_start')
    test_x = my_drop(test_x, 'PAR_SEXE')


    test_x =my_drop(test_x, 'PAR_ENT_ECART_PLACE')
    test_x =my_drop(test_x, 'PAR_VICTOIRE')
    test_x =my_drop(test_x, 'PAR_VICTOIRE_Q')
    test_x =my_drop(test_x, 'CHEVAL_QUINTE')
    test_x =my_drop(test_x, 'PAR_ENT_NB_COURSE')
    test_x =my_drop(test_x, 'p2018')
    test_x =my_drop(test_x, 'MOIS_COURSE')




    test_x['v0'] = 0.0
    test_x['v1'] = 0.0
    test_x['v2'] = 0.0
    test_x['v3'] = 0.0
    test_x['sel'] = 0




    test_x.head(10)

    return test_x



def reduc_var(model, X_app, y_app, test_X, test_y):
    from sklearn.feature_selection import RFE
    selecteur=RFE(estimator=model)

    #lancer la recherhe
    sol=selecteur.fit(X_app, y_app)

    #Nombre de var. selectionnées
    print(sol.n_features_)

    #liste des variables sélectionnées
    print(sol.support_)

    #ordre de suppression
    print(sol.ranking_)

    X_new_app=X_app[:,sol.support_]
    print(X_new_app.shape)

    #construction de la base test aux memes variables
    X_new_test = test_X[:, sol.support_]

    print(X_new_test.shape)

    #prediction du modele reduit sur l'ech. test
    y_pred_sel=model.predict(X_new_test)

    #evaluation
    print(metrics.accuracy_score(test_y, y_pred_sel))





def model_avec_selectFrom(model, X_train, y_train,X_test,y_test,model2):




    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # Fit model using each importance as a threshold
    thresholds = np.sort(model.feature_importances_)
    print(thresholds)


    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)


        # train model
        selection_model = model2
        selection_model.fit(select_X_train, y_train,verbose=True)
        # eval model
        select_X_test = selection.transform(X_test)

        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))



def recherche_best_feature(model, X, y):
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    # Build a classification task using 3 informative features

    # Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct
# classifications
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2),
              scoring='accuracy', verbose=50)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def smote_valid(cart,X, y):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import model_selection as ms
    from sklearn import datasets, metrics, tree

    from imblearn import over_sampling as os
    from imblearn import pipeline as pl

    print(__doc__)

    RANDOM_STATE = 42

    scorer = metrics.make_scorer(metrics.cohen_kappa_score)


    smote = os.SMOTE()

    pipeline = pl.make_pipeline(smote, cart)

    param_range = range(5, 6)
    train_scores, test_scores = ms.validation_curve(
        pipeline, X, y, param_name="smote__k_neighbors", param_range=param_range,
        cv=3, scoring=scorer, n_jobs=1)

    print(train_scores)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(param_range, test_scores_mean, label='SMOTE')
    ax.fill_between(param_range, test_scores_mean + test_scores_std,
                    test_scores_mean - test_scores_std, alpha=0.2)
    idx_max = np.argmax(test_scores_mean)
    plt.scatter(param_range[idx_max], test_scores_mean[idx_max],
                label=r'Cohen Kappa: ${0:.2f}\pm{1:.2f}$'.format(
                    test_scores_mean[idx_max], test_scores_std[idx_max]))

    plt.title("Validation Curve with SMOTE-CART")
    plt.xlabel("k_neighbors")
    plt.ylabel("Cohen's kappa")

    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    plt.xlim([1, 10])
    plt.ylim([0.4, 0.8])

    plt.legend(loc="best")
    plt.show()


def get_train_test(df, y_col, x_cols, ratio):
    """
    This method transforms a dataframe into a train and test set, for this you need to specify:
    1. the ratio train : test (usually 0.7)
    2. the column with the Y_values
    """
    mask = np.random.rand(len(df)) # & lt;ratio

    df_train = df[mask]
    df_test = df[~mask]

    Y_train = df_train[y_col].values
    Y_test = df_test[y_col].values

    X_train = df_train[x_cols].values
    X_test = df_test[x_cols].values
    return df_train, df_test, X_train, Y_train, X_test, Y_test


def return_dict_classi():
    dict_classifiers = {
        "Logistic Regression": LogisticRegression(),
        "Nearest Neighbors": KNeighborsClassifier(),
        "Linear SVM": SVC(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=1000),
        "Decision Tree": tree.DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=1000),
        "Neural Net": MLPClassifier(alpha=1),
        "Naive Bayes": GaussianNB(),
        "xgbclassifier": XGBClassifier()
        # "AdaBoost": AdaBoostClassifier(),
        # "QDA": QuadraticDiscriminantAnalysis(),
        # "Gaussian Process": GaussianProcessClassifier()
    }

    return dict_classifiers



def cross_search(model,tuned_parameters, train_x,train_y, test_x, test_y,allure ):
    from sklearn.metrics import make_scorer


    scoring = ['accuracy']

   # scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}


    cv = StratifiedKFold(n_splits=50,shuffle=True, random_state=7)


    print()

    #        clf = GridSearchCV(model, tuned_parameters, cv=cv, verbose=50, n_jobs=4, refit=True,
    #                   scoring='%s_macro' % score)
    clf = GridSearchCV(model, tuned_parameters, cv=cv, verbose=50, n_jobs=4, refit=True,
                       scoring='accuracy')


    start_time = timer(None)
    clf.fit(train_x, train_y)




    timer(start_time)


    print(clf)
    #print(clf.cv_results_)

    print("Meilleurs parametres trouvés :")
    print("****************************")
    print(clf.best_params_)
    print("****************************")
    print("Meilleur estimateur")
    print(clf.best_estimator_)
    print(clf.best_score_)

    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    # plot

    #pyplot.errorbar(learning_rate, means, yerr=stds)
    #pyplot.title("XGBoost learning_rate vs Log Loss")
    #pyplot.xlabel('learning_rate')
    #pyplot.ylabel('Log Loss')
    #pyplot.savefig('learning_rate.png')




    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_y, clf.predict(test_x)
    print(classification_report(y_true, y_pred))
    print()
    predictions = [round(value) for value in y_pred]
    PROBA = clf.predict_proba(test_x)
    # evaluate predictions
    accuracy = accuracy_score(test_y, predictions)
    print(">>>>>>>>>>  Accuracy: %.3f%%" % (accuracy * 100.0))

    save_mymodel(clf.best_estimator_, allure)



def cross_search_random(model,tuned_parameters, train_x,train_y, test_x, test_y,allure,n_iter_search ):


    scores = ['precision','recall']


    cv = StratifiedKFold(n_splits=10)
    print('cross_search_random')

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = RandomizedSearchCV(model, tuned_parameters,  n_iter=n_iter_search, cv=cv, verbose=50, n_jobs=4, refit=True,
                           scoring='%s_macro' % score)


        start_time = timer(None)
        clf.fit(train_x, train_y)
        timer(start_time)


        print(clf)

        print("Meilleurs parametres trouvés :")
        print("****************************")
        #print(clf.best_params_)
        print("****************************")
        print("Meilleur estimateur")
        print(clf.best_estimator_)

        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_y, clf.predict(test_x)
        print(classification_report(y_true, y_pred))
        print()
        predictions = [round(value) for value in y_pred]
        PROBA = clf.predict_proba(test_x)
        # evaluate predictions
        accuracy = accuracy_score(test_y, predictions)
        print(">>>>>>>>>>  Accuracy: %.3f%%" % (accuracy * 100.0))

        save_mymodel(clf.best_estimator_, allure)









def batch_classify(dict_classifiers, X_train, Y_train, X_test, Y_test, no_classifiers=5, verbose=True):
    """
    This method, takes as input the X, Y matrices of the Train and Test set.
    And fits them on all of the Classifiers specified in the dict_classifier.
    The trained models, and accuracies are saved in a dictionary. The reason to use a dictionary
    is because it is very easy to save the whole dictionary with the pickle module.

    Usually, the SVM, Random Forest and Gradient Boosting Classifier take quiet some time to train.
    So it is best to train them on a smaller dataset first and
    decide whether you want to comment them out or not based on the test accuracy score.
    """

    dict_models = {}
    for classifier_name, classifier in list(dict_classifiers.items())[:no_classifiers]:
        t_start = time.clock()
        classifier.fit(X_train, Y_train)
        t_end = time.clock()

        t_diff = t_end - t_start
        train_score = classifier.score(X_train, Y_train)
        test_score = classifier.score(X_test, Y_test)

        dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s  {sc} {sc2}".format(c=classifier_name, f=t_diff, sc=train_score , sc2=test_score))
    return dict_models


def display_dict_models(dict_models, sort_by='test_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['test_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)), columns=['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    print(df_.sort_values(by=sort_by, ascending=False))


