import  sys
sys.path.insert(0, "C:/projets_python/diabolo")

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import etude_variable.MyLog as log
import etude_variable.analyse as ana


def analyseMusic(mus):
    return 0


def lecture_data(Fichier, xnames, xindex_col, allure=1, mode_debug=0, avec_index=True):

    log.traceLogInfo("Lecture data %s" % Fichier)
    if avec_index == True:
        df = pd.read_csv(Fichier, index_col=xindex_col,
                         sep=';',
                         names=xnames, skipinitialspace=True,
                         encoding='Latin-1')
    else:
        df = pd.read_csv(Fichier,
                         index_col=None,
                         sep=';',
                         names=xnames, skipinitialspace=True,
                         encoding='Latin-1')

    df = df.groupby("ALLURE")
    df = df.get_group(allure)





    return df



def suppression_colonne(df2,allure):

    # df=my_drop(df, "PAR_AGE")
    print('Suppression colonnne ', allure)

    if  allure==0:

        df2.drop(["FIN_ligne"], axis=1, inplace=True)

        df2.drop(["PAR_NP"], axis=1, inplace=True)
        df2.drop(["cendre"], axis=1, inplace=True)
        df2.drop(["MUSIC_CHEVAL"], axis=1, inplace=True)
        df2.drop(["MUSIC_ENT"], axis=1, inplace=True)
        df2.drop(["MUSIC_JOC"], axis=1, inplace=True)
        df2.drop(["grande_piste"], axis=1, inplace=True)
        df2.drop(["PAR_VALEUR"], axis=1, inplace=True)
        df2 = my_drop(df2, 'PAR_REUSSITE_GAGNE')
        df2 = my_drop(df2, 'PAR_RUESSITE_PLACE')
        df2 = my_drop(df2, 'pAR_JOC_REUSSITE_GAGNE')
        df2 = my_drop(df2, 'PAR_JOC_REU_PLACE')
        df2 = my_drop(df2, 'PAR_JOC_ECART_PLACE')
        df2 = my_drop(df2, 'PAR_ENT_REUSSITE_GAGNE')
        df2 = my_drop(df2, 'PAR_ENT_REU_PLACE')

        df2 = my_drop(df2, 'PAR_CARRIERE_Q')
        df2 = my_drop(df2, 'PAR_REUSSITE_QUINTE')

        df2 = my_drop(df2, 'CHEVAL')
        df2 = my_drop(df2, 'NOM_JOC')
        df2 = my_drop(df2, 'NOM_ENTR')
        df2 = my_drop(df2, 'ALLURE')


        df2 = my_drop(df2, 'PAR_PLACE_Q')
        df2 = my_drop(df2, 'autostart')

        df2 = my_drop(df2, 'PAR_PLACE')

        df2 = my_drop(df2, 'PAR_JOC_PLACE_3P')

        df2 = my_drop(df2, 'PAR_ARRIVE')
        df2 = my_drop(df2, 'HIPPO')





    if allure == 1:
        df2 = my_drop(df2, "POIDS")
        df2 = my_drop(df2, "CORDE")


    if allure == 3:
        df2 = my_drop(df2, "POIDS")
        df2 = my_drop(df2, "CORDE")



    if allure == 4:
        df2 = my_drop(df2, "CORDE")


    if allure == 5:
        df2 = my_drop(df2, "CORDE")


    return df2




def my_drop(df, col):
    if col in df:
        df.drop([col], axis=1, inplace=True)
    return df


# Creating bins for the win column
def assign_selection(W):
    if W < 4:
        return 1
    else:
        return 0


def premiere_ligne(m, dist):
    if dist < m:
        return 1
    else:
        return 0


# Creating bins for the win column
def assign_selection2(W):
    if W == 1.0:
        return 1
    if W == 0.0:
        return 0

def MY_REUSSITE_CHEVAL(par_reussite_gagne,par_reussite_place):
    if par_reussite_place == 0.0:
        return 0

    return par_reussite_gagne/par_reussite_place


def info_dataFrame(df):

   j=0
   print(df.head(5))





def explo_variable(dataset, svariable):
    mediane = np.median(dataset)

    print("La mediane              " + svariable + "    : ", round(mediane, 2))

    maxx = np.max(dataset)
    print("Le max              " + svariable + "    : ", round(maxx, 2))

    minn = np.min(dataset)
    print("Le min              " + svariable + "    : ", round(minn, 2))

    moy = np.mean(dataset)
    print("La moyenne               " + svariable + "   : ", round(moy, 2))

    variance = np.var(dataset)
    print("La variance est          " + svariable + "   : ", round(variance, 2))

    ecartType = np.std(dataset)
    print("Le ecart type            " + svariable + "   : ", round(ecartType, 2))

    print(dataset.describe())

    print("")


def calul_data_allure(allure_etudier, mode_debug=0, avec_index=True):
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
              'FIN_ligne']

    index_col = ['IDPARTCIPANT', 'IDCOURSE']
    # Lecture des données sans index
    print(" Lecture DATA")
    df = lecture_data('d:\data_diabolo.csv', xnames, xindex_col=index_col, allure=allure_etudier, avec_index=False)
    print("*********\n\n")

    print(df.head(10))


    # Les suppressions ont deja été faites dans lecture_data

    #df = df[df.PAR_GAIN > 0]
    #df = df[df.PAR_GAIN < 100000]
    df = df[df.Point > 0]
    #df = df[df.PAR_COTEDER > 0]
    df = df[df.PAR_ARRIVE > 0]  # On garde la ligne qui possede information arrivée

    df = df.groupby("ALLURE")
    df = df.get_group(allure_etudier)
    df = my_drop(df, 'ALLURE')

    if allure_etudier == 2 or allure_etudier == 4 or allure_etudier == 5:
        df = df[df.POIDS > 0]
        df = df[df.POIDS < 80]

        # pour une autre allure le poids est supprimer
        # si galop, haie, steeple on a un poids superieur a 0

    #  on a lu les données avant l 'ajout de la colonne SELECTIOB
    df['SELECTION2'] = df['PAR_ARRIVE'].apply(assign_selection)

    df['SELECTION2'] = df['PAR_ARRIVE'].apply(assign_selection)

    df.drop(["PAR_ARRIVE"], axis=1, inplace=True)

    Lib_features_df = df.columns



    # retourn un NUMPY ARRAY
    #xdata = df.values

    df_gagnant=df
    print("head")
    df.head(5)
    df.info()





    #df_gagnant = pd.DataFrame(data=xdata, columns=Lib_features_df)

    # Pour une conversion en INTEGER DE SELECTION
    df_gagnant['SELECTION'] = df_gagnant['SELECTION2'].apply(assign_selection2)
    df_gagnant.drop(["SELECTION2"], axis=1, inplace=True)

    df_gagnant = df_gagnant.set_index(index_col)
    #df_gagnant.head(5)

    df_gagnant_len = len(df_gagnant.columns) - 1
    Lib_features = df_gagnant.columns[:df_gagnant_len]

    i = 0

    feature_columns = Lib_features  ##<<<<<<<<<<<<<<<<
    response_column = ['SELECTION']  ##<<<<<<<<<<<<<<<<
    log.traceLogdebug("Features                   : %s " % Lib_features, " <<<<************")

    #ratio=ana.afficheDesequilibreClasse(df_gagnant)
    #print("RATIO ",ratio)
    print("head")
    df_gagnant.head(5)
    df_gagnant.info()




    # ---------------------------------------------------
    return df_gagnant, feature_columns, response_column, 0


def save_mymodel(clf, allure):
    log.traceLogInfo("Sauvegarde modele ...")
    joblib.dump(clf, 'diabolo' + str(allure) + '.pkl')


def load_mymodel(allure):
    log.traceLogInfo("Restauarion modele")
    clf = joblib.load('diabolo' + str(allure) + '.pkl')
    return clf




def encodage(df_gagnant):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    #df_gagnant[['HIPPO']] = le.fit_transform(df_gagnant[['HIPPO']])
    #df_gagnant[['PAR_PROPRIO']] = le.fit_transform(df_gagnant[['PAR_PROPRIO']])
    ##f_gagnant[['NOM_JOC']] = le.fit_transform(df_gagnant[['NOM_JOC']])
    #df_gagnant[['NOM_ENTR']] = le.fit_transform(df_gagnant[['NOM_ENTR']])
    #df_gagnant[['CHEVAL']] = le.fit_transform(df_gagnant[['CHEVAL']])
    #df_gagnant[['PAR_NUM']] = le.fit_transform(df_gagnant[['PAR_NUM']])
   # df_gagnant[['MUSIC_CHEVAL']] = le.fit_transform(df_gagnant[['MUSIC_CHEVAL']])



    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import quantile_transform





    scaler = RobustScaler()
    scaler = RobustScaler( with_scaling=False, quantile_range=(25.0, 75.0), copy=False)

    critere_scale = ['PAR_CARRIERE', 'PAR_AGE','PAR_PLACE_Q','PAR_CARRIERE', 'PAR_ENT_ECART_GAGNANT','PAR_ENT_RAPPORT_GAGNANT_M',
                     'PAR_ENT_REU_PLACE', 'PAR_ENT_REUSSITE_GAGNE','pAR_JOC_REUSSITE_GAGNE','pAR_JOC_RAPPORT_GAGNANT_M','PAR_JOC_PLACE_3P', 'PAR_ENT_VICTOIRE',
                     'pAR_JOC_ECART_GAGNANT', 'PAR_JOC_ECART_PLACE','PAR_JOC_NB_COURSE',  'PAR_JOC_REU_PLACE','pAR_JOC_VICTOIRE',
                     'PAR_NUM', 'PAR_PLACE','PAR_REUSSITE_3P','PAR_REUSSITE_GAGNE', 'PAR_RUESSITE_PLACE']

    #df_gagnant[critere_scale] = quantile_transform(df_gagnant[critere_scale], n_quantiles=4)

    #df_gagnant[critere_scale] = qt.fit_transform(df_gagnant[critere_scale])
    #df_gagnant[critere_scale] = quantile_transform(df_gagnant[critere_scale], n_quantiles=5, random_state=42,subsample=300000)




    #df_gagnant[critere_scale] = scaler.fit_transform(df_gagnant[critere_scale])

    return df_gagnant,critere_scale
