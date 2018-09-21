import  sys

sys.path.insert(0, "C:/projets_python/diabolo")

import etude_variable.MyLog as log


import pandas as pd

from skopt import BayesSearchCV
import numpy as np
from collections import Counter

from scipy.stats import randint

from etude_variable import lecture_data as ld
from skopt.space import Real, Integer

from sklearn.metrics import  classification_report


from matplotlib import pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from matplotlib import pyplot

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import learning_curve

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from sklearn import  metrics
from sklearn.metrics import  roc_auc_score

from time import time
from operator import itemgetter
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.utils import class_weight

from sklearn.feature_selection import SelectFromModel

import xgboost as xgb

import time


class Timer:
  def __init__(self):
    self.start = time.time()

  def restart(self):
    self.start = time.time()

  def get_time(self):
    end = time.time()
    m, s = divmod(end - self.start, 60)
    h, m = divmod(m, 60)
    time_str = "%02d:%02d:%02d" % (h, m, s)
    return time_str





#Définir une méthode pour imprimer la matrice de confusion et les indicateurs de performance
def Print_confusion_matrix(cm, auc, heading):
    print('\n', heading)
    print(cm)
    true_negative  = cm[0,0]
    true_positive  = cm[1,1]
    false_negative = cm[1,0]
    false_positive = cm[0,1]
    total = true_negative + true_positive + false_negative + false_positive
    accuracy = (true_positive + true_negative)/total
    precision = (true_positive)/(true_positive + false_positive)
    recall = (true_positive)/(true_positive + false_negative)
    misclassification_rate = (false_positive + false_negative)/total
    F1 = (2*true_positive)/(2*true_positive + false_positive + false_negative)
    print('accuracy.................%7.4f' % accuracy)
    print('precision................%7.4f' % precision)
    print('recall...................%7.4f' % recall)
    print('F1.......................%7.4f' % F1)
    print('auc......................%7.4f' % auc)




#Définir une fonction d'utilité pour signaler les meilleurs scores
def Report_scores(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")






def my_randomize_estimate3(estimateur, fit_direct, train_x, train_y,nb_iteration, verbose, test_x, test_y,allure, my_scoring="roc_auc",cv=3, nb_job=1,ratio=0):
    r=0


def  my_randomize_estimateur(estimateur, fit_direct, train_x, train_y,nb_iteration, verbose, test_x, test_y,allure, my_scoring="roc_auc",cv=3, nb_job=1,ratio=0):
    my_timer = Timer()

    if allure>0 :

          train_accuracy = []
          test_accuracy = []

          #'min_child_weight': (11, 20),
          #'gamma': (0.0001, 1.0),
          #'colsample_bytree': (0.5, 0.7)

          if fit_direct==False:
                    # Estimamteur de recherche



                  ju

          else:
              # ESTIMATEUR SANS RECHERCHE
              print("Lecture des données estimateur ",allure)

              # estimator1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              #   colsample_bytree=0.6, gamma=0.1, learning_rate=0.07,
              #   max_delta_step=0, max_depth=9, max_features='0.7',
              #   min_child_weight=9, n_estimators=190, n_jobs=3,
              #    objective='binary:logistic', random_state=10,
              #   reg_alpha=0.6763110907709939, reg_lambda=1, scale_pos_weight=1, seed=482,
              #   silent=True, subsample=0.35615749454318585)

              estimator1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                         colsample_bytree=0.6, gamma=0.1, learning_rate=0.09,
                                         max_delta_step=0, max_depth=11, max_features='auto',
                                         min_child_weight=13, n_estimators=1000, n_jobs=3,
                                         nthread=None, objective='binary:logistic', random_state=10,
                                         reg_alpha=0.03, reg_lambda=1, scale_pos_weight=3.58, seed=27,
                                         silent=True, subsample=0.6)

              # retour avec les parmetres de l'estimateur

              return estimator1



    if allure == 2:

        xgbclf = XGBClassifier(base_score=0.5,
                               booster='gbtree',

                               max_depth=5,
                               min_child_weight=12,
                               n_jobs=3,
                               nthread=None,
                               objective='binary:logistic',
                               seed=804,
                               silent=True)

        xb_search = {'n_estimators': (400, 6000),

                     'random_state': (9, 45),
                     'learning_rate':(0.6,0.7,0.8,0.9)
                     }



        estimator2 = BayesSearchCV(xgbclf, n_iter=nb_iteration,
                                   search_spaces=[(xb_search,10)]
                                   ,
                                   scoring=my_scoring, verbose=verbose, n_jobs=nb_job, cv=cv)

        # callback handler
        def on_step2(optim_result):

            score = estimator2.best_score_
            print("GALOP : best score: %s" % score)
            print("GALOP : best score: %s" % estimator2.best_params_)

            if score >= 0.98:
                print(' TROT : Le score 0.98 est atteint !')
                return True

        estimator2.fit(train_x, train_y["SELECTION"].ravel(), callback=on_step2)


        return estimator2






    if allure == 3:

        xgbclf = XGBClassifier(learning_rate=0.07, booster='gbtree', objective='binary:logistic')

        estimator3 = BayesSearchCV(xgbclf, n_iter=nb_iteration,
                                   search_spaces=[
                                       {'max_depth': (8, 9, 10)},
                                       {'random_state': (38, 42, 43)},
                                       {'min_child_weight': (10, 12, 14)},
                                       {'n_estimators': (150, 200, 250)}

                                   ],
                                   scoring=my_scoring, verbose=verbose, n_jobs=nb_job, cv=cv)

        # callback handler
        def on_step3(optim_result):

            score = estimator3.best_score_
            print("3 : best score: %s" % score)
            if score >= 0.98:
                print(' 3 : Le score 0.98 est atteint !')
                return True

        estimator3.fit(train_x, train_y["SELECTION"].ravel(), callback=on_step3)

        return estimator3


    if allure == 4:

        xgbclf = XGBClassifier(learning_rate=0.07, booster='gbtree', objective='binary:logistic')

        estimator4 = BayesSearchCV(xgbclf, n_iter=nb_iteration,
                                   search_spaces=[
                                       {'max_depth': (9, 10, 11)},
                                       {'random_state': (38, 42, 43)},
                                       {'min_child_weight': (10, 12, 14)},
                                       {'n_estimators': (80, 100, 110)}
                                   ],
                                   scoring=my_scoring, verbose=verbose, n_jobs=nb_job, cv=cv)

        # callback handler
        def on_step4(optim_result):

            score = estimator4.best_score_
            print("4 : best score: %s" % score)
            if score >= 0.98:
                print(' 4 : Le score 0.98 est atteint !')
                return True

        estimator4.fit(train_x, train_y["SELECTION"].ravel(), callback=on_step4)


        return estimator4


    if allure == 5:

        xgbclf = XGBClassifier(learning_rate=0.07, booster='gbtree', objective='binary:logistic')

        estimator5 = BayesSearchCV(xgbclf, n_iter=nb_iteration,
                                   search_spaces=[
                                       {'max_depth': (9, 10, 11)},
                                       {'random_state': (38, 42, 43)},
                                       {'min_child_weight': (10, 12, 14)},


                                       {'n_estimators': (80, 100, 110)}
                                   ],
                                   scoring=my_scoring, verbose=verbose, n_jobs=nb_job, cv=cv)

        # callback handler
        def on_step5(optim_result):

            score = estimator5.best_score_
            print("5 : best score: %s" % score)
            if score >= 0.98:
                print(' 5 : Le score 0.98 est atteint !')
                return True

        estimator5.fit(train_x, train_y["SELECTION"].ravel(), callback=on_step5)


        return estimator5







def my_randomize_estimateur2(estimateur, ratio, fit_direct, train_x, train_y,nb_iteration, verbose, test_x, test_y,allure, my_scoring="roc_auc",cv=3, nb_job=1):

    # 'gamma': [i / 10.0 for i in range(0, 5)],
    # 'colsample_bytree': [i / 10.0 for i in range(6, 10)],
    # 'subsample': [i / 10.0 for i in range(6, 10)],
    # 'colsample_bylevel': [0.5, 0.6, 0.7],
    # param_grid = dict(learning_rate=learning_rate,                          n_estimators=n_estimators)
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.pipeline import Pipeline


    if allure == 1:
                estimator =  estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bytree=0.9, gamma=0.3, learning_rate=0.07,
                max_delta_step=4, max_depth=10, max_features='sqrt',
                min_child_weight=13, n_estimators=1300, n_jobs=1,
                nthread=None, objective='binary:logistic', random_state=10,
                reg_alpha=0.2, reg_lambda=1, scale_pos_weight=1, seed=400,
                silent=True, subsample=0.9)








    if allure == 3:
                estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bytree=0.9, gamma=0.3, learning_rate=0.07,
                max_delta_step=4, max_depth=10, max_features='sqrt',
                min_child_weight=13, n_estimators=1300, n_jobs=1,
                nthread=None, objective='binary:logistic', random_state=10,
                reg_alpha=0.2, reg_lambda=1, scale_pos_weight=1, seed=400,
                silent=True, subsample=0.9)

    if allure == 2:
                estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.4,
                colsample_bytree=0.9, gamma=0.3, learning_rate=0.07,
                max_delta_step=4, max_depth=10, max_features='sqrt',
                min_child_weight=13, n_estimators=1300, n_jobs=1,
                nthread=None, objective='binary:logistic', random_state=10,
                reg_alpha=0.2, reg_lambda=1.0, scale_pos_weight=1, seed=400,
                silent=True, subsample=0.9)

    if allure == 4:
                 estimator =  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bytree=0.6, gamma=0.1, learning_rate=0.07,
                max_delta_step=0, max_depth=9, max_features='sqrt',
                min_child_weight=14, n_estimators=100, n_jobs=1,
                nthread=None, objective='binary:logistic', random_state=10,
                reg_alpha=0.03, reg_lambda=1, scale_pos_weight=1, seed=27,
                silent=True, subsample=0.9)

    if allure == 5:
                estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                       colsample_bytree=0.6, gamma=0.1, learning_rate=0.07,
                       max_delta_step=0, max_depth=9, max_features='sqrt',
                       min_child_weight=12, n_estimators=100, n_jobs=1,
                       nthread=None, objective='binary:logistic', random_state=10,
                       reg_alpha=0.03, reg_lambda=1, scale_pos_weight=1, seed=27,
                       silent=True, subsample=0.9)



    return estimator







def get_class_weights(y):
    import collections
    counter = collections.Counter(y)



    majority = max(counter.values())
    return  {cls: float(majority/count) for cls, count in counter.items()}










def CorrelationDesVariable(df_gagnant):
    fig2, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df_gagnant.corr(), annot=True, fmt=".2f", linewidths=0.5, ax=ax)
    plt.show()

    # --------------------------SEPARATION DES DONNES------------------------------




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


def afficheDesequilibreClasse(df):
    target_count = df['SELECTION'].value_counts()

    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])
    print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

    target_count.plot(kind='bar', title='Count (target)')
    plt.show()

    normal_trans_perc = sum(df['SELECTION'] == 0) / (sum(df['SELECTION'] == 0) + sum(df['SELECTION'] == 1))
    fraud_trans_perc = 1 - normal_trans_perc
    print('Total number of records : {} '.format(len(df)))
    print('Nombre de participations avec SELECTION=0 : {}'.format(sum(df['SELECTION'] == 0)))
    print('Nombre de participations avec SELECTION=1  : {}'.format(sum(df['SELECTION'] == 1)))
    print('Pourcentage 0: {:.4f}%,  pourcentage 1 : {:.4f}%'.format(normal_trans_perc * 100,
                                                                                                      fraud_trans_perc * 100))


    return 0








def Plot_learning_curve(estimator,
                        title, X, y,
                        ylim=None,
                        cv=None,
                        n_jobs=1,
                        train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X, y,
                                                            cv=cv,
                                                            n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
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
    plt.show()




def plot_importance(feature_columns, model):
    importances = pd.DataFrame({'feature': feature_columns, 'importance': np.round(model.feature_importances_, 3)})
    importances = importances.sort_values('importance', ascending=False).set_index('feature')
    print("")
    print( importances)
    importances.plot.bar()



#Définir une méthode pour tracer l'importance du prédicteur
def Plot_predictor_importance(best_model, feature_columns):

    feature_importance = best_model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    y_pos  = np.arange(sorted_idx.shape[0]) + .7
    fig, ax = plt.subplots( figsize=(10, 10))



    ax.barh(y_pos,
            feature_importance[sorted_idx],
            align='center',
            color='green',
            ecolor='black',
            height=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_columns)
    ax.invert_yaxis()



    ax.set_xlabel('Relative Importance')
    ax.set_title('Predictor Importance')
    plt.show()


def my_split(taille_entrainement, df,feature_columns,response_column ,random_state=42,mode_debug=0):
    train_x, test_x, train_y, test_y = split_dataset(df,train_percentage=0.85,
                                                         feature_headers=feature_columns,
                                                     target_header= response_column,

                                                         random_state=42,
                                                     mode_debug=mode_debug
                                                     )

    #
    # # Stats of training data
    # print('---------Training data statistics-----------')
    # normal_trans_perc = sum(train_y['SELECTION'] == 0) / (sum(train_y['SELECTION'] == 0) + sum(train_y['SELECTION'] == 1))
    # fraud_trans_perc = 1 - normal_trans_perc
    # print('Total number of records : {} '.format(len(train_y)))
    # print('Total 0 : {}'.format(sum(train_y['SELECTION'] == 0)))
    # print('Total 1 : {}'.format(sum(train_y['SELECTION'] == 1)))
    # print('Percent 0 : {:.4f}%,  1 is : {:.4f}%'.format(normal_trans_perc * 100,
    #                                                                                                   fraud_trans_perc * 100))
    #
    # # Stats of testing data
    # print('---------Testing data statistics-----------')
    # normal_trans_perc = sum(test_y['SELECTION'] == 0) / (sum(test_y['SELECTION'] == 0) + sum(test_y['SELECTION'] == 1))
    # fraud_trans_perc = 1 - normal_trans_perc
    # print('Total number of records : {} '.format(len(test_y)))
    # print('Total 0 : {}'.format(sum(test_y['SELECTION'] == 0)))
    # print('Total 1: {}'.format(sum(test_y['SELECTION'] == 1)))
    # print('Percent 0 : {:.4f}%,  1 : {:.4f}%'.format(normal_trans_perc * 100,
    #                                                                                                   fraud_trans_perc * 100))

    return train_x, test_x, train_y, test_y



def my_split_XBOOST(taille_entrainement, df,feature_columns,
                                                  response_column ,
                                                  random_state=42,mode_debug=0):

    #class xgboost.DMatrix(data, label=None, missing=None, weight=None,
    # silent=False, feature_names=None, feature_types=None, nthread=None)
    trainSize = 0.80
    train_x, test_x, train_y, test_y = split_dataset(df,
                                                         taille_entrainement,
                                                          feature_columns,
                                                          response_column,
                                                         random_state=random_state,
                                                     mode_debug=mode_debug
                                                     )


    return train_x, test_x, train_y, test_y




def run_randomsearch(X, y, clf, para_dist, cv=5, n_iter_search=20):
    """Run a random search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_dist -- [dict] list, distributions of parameters
                  to sample
    cv -- fold of cross-validation, default 5
    n_iter_search -- number of random parameter sets to try,
                     default 20.

    Returns
    -------
    top_params -- [dict] from report()
    """
    random_search = RandomizedSearchCV(clf,
                        param_distributions=param_dist,
                        n_iter=n_iter_search)

    start = time.time()
    model=random_search.fit(X, y)

    print(("\nRandomizedSearchCV took {:.2f} seconds "
           "for {:d} candidates parameter "
           "settings.").format((time.time() - start),
                               n_iter_search))

    top_params = report2(random_search.cv_results_, 3)


    return  top_params,model


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


def report2(grid_scores, n_top=15):
    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters


def smot(train_x,train_y,feature_columns):
    from imblearn.ensemble import BalanceCascade
    from sklearn.ensemble import RandomForestClassifier


    #sm = RandomOverSampler(ratio='majority')
    #from imblearn.ensemble import BalanceCascade



    sm=BalanceCascade (random_state =42,classifier=RandomForestClassifier() )



    print('Détail du nombre par CLASSE  Y {}'.format(Counter(train_y)))
    X_res, y_res = sm.fit_sample(train_x, train_y)

    my_list = map(lambda x: x[0],y_res)
    train_y= pd.Series(my_list)
    print(' Détail du nombre par CLASSE Y  {}'.format(Counter(train_y)))


    # reconstitution DATAFRAME
    train_x = pd.DataFrame(X_res, columns=feature_columns)


    return train_x,train_y





def smot2(train_x,train_y,feature_columns):

    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import ADASYN
    from sklearn.svm import  SVC
    from imblearn.under_sampling import  CondensedNearestNeighbour






    print('\nOriginal dataset shape {}'.format(Counter(train_y)))

    sm = SMOTEENN(ratio='minority',n_jobs=3,random_state=42,n_neighbors=50,smote=SMOTE() )
    #sm = ADASYN(ratio='minority', n_jobs=3,random_state=42,n_neighbors=100)



    #sm = SMOTE(ratio='minority', n_jobs=3, random_state=42,m_neighbors=200)

    #sm = CondensedNearestNeighbour(ratio='majority', random_state=42)



    log.traceLogInfo("\nFIT DE SMOT2 ...equilibrage")
    X_res, y_res = sm.fit_sample(train_x, train_y)

    print('\nResampled dataset shape {}'.format(Counter(y_res)))
    # reconstitution DATAFRAME
    train_x = pd.DataFrame(X_res, columns=feature_columns)
    train_y = pd.Series(y_res)


    return train_x,train_y







def metrique_classe(y_pred,y_true,xclass):
    from imblearn.metrics import specificity_score
    from imblearn.metrics import sensitivity_score


    from imblearn.metrics import geometric_mean_score



    # La sensibilité est le rapport où est le nombre de vrais positifs et le nombre de faux négatifs.
    # La sensibilité quantifie la capacité à éviter les faux négatifs.tp


    # estimator issu de quelques FIT


    log.traceLogInfo("Classe ", xclass)
    if xclass==0:
        log.traceLogInfo("Classe 0")
    if xclass == 1:
        log.traceLogInfo("Classe 1")

    log.traceLogInfo("Sensibilité  du re-equilibrage des données sur le TEST")
    #log.traceLogInfo("Binary ",sensitivity_score(y_true, y_pred, average='binary', pos_label=xclass))

    log.traceLogInfo("La spécificité est intuitivement la capacité du classificateur à trouver tous les échantillons positifs")

    log.traceLogInfo("Binary ")
    log.traceLogInfo(specificity_score(y_true, y_pred, labels=None, pos_label=xclass, average='binary', sample_weight=None))




    print("\nCalculer la moyenne géométrique")
    print(geometric_mean_score(y_true, y_pred,labels=None, pos_label=xclass))

    print("\n Calculer  sensitivity score")
    print("La sensibilité est le rapport où est le nombre de vrais positifs et le nombre de faux négatifs.")
    print("La sensibilité quantifie la capacité à éviter les faux négatifs.")

    print(sensitivity_score(y_true, y_pred, labels=None, pos_label=xclass,average='binary'))






def my_fit2018(estimateur,          nb_iter, my_nb_splits, my_test_size,          my_random_state,test_x,test_y, train_x, train_y, featurecolums, allure,mode_debug=0):
    h=0

    # ICI ON FAIT UNE RECHERCHE DES BON PARAMTRE
    xgbclf = XGBClassifier(base_score=0.5,
                           booster='gbtree',
                           learning_rate=0.07,
                           max_depth=6,  # ok
                           min_child_weight=17,  # ok
                           n_jobs=3,
                           nthread=None,
                           objective='binary:logistic',
                           scale_pos_weight=0,
                           n_estimators=1000,

                           seed=382,
                           silent=True)

    print("\nPositionnement de la recherche ....\n")
    xb_search = {

        'random_state': randint(9, 45),
        'gamma': [i / 10.0 for i in range(0, 9)],

        'subsample': [i / 10.0 for i in range(2, 9)],
        'colsample_bytree': [i / 10.0 for i in range(2, 9)]

    }
    clf = RandomizedSearchCV(xgbclf, param_distributions=xb_search, n_iter=nb_iter, scoring="roc_auc", error_score=0, verbose=3, n_jobs=3)




def my_fit(estimateur,          nb_iter, my_nb_splits, my_test_size,   ratio,       my_random_state,test_x,test_y, train_x, train_y, featurecolums, allure,mode_debug=0):

    my_timer = Timer()
    fit_direct = True



    test_y = test_y['SELECTION'].ravel()
    train_y = train_y['SELECTION'].ravel()

    # Correction du desequiilibre
    #train_x, train_y = smot2(train_x=train_x, train_y=train_y, feature_columns=featurecolums)
    #test_x, test_y = smot2(train_x=test_x, train_y=test_y, feature_columns=featurecolums)

    sample_weight = train_y.shape[0] / (2 * np.bincount(train_y))
    print("class_weight = ", sample_weight)
    sample_weight=[0.63,2.28]





    if fit_direct == True  :
        log.traceLogInfo("\nTEST DU FIT ......DIRECT")
        kfold = StratifiedKFold(n_splits=5)
        log.traceLogInfo("\nfold ......DIRECT")
        model2 = my_randomize_estimateur2(estimateur, ratio=ratio,fit_direct=fit_direct, train_x=train_x, train_y=train_y, test_x=test_x,
                                            test_y=test_y,
                                            allure=allure, nb_iteration=nb_iter, verbose=0, my_scoring="roc_auc", cv=kfold, nb_job=2)
        model=model2



        eval_set = [(train_x, train_y), (test_x, test_y)]
        log.traceLogInfo("FIT ...")



        model.fit(train_x, train_y, eval_metric=["error", "auc"], eval_set=eval_set, verbose=False, early_stopping_rounds=20,sample_weight=sample_weight)
        log.traceLogInfo("predic")

        y_pred = model.predict(test_x)

        #metrique_classe(y_pred=y_pred,y_true=test_y,xclass=0)
        #metrique_classe(y_pred=y_pred, y_true=test_y, xclass=1)
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(test_y, predictions)
        print("************************************************************  Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Classification report \n")

        # eval model

        #model_avec_selectFrom(model, train_x, train_y,test_x,test_y,model2)



        print(classification_report(test_y, predictions))






        # retrieve performance metrics
        results = model.evals_result()
        epochs = len(results['validation_0']['auc'])
        x_axis = range(0, epochs)

        # plot log loss
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0']['error'], label='Train')
        ax.plot(x_axis, results['validation_1']['error'], label='Test')
        ax.legend()
        pyplot.ylabel('auc')
        pyplot.title('XGBoost auc')
        pyplot.show()

        # plot classification error
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0']['auc'], label='Train')
        ax.plot(x_axis, results['validation_1']['auc'], label='Test')
        ax.legend()
        pyplot.ylabel('Classification Error')
        pyplot.title('XGBoost Classification Error')
        pyplot.show()

        #Plot_predictor_importance(best_model=model, feature_columns=featurecolums)

        plot_importance(feature_columns=featurecolums,model=model)




    else:
        # estimator trouvé lors des recherches
        # RECHERCHE DE PARAMETRES
        estimator = my_randomize_estimateur(estimateur, fit_direct=fit_direct, train_x=train_x, train_y=train_y, test_x=test_x,
                                            test_y=test_y,
                                            allure=allure, nb_iteration=nb_iter, verbose=3, my_scoring="roc_auc", cv=10, nb_job=3)
        model = estimator
        #model = estimator.best_estimator_

    evaluation(mybest_model=model, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, mode_debug=1)

    if fit_direct == False:
        print("Report score \n")
        # Report_scores(estimator.cv_results_, n_top=nb_iter)
        # Plot_learning_curve(estimator, '', train_x, train_y["SELECTION"].ravel(),cv=cv_, n_jobs=4)
        # -----------------------------------------------------------------------
        # BEST MODEL IS
        best_model = model  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        # -----------------------------------------------------------------------
        print('\nbest_model                                  : ', best_model)
        # print('\nbest_model Nombre de Feature  :  ', best_model.n_features_)
        print('\nbest_model  Classes                      : ', best_model.classes_)
    else:
        best_model = model

    return best_model











def afficheEvalSet(mybest_model,
                    set_train,
                   set_train_cible,
                   set_test,
                   set_test_cible,
                   mode_debug=0,
                   type_eval=1):

    print("\nEVAL SET ....")

    set_test_cible_predicted = mybest_model.predict(set_test)
    set_train_cible_predicted = mybest_model.predict(set_train)

    print('Accuracy is: ', mybest_model.score(set_test, set_test_cible))  # accuracy


    if type_eval==1:
                print("Evaluation TEST-----------------------------------------------------------------------")

                set_test_cible_probabilities = mybest_model.predict_proba(set_test)
                score = set_test_cible_probabilities[:, 1]
                print("***********************************************************  ")
                print("\nScores des classes de test  %s" %score, " **************  ")
                print("***********************************************************  ")




                auc = roc_auc_score(set_test_cible, set_test_cible_predicted)
                print("TEST roc_auc_score :  %s" % auc)







    else:


                set_train_cible_probabilities = mybest_model.predict_proba(set_train)
                auc = roc_auc_score(set_train_cible, set_train_cible_predicted)
                print("TRAINING roc_auc_score :  %s" % auc)







    if mode_debug == 1:
        for x, y in [(set_train, set_train_cible), (set_test, set_test_cible)]:
            yp = mybest_model.predict(x)
            cm = confusion_matrix(y, yp.ravel())
            print(cm)

    import matplotlib.pyplot as plt

    if mode_debug == 1:
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    if type_eval==1:

        ntotal = len(set_test)
        correct = set_test_cible == set_test_cible_predicted
        numCorrect = sum(correct)
        percent = round((100.0 * numCorrect) / ntotal, 3)

        log.traceLogInfo("Classification Correcte des données de test : {0:d}/{1:d}  {2:8.3f}%".format(numCorrect, ntotal, percent))

        prediction_score = 100.0 * mybest_model.score(set_test, set_test_cible)
        log.traceLogInfo('\nScore  TEST  : %8.3f  ************************' % prediction_score)

    else:

        ntotal = len(set_train)
        correct = set_train_cible== set_train_cible_predicted
        numCorrect = sum(correct)
        percent = round((100.0 * numCorrect) / ntotal, 3)

        log.traceLogInfo("\n CLASSIFICATION CORRECTE DES DONNEES DE Train  : {0:d}/{1:d}  {2:8.3f}%".format(numCorrect, ntotal, percent))

        prediction_score = 100.0 * mybest_model.score(set_train, set_train_cible)
        log.traceLogInfo('Score  TRAINING  : %8.3f  ************************' % prediction_score)





def evaluation(mybest_model,
                  train_x, test_x,
                   train_y, test_y,
                    mode_debug=0):


    afficheEvalSet(mybest_model=mybest_model,
                         set_train=train_x,
                         set_train_cible=train_y,
                         set_test=test_x,
                          set_test_cible=test_y,
                   mode_debug=mode_debug, type_eval=1)

    afficheEvalSet(mybest_model=mybest_model,
                         set_train=train_x,
                         set_train_cible=train_y,
                         set_test=test_x,
                          set_test_cible=test_y,
                   mode_debug=mode_debug, type_eval=2)






def my_crossValidation(estimator, df, featureColums, targetColums,scoring,dtrainpred_prob, dtrain_predict, xtrain):

    from sklearn.model_selection import cross_validate


    cv_score = cross_validate(estimator, estimator[featureColums], estimator[targetColums], cv=10, scoring='roc_auc')
    # Print model report:
    print("\nModel Report")
    print("\nAccuracy : %.4g" % metrics.accuracy_score(estimator[targetColums].values, dtrain_predict))
    print( "\nAUC Score (train): %f" % metrics.roc_auc_score (estimator[targetColums], dtrainpred_prob) )
    print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (   np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
    feat_imp = pd.Series(estimator.feature_importances_, featureColums).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')




def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf, param_grid=param_grid,  cv=cv)
    start = time.time()

    grid_search.fit(X, y)


    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time.time() - start,
                len(grid_search.cv_results_)))

    top_params = report2(grid_search.cv_results_, 4)

    return  top_params, model.best_estimator_



def modelfit(alg, dtrain, featurecolums,targetcolums, cv_folds=5, early_stopping_rounds=50):

    print("Model fit")

    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[featurecolums].values, label=dtrain[targetcolums].values)
    print("defd xgtrain")
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, show_stdv=True)

    print("cv result ok")
    alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[featurecolums], dtrain[targetcolums], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[featurecolums])
    dtrain_predprob = alg.predict_proba(dtrain[featurecolums])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[targetcolums].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[targetcolums], dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')



def fit2018(model, nb_iter,test_x, test_y, train_x, train_y, featurecolums):

    # RECHERCHE DE PARAMETRES

    print("FIT sans recherche ......")
    eval_set = [(train_x, train_y), (test_x, test_y)]
    model.fit(train_x, train_y, eval_metric=["error", "auc"], eval_set=eval_set, verbose=True, early_stopping_rounds=15)


    y_pred = model.predict(test_x)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(test_y, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')




    # plot classification error
    fig, ax = pyplot.subplots()
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()


    pyplot.ylabel('Classification Error')
    pyplot.title('XGBoost Classification Error')
    pyplot.show()

    Plot_predictor_importance(best_model=model, feature_columns=featurecolums)




def fit_special(estimateur,    test_x,test_y, train_x, train_y):



    # Use SelectFromModel
    thresholds = np.sort(estimateur.best_estimator_.named_steps["clf"].feature_importances_)

    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(estimateur, threshold=thresh, prefit=True)
        select_X_train = selection.transform(train_x)

        # train model
        selection_model = estimateur
        selection_model.fit(select_X_train, train_y)

        # eval model
        select_X_test = selection.transform(test_x)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(test_y, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))
        print(confusion_matrix(test_y, predictions))
        print(classification_report(test_y, predictions))











