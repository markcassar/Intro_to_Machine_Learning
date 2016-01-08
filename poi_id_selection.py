#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFpr, chi2, f_classif
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from ggplot import *
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


#from feature_format import featureFormat, targetFeatureSplit
#from tester import test_classifier, dump_classifier_and_data

# Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# create a dataframe from 'data_dict'
my_df = pd.DataFrame.from_dict(data_dict, orient='index')
# force data types to be numeric where appropriate
my_df= my_df.convert_objects(convert_numeric=True)

# Remove outliers
# remove salary outlier associated with 'TOTAL' since this value is not
# associated with a person 
#my_df = my_df[ (my_df['salary'] < 2e7) ] 
#my_df = my_df[ (my_df['salary'] < 2e7) | my_df['salary'].isnull() ] 

# remove 'THE TRAVEL AGENCY IN THE PARK' as it is mostly NaN's and doesn't
# seem to be associated with a person
#my_df = my_df.drop( ['TOTAL'] )
my_df = my_df.drop( ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL'] )

# leave in other 'outliers' as they are valid 
# data points and seem to contain information that may help classify people

# Create new feature(s)
# based on correlations between features and 'poi' 
# (see Enron_data_exploration.ipynb for details) build new features to try
# to amplify the correlation to 'poi' 

# someone may send or receive a lot of email messages so would most likely
# send or receive a lot from a poi so may be better to look at the proportion
# of email messages from or to a poi
my_df.loc[:,'from_ratio'] = my_df.from_this_person_to_poi / my_df.from_messages
my_df.loc[:,'to_ratio'] = my_df.from_poi_to_this_person / my_df.to_messages

# based on highest correlations with 'poi' variable build a few that may
# amplify this effect for classification
my_df.loc[:,'stock_salary'] = my_df.exercised_stock_options * my_df.salary
my_df.loc[:,'bonus_salary'] = my_df.bonus * my_df.salary
my_df.loc[:,'bonus_stock'] = my_df.bonus * my_df.exercised_stock_options

# create original and engineered feature lists
# features_list is a list of strings, each of which is a feature name.
# first feature must be "poi".

#features_list = ['poi', 'salary', 'exercised_stock_options', 'bonus',  
#                    'total_stock_value', 'from_ratio', 
#                    'stock_salary', 'bonus_salary', 'bonus_stock',
#                    'from_poi_to_this_person'] 
                    
features_list = ['poi', 'salary', 'to_messages', 'total_payments',  
                    'exercised_stock_options', 'bonus', 'restricted_stock', 
                    'shared_receipt_with_poi', 'total_stock_value', 'expenses', 
                    'from_messages', 'other', 'from_this_person_to_poi', 
                    'deferred_income', 'long_term_incentive', 
                    'from_poi_to_this_person']  
                    
eng_features_list = ['from_ratio', 'to_ratio',
                    'stock_salary', 'bonus_salary', 'bonus_stock'] 
                    
# can join these two lists using +

### Extract features and labels from dataset for local testing
my_df = my_df[ features_list + eng_features_list]
print( my_df.shape)

# fill in NaN values strategy -- test all 3 to gauge impact on accuracy
#my_df = my_df.fillna(0)
#my_df = my_df.fillna( my_df.median() )
my_df = my_df.fillna( my_df.mean() )

my_df_array = np.array( my_df )
old_features_array = my_df_array[ :, 1: ]
values_array = my_df_array[ :, [0] ].astype(int)
values_array = np.ravel(values_array)

# scale features for kNN and SVM classifiers
#old_features_array_scaled = preprocessing.robust_scale( old_features_array )
old_features_array_scaled = preprocessing.scale( old_features_array )
#scaler = preprocessing.MinMaxScaler()
#old_features_array_scaled = scaler.fit_transform( old_features_array )

rf_precision = {}
rf_recall = {}
nb_precision = {}
nb_recall = {}
svm_precision = {}
svm_recall = {}
knn_precision = {}
knn_recall = {}

# select most important features

# test values of k from 1-20
for i in range(1,21):
#for i in range(1,10):
    print i
    best = i
    
    selector = SelectKBest(f_classif, k=best)
    
    ## comment out one of the 'features_array' lines to get appropriate features
    ## depending on classifier chosen, so for my chosen classifiers:
    #         unscaled -- Random Forest and Naive Bayes 
    #         scaled -- kNN and SVM
    #
    # scaled features
    features_array_scaled = selector.fit_transform(old_features_array_scaled, values_array)
    # unscaled features
    features_array = selector.fit_transform(old_features_array, values_array)
    
    # get selected feature scores and names
    feature_names = [my_df.columns[j+1] for j in selector.get_support(indices=True)]
    select_features = dict( zip(feature_names, selector.scores_) )
    print
    print(select_features)
    print
    
    # split scaled or unscaled data into train and test sets
    # although 'scaled' is added to labels data, it is not scaled, nomenclature
    # is just for symmetry
    features_train_scaled, features_test_scaled, \
        labels_train_scaled, labels_test_scaled = train_test_split( \
                                    features_array_scaled, values_array, \
                                    test_size = 0.2, random_state=16)
    
    features_train, features_test, labels_train, labels_test = train_test_split( \
                                    features_array, values_array, \
                                    test_size = 0.2, random_state=16)
    
    # adjust the shape of the labels for use in sklearn
    labels_train_scaled = np.ravel(labels_train_scaled)
    labels_test_scaled = np.ravel(labels_test_scaled)
    
    labels_train = np.ravel(labels_train)
    labels_test = np.ravel(labels_test)
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    #### CODE SECTION: TEST CLASSIFIERS 
    # parameters for benchmark recall and precision from BENCHMARK TUNING
    # section of code
    
    # Random Forest classifier
    clf_rf = RandomForestClassifier(criterion='gini', min_samples_split=4, \
                                n_estimators=9, random_state=42, \
                                class_weight={False: 1, True: 12}
    )
    clf_rf.fit(features_train, labels_train)
    
    rf_y_pred = clf_rf.predict(features_test)
    
    print('Random Forest')
    #print('accuracy score: ', clf_rf.score(features_test, labels_test) )
    print('precision score: ', precision_score(labels_test, rf_y_pred) )
    print('recall score: ', recall_score(labels_test, rf_y_pred) )
    #print('f1 score: ', f1_score( labels_test, rf_y_pred) )
    #print( confusion_matrix( labels_test, rf_y_pred ) )
    rf_precision[i] = precision_score(labels_test, rf_y_pred)
    rf_recall[i] = recall_score(labels_test, rf_y_pred)
    #print rf_precision
    
    print
    print
    
    
    # naive bayes classifier
    clf_nb = GaussianNB()    
    clf_nb.fit(features_train, labels_train)
    
    nb_y_pred = clf_nb.predict(features_test)
    
    print('Naive Bayes')
    #print('accuracy score: ', clf_nb.score(features_test, labels_test) )
    print('precision score: ', precision_score(labels_test, nb_y_pred) )
    print('recall score: ', recall_score(labels_test, nb_y_pred) )
    #print('f1 score: ', f1_score( labels_test, nb_y_pred) )
    #print( confusion_matrix( labels_test, nb_y_pred ) )
    nb_precision[i] = precision_score(labels_test, nb_y_pred)
    nb_recall[i] = recall_score(labels_test, nb_y_pred)
    #print nb_precision
    
    print 
    print 
    
    # support vector machine classifier
    # SVC parameters chosen via gridsearch (see code below from Udacity reviewer)
    clf_svc = SVC(C=1e-05, kernel='linear', gamma='auto', tol=0.1, \
                    class_weight={False:1, True:12}, random_state=42)
    clf_svc = SVC() 
    
    clf_svc.fit(features_train_scaled, labels_train_scaled)
    
    svc_y_pred = clf_svc.predict(features_test_scaled)
    
    print('Support Vector Machine')
    #print('accuracy score: ', clf_svc.score(features_test_scaled, labels_test_scaled) )
    print('precision score: ', precision_score(labels_test_scaled, svc_y_pred) )
    print('recall score: ', recall_score(labels_test_scaled, svc_y_pred) )
    #print('f1 score: ', f1_score( labels_test_scaled, svc_y_pred) )
    #print( confusion_matrix( labels_test_scaled, svc_y_pred ) )
    svm_precision[i] = precision_score(labels_test_scaled, svc_y_pred)
    svm_recall[i] = recall_score(labels_test_scaled, svc_y_pred)
    #print svm_precision
    
    print
    print 
    
    # k nearest neighbor classifier
    # 
    clf_knn = KNeighborsClassifier(n_neighbors=1, leaf_size=20)
    clf_knn.fit(features_train_scaled, labels_train_scaled)
    
    knn_y_pred = clf_knn.predict(features_test_scaled)
    
    print('Nearest Neighbours')
    #print('accuracy score: ', clf_knn.score(features_test_scaled, labels_test_scaled) )
    print('precision score: ', precision_score(labels_test_scaled, knn_y_pred) )
    print('recall score: ', recall_score(labels_test_scaled, knn_y_pred) )
    #print('f1 score: ', f1_score( labels_test_scaled, knn_y_pred) )
    #print( confusion_matrix( labels_test_scaled, knn_y_pred ) )
    knn_precision[i] = precision_score(labels_test_scaled, knn_y_pred)
    knn_recall[i] = recall_score(labels_test_scaled, knn_y_pred)
    #print knn_precision
    
    # best k value = 6
    # features in this set
    # {'bonus_stock': 6.8538878046074689, 
    #  'from_ratio': 7.734638749578659, 
    #  'exercised_stock_options': 9.3986743955690208, 
    #  'bonus_salary': 11.437118489798213, 
    #  'total_stock_value': 0.36823451036833349, 
    #  'stock_salary': 29.13338963978693}

    
#### PLOT DATA
# keep getting python ggplot error
# use data in precision and recall dictionaries to plot vs k
# in R (see feature_selection.R file for code)

     
######### CODE SECTION: BENCHMARK TUNING
# tune algorithm parameters on full original + engineered feature set
# for benchmark precision and recall values 
#
### Random Forest classifier tuning
#
#pipe = Pipeline(steps=[('clf', RandomForestClassifier())])
#cv = cross_validation.StratifiedShuffleSplit(labels_train,n_iter = 50, \
#                                                random_state = 42)
#
#parameters = {'clf__n_estimators':[6,7,8,9,10], 'clf__criterion':('gini','entropy'), 
#                'clf__min_samples_split':[2,4,6], 
#                'clf__class_weight': [{True: 12, False: 1},
#                                    {True: 10, False: 1},
#                                    {True: 8, False: 1},
#                                    {True: 15, False: 1},
#                                    {True: 4, False: 1},
#                                    'balanced', None]  
#            }
#
#rf_gridsearch = GridSearchCV(pipe, param_grid=parameters, cv=cv, scoring='recall')
#rf_gridsearch.fit(features_train, labels_train)
#
### pick a winner
#best_clf = rf_gridsearch.best_estimator_
#print best_clf
#
# result
#Pipeline(steps=[('clf', RandomForestClassifier(bootstrap=True, class_weight={False: 1, True: 12},
#            criterion='gini', max_depth=None, max_features='auto',
#            max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=4,
#            min_weight_fraction_leaf=0.0, n_estimators=9, n_jobs=1,
#            oob_score=False, random_state=None, verbose=0,
#            warm_start=False))])




### Gaussian Naive Bayes classifier tuning
#
# no parameters to tune



### SVM classifier tuning
#
# code to fix issue with SVM producing 0's for precision and recall
# from Udacity Reviewer comments
#
#clf_params= {
#                       'clf__C': [1e-5, 1e-2, 1e-1, 1, 10, 1e2, 1e5],
#                       'clf__gamma': [0.0],
#                       'clf__kernel': ['linear', 'poly', 'rbf'],
#                       'clf__tol': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  
#                       'clf__class_weight': [{True: 12, False: 1},
#                                               {True: 10, False: 1},
#                                               {True: 8, False: 1},
#                                               {True: 15, False: 1},
#                                               {True: 4, False: 1},
#                                               'balanced', None]        
#                                             # 'auto' deprecated use 'balanced'
#                      }
#
#pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()), ('clf', SVC())])
#cv = cross_validation.StratifiedShuffleSplit(labels_train,n_iter = 50, \
#                                                random_state = 42)
#a_grid_search = GridSearchCV(pipe, param_grid = clf_params,cv = cv, \
#                                scoring = 'recall')
#a_grid_search.fit(features_train,labels_train)
#
## pick a winner
#best_clf = a_grid_search.best_estimator_
#print best_clf

# result
#Pipeline(steps=[('minmaxer', MinMaxScaler(copy=True, feature_range=(0, 1))), ('clf', SVC(C=1e-05, cache_size=200, class_weight={False: 1, True: 12}, coef0=0.0,
#  decision_function_shape=None, degree=3, gamma=0.0, kernel='linear',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.1, verbose=False))])




### kNN classifier tuning
#
#pipe = Pipeline(steps=[('minmaxer', MinMaxScaler()), ('clf', KNeighborsClassifier())])
#cv = cross_validation.StratifiedShuffleSplit(labels_train,n_iter = 50, \
#                                                random_state = 42)
#
#parameters = {'clf__n_neighbors':[1,2,3,4,5,6,7,8], 'clf__weights':('uniform','distance'), 
#                'clf__leaf_size':[20,30,40], 'clf__p':[1,2] }
#
#knn_gridsearch = GridSearchCV(pipe, param_grid=parameters, cv=cv, scoring='recall')
#knn_gridsearch.fit(features_train, labels_train)
#
### pick a winner
#best_clf = knn_gridsearch.best_estimator_
#print best_clf
#
# result
#Pipeline(steps=[('minmaxer', MinMaxScaler(copy=True, feature_range=(0, 1))), ('clf', KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski',
#           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
#           weights='uniform'))])



###########################



### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
