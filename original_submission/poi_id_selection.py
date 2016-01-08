#!/usr/bin/python

import sys
import pickle
#sys.path.append("../tools/")
import pandas as pd
import numpy as np

#from feature_format import featureFormat, targetFeatureSplit
#from tester import test_classifier, dump_classifier_and_data


### Task 1: Select which features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'total_payments',  
                    'exercised_stock_options', 'bonus', 'restricted_stock', 
                    'shared_receipt_with_poi', 'total_stock_value', 'expenses', 
                    'from_messages', 'other', 'from_this_person_to_poi', 
                    'deferred_income', 'long_term_incentive', 
                    'from_poi_to_this_person', 'from_ratio', 'to_ratio',
                    'stock_salary', 'bonus_salary', 'bonus_stock'] 

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# create a dataframe from 'data_dict'
my_df = pd.DataFrame.from_dict(data_dict, orient='index')
# force data types to be numeric where appropriate
my_df= my_df.convert_objects(convert_numeric=True)

### Task 2: Remove outliers
# remove salary outlier associated with 'TOTAL' since this value is not
# associated with a person, leave in other 'outliers' as they are valid 
# data points and seem to contain information that may help classify people
#my_df = my_df[ (my_df['salary'] < 2e7) ] 
my_df = my_df[ (my_df['salary'] < 2e7) | my_df['salary'].isnull() ] 

### Task 3: Create new feature(s)
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


### Extract features and labels from dataset for local testing
my_df = my_df[ features_list ]

# fill in NaN values strategy -- test all 3 to gauge impact on accuracy
#my_df = my_df.fillna(0)
#my_df = my_df.fillna( my_df.median() )
my_df = my_df.fillna( my_df.mean() )

my_df_array = np.array( my_df )
old_features_array = my_df_array[ :, 1: ]
values_array = my_df_array[ :, [0] ].astype(int)
values_array = np.ravel(values_array)

# test impact of scaling, use robust_scale due to outlier values
from sklearn import preprocessing
old_features_array_scaled = preprocessing.robust_scale( old_features_array )

# select most important features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFpr, chi2, f_classif

# test values of k from 2-14
selector = SelectKBest(f_classif, k=20)
features_array = selector.fit_transform(old_features_array_scaled, values_array)

# split scaled or unscaled data into train and test sets
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split( \
                                features_array, values_array, \
                                test_size = 0.2, random_state=16)

# adjust the shape of the labels for use in sklearn
labels_train = np.ravel(labels_train)
labels_test = np.ravel(labels_test)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# test out 3 classifiers
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(criterion='entropy', min_samples_split=2, \
                            n_estimators=8, random_state=16)
clf_rf.fit(features_train, labels_train)

rf_y_pred = clf_rf.predict(features_test)


from sklearn.svm import SVC
clf_svc = SVC()
clf_svc.fit(features_train, labels_train)

svc_y_pred = clf_svc.predict(features_test)


from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()    
clf_nb.fit(features_train, labels_train)

nb_y_pred = clf_nb.predict(features_test)

from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(features_train, labels_train)

knn_y_pred = clf_knn.predict(features_test)


# check accuracy, precision, and recall of the 3 classifiers
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


print('Random Forest')
print('accuracy score: ', clf_rf.score(features_test, labels_test) )
print('precision score: ', precision_score(labels_test, rf_y_pred) )
print('recall score: ', recall_score(labels_test, rf_y_pred) )
print('f1 score: ', f1_score( labels_test, rf_y_pred) )
print( confusion_matrix( labels_test, rf_y_pred ) )

print()
print()
print('Support Vector')
print('accuracy score: ', clf_svc.score(features_test, labels_test) )
print('precision score: ', precision_score(labels_test, svc_y_pred) )
print('recall score: ', recall_score(labels_test, svc_y_pred) )
print('f1 score: ', f1_score( labels_test, svc_y_pred) )
print( confusion_matrix( labels_test, svc_y_pred ) )

print('Naive Bayes')
print('accuracy score: ', clf_nb.score(features_test, labels_test) )
print('precision score: ', precision_score(labels_test, nb_y_pred) )
print('recall score: ', recall_score(labels_test, nb_y_pred) )
print('f1 score: ', f1_score( labels_test, nb_y_pred) )
print( confusion_matrix( labels_test, nb_y_pred ) )

print('Nearest Neighbours')
print('accuracy score: ', clf_knn.score(features_test, labels_test) )
print('precision score: ', precision_score(labels_test, knn_y_pred) )
print('recall score: ', recall_score(labels_test, knn_y_pred) )
print('f1 score: ', f1_score( labels_test, knn_y_pred) )
print( confusion_matrix( labels_test, knn_y_pred ) )

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.grid_search import GridSearchCV

parameters = { 'n_neighbors':[1,2,3,4,5,6,7,8], 'weights':('uniform', 'distance'), 
                'leaf_size':[20,30,40], 'p':[1,2] }
clf = GridSearchCV(clf_knn, parameters)
clf.fit(features_train, labels_train)

clf_y_pred = clf.predict(features_test)

print('Tuned KNN')
print('accuracy score: ', clf.score(features_test, labels_test) )
print('precision score: ', precision_score(labels_test, clf_y_pred) )
print('recall score: ', recall_score(labels_test, clf_y_pred) )
print( confusion_matrix( labels_test, clf_y_pred ) )


# get selected feature names

feature_names = [my_df.columns[i+1] for i in selector.get_support(indices=True)]

