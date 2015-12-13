#!/usr/bin/python

import sys
import pickle
#sys.path.append("../tools/")
import pandas as pd
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

""" overall plan

1. remove 'director_fees', 'loan_advances', and 'restricted_stock_deferred' due 
to high % NaN and no data on POI

2. remove 'email_address' as it is non-numeric and everyone in a company would 
have one so should not be a distinguishing feature between POI and non-POI

3. remove 'TOTAL' from dataset as not valid person; keep all other 'outliers' 
as they are valid data points

4. create 'from_ratio', 'to_ratio', 'stock_salary', 'bonus_stock', and 
'bonus_to_salary' as new features

5. test random forest, support vector machine, and naive bayes classifiers

6. select features using selectkbest

7. tune classifier parameters using gridsearchcv 

"""



### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# create a dataframe from 'data_dict'
my_df = pd.DataFrame.from_dict(data_dict, orient='index')
# force data types to be numeric where appropriate
my_df= my_df.convert_objects(convert_numeric=True)

# remove salary outlier associated with 'TOTAL' since this value is not
# associated with a person, leave in other 'outliers' as they are valid 
# data points and seem to contain information that may help classify people

# remove 'THE TRAVEL AGENCY IN THE PARK' as it is mostly NaN's and doesn't
# seem to be associated with a person
my_df = my_df.drop( ['THE TRAVEL AGENCY IN THE PARK', 'TOTAL'] )


# set a couple values to NaN as noted in "Enron_data_exploration.ipynb
#my_df.ix[ 'BHATNAGAR SANJAY', 'restricted_stock'] = np.nan
#my_df.ix[ 'BELFER ROBERT', 'total_stock_value'] = np.nan


# based on outcomes from 'poi_id_selection.py' need to create the following
# new features

# someone may send a lot of email messages so would most likely send  a lot 
# from a poi so may be better to look at the proportion of email messages 
# from  a poi
my_df.loc[:,'from_ratio'] = my_df.from_this_person_to_poi / my_df.from_messages

# based on highest correlations with 'poi' variable build a few that may
# amplify this effect for classification
my_df.loc[:,'stock_salary'] = my_df.exercised_stock_options * my_df.salary
my_df.loc[:,'bonus_salary'] = my_df.bonus * my_df.salary
my_df.loc[:,'bonus_stock'] = my_df.bonus * my_df.exercised_stock_options


### The first feature must be "poi".
# based on work in 'poi_id_selection.py' the following features gave the best
# results.
#features_list = ['poi', 'salary', 'exercised_stock_options', 'bonus',  
#                    'total_stock_value', 'from_ratio', 
#                    'stock_salary', 'bonus_salary', 'bonus_stock']

features_list = ['poi', 'salary', 'exercised_stock_options', 'bonus',  
                    'total_stock_value', 'from_ratio', 
                    'stock_salary', 'bonus_salary', 'bonus_stock',
                    'from_poi_to_this_person'] 

#features_list = ['poi', 'salary', 'to_messages', 'total_payments',  
#                    'exercised_stock_options', 'bonus', 'restricted_stock', 
#                    'shared_receipt_with_poi', 'total_stock_value', 'expenses', 
#                    'from_messages', 'other', 'from_this_person_to_poi', 
#                    'deferred_income', 'long_term_incentive', 
#                    'from_poi_to_this_person', 'from_ratio', 
#                    'stock_salary', 'bonus_salary', 'bonus_stock'] 
                    
### Extract features and labels from dataset for local testing
my_df = my_df[ features_list ]

# fill in NaN values using mean values
my_df = my_df.fillna( 0 )

# sklearn needs numpy arrays as input
# create full array from dataframe
my_df_array = np.array( my_df )
# create array for features
old_features_array = my_df_array[ :, 1: ]
# create array for labels and adjust shape so it is a vector
values_array = my_df_array[ :, [0] ].astype(int)
values_array = np.ravel(values_array)

# scale the data
from sklearn import preprocessing
old_features_array_scaled = preprocessing.robust_scale( old_features_array )


# split scaled or unscaled data into train and test sets
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split( \
                                old_features_array_scaled, values_array, \
                                test_size = 0.3, random_state=16)

# adjust the shape of the labels for use in sklearn
labels_train = np.ravel(labels_train)
labels_test = np.ravel(labels_test)

# create classifier with tuned parameters

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3, leaf_size=30, weights='uniform', p=1)
clf.fit(features_train, labels_train)

knn_y_pred = clf.predict(features_test)


# check accuracy, precision, recall, and f1 score 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

print('Nearest Neighbours')
print('accuracy score: ', clf.score(features_test, labels_test) )
print('precision score: ', precision_score(labels_test, knn_y_pred) )
print('recall score: ', recall_score(labels_test, knn_y_pred) )
print('f1 score: ', f1_score( labels_test, knn_y_pred) )
print( confusion_matrix( labels_test, knn_y_pred ) )


my_df_transpose = my_df.T #.to_dict(my_df)
#my_df_transpose = my_df_transpose.fillna(0)
my_dict = my_df_transpose.to_dict()

### Store to my_dataset for easy export below.
#my_dataset = data_dict
my_dataset = my_dict

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)