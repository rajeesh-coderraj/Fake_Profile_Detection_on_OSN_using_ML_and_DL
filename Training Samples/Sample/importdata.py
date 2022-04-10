...
# Load dataset

#Load csv file from drive
file = open('F:\Main Project\Sample\dataset\iris.csv')
type(file)

#import pandas,csv and Read csv file from drive
import csv
import pandas as pd
#Read csv file from drive
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset =pd.read_csv(file)
csvreader = csv.reader(file)

#print the dataset dimensions
import loadlib
import test
# shape / row details
print(dataset.shape)

# descriptions
print(dataset.describe())

# head /column details
print(dataset.head(20))

# class distribution
print(dataset.groupby('class').size())

# box and whisker plots

import matplotlib.pyplot as plt
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

...
# histograms
dataset.hist()
plt.show()
import pandas
# scatter plot matrix

# pandas.plotting.scatter_matrix(dataset)

plt.show()

import numpy as np
from sklearn.model_selection import train_test_split
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1) 
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# Spot Check Algorithms

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())) 