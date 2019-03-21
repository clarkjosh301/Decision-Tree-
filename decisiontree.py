import pandas as pd
adult_doc = r'C:\Users\CHESTER\Documents\adult.csv'
adult_df = pd.read_csv(adult_doc)
#cleaning data and making tree more accurate i.e more branches = less accuracy
#dataset from https://www.kaggle.com/uciml/adult-census-income
obj_df = adult_df.select_dtypes(include=['object']).copy()
obj_df['education'].replace(['11th', '10th', '7th-8th', '9th', '12th', '5th-6th', '1st-4th', 'Preschool', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm'], 'no_Bachelors+', inplace=True)
obj_df['education'].replace(['Bachelors', 'Masters', 'Prof-school', 'Doctorate'], 'Bachelors+', inplace=True)
obj_df['marital_status'].replace(['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent'], 'Married', inplace=True)
obj_df['marital_status'].replace(['Separated'], 'Divorced', inplace=True)
#decide to get rid of data with '?', missing data is rather small, decide not to use dummy variable
obj_df = obj_df[obj_df.occupation != '?']
obj_df = obj_df[obj_df.workclass != '?']
obj_df = obj_df[obj_df.native_country != '?']
adult_df = adult_df[adult_df.occupation != '?']
adult_df = adult_df[adult_df.workclass != '?']
adult_df = adult_df[adult_df.native_country != '?']
obj_df = obj_df[obj_df.workclass != 'Self-emp-not-inc']
adult_df = adult_df[adult_df.workclass != 'Self-emp-not-inc']
obj_df['workclass'].replace(['Federal-gov', 'Local-gov', 'State-gov'], 'Non-Private', inplace=True)
obj_df['workclass'].replace(['Self-emp-inc'], 'Private', inplace=True)
obj_df = obj_df[obj_df.workclass != 'Without-pay']
adult_df = adult_df[adult_df.workclass != 'Without-pay']
#turning object variables into integers 
data_cleanup = {'education': {'no_Bachelors+': 0, 'Bachelors+': 1},
		    'sex': {'Female': 0, 'Male': 1}}
obj_df.replace(data_cleanup, inplace=True)
age = adult_df['age']
education = obj_df['education']
sex = obj_df['sex']
cleaned_workclass = {'workclass': {'Private': 0, 'Non-Private': 1}}
obj_df.replace(cleaned_workclass, inplace=True)
workclass = obj_df['workclass']
Marital_status_cleaned = {'marital_status': {'Widowed': 0, 'Married': 1, 'Divorced': 2, 'Never-married': 3}}
obj_df.replace(Marital_status_cleaned, inplace=True)
marital_status = obj_df['marital_status']
#for some reason the number of white people in this data set is around 70%, and Asian + Amer-indian races only account for 2%
#of data points, decide to just use a binary boolean variable to make model more accurate 
obj_df['race'].replace(['Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo'], 'Other', inplace=True)
race_cleaned = {'race': {'Other': 0, 'White': 1}}
obj_df.replace(race_cleaned, inplace=True)
race = obj_df['race']
X = [sex, age, education, workclass,  marital_status, race]
income_cleanup = {'income': {'<=50K': 0, '>50K': 1}}
obj_df.replace(income_cleanup, inplace=True)
income = obj_df['income']
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
X = np.array (X)
yz = obj_df['income']
y = np.array (yz)
X = X.transpose()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0)
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=7)
clf = clf.fit(X_train,y_train)
from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True):
	y_pred=clf.predict(X)
	if show_accuracy:
		print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)),"\n")


measure_performance(X_train, y_train, clf)
#accuracy = .821
