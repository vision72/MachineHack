import re
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mean_squared_error
# from sklearn.cross_validation import train_test_split
## In the next Part.


























train_data = pd.read_csv('/Users/apple/Desktop/Vision/Python_Workspace/dataScience/MachineHack/DocPrice/data/Final_Train.csv')
test_data  = pd.read_csv('/Users/apple/Desktop/Vision/Python_Workspace/dataScience/MachineHack/DocPrice/data/Final_Test.csv')

sel_cols = ['Qualification', 'Experience', 'Rating', 'Profile']
## Could have also used Dropout for Miscellaneous_Info

X_train = train_data[sel_cols] # The Training Variable
X_test = test_data[sel_cols]
y_train = train_data['Fees'] # The Prediction Variable

X_train = X_train.replace(np.NaN, 0)
X_test = X_test.replace(np.NaN, 0)
print(X_train.head())




















# Experience 

print "\n--------**********-----------\n"

s = list()
s2= list()
for e in X_train[['Experience']].iterrows():
	s.append(e)
sit = []
for i in s:
	num = re.findall('\d+', str(i))
	sit.append(num[1])
for e in X_test[['Experience']].iterrows():
	s2.append(e)
sit2 = []
for i in s2:
	num = re.findall('\d+', str(i))
	sit2.append(num[1])

print "\n--------**********-----------\n"

x = X_train.columns[1]
X_train.drop(x, axis = 1, inplace = True)
X_train[x] = sit
# print (X_train.tail())

x2 = X_test.columns[1]
X_test.drop(x2, axis = 1, inplace = True)
X_test[x2] = sit2
# print (X_test.tail())

print "\n-------***********-----------\n"

# Qualification (text)

encoder = LabelEncoder()
X_test.Qualification = X_test.Qualification.replace(',', ' ')
X_train.Qualification = X_train.Qualification.replace(',',' ')
# X_train.columns[0].get_dummies(sep=',')

print "\n-------***********-----------\n"
X_test.Qualification = encoder.fit_transform(X_test.Qualification)
X_train.Qualification = encoder.fit_transform(X_train.Qualification)
# Profile (LabelEncoder)
X_test.Profile = encoder.fit_transform(X_test.Profile)
X_train.Profile = encoder.fit_transform(X_train.Profile)
# print (X_train.tail())
X_train.Rating = encoder.fit_transform(X_train.Rating)
X_test.Rating = encoder.fit_transform(X_test.Rating)

print "\n-------***********-----------\n"

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
my_solution = pd.DataFrame({"Fees": pred})
my_solution.to_csv("my_solution_one.csv")
# # mean_squared_error(y_test)