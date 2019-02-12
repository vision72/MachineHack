## Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB as naive
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("/Users/apple/Desktop/Vision/Python_Workspace/dataScience/MachineHack/Titanic/data/titanic/train.csv")
sel_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
label_encoder_x = LabelEncoder()

















# Fill Nan values with average age
df['Age'] = df['Age'].fillna((df['Age'].mean()))

# Fill Nan Categorical values with Unknown age
df['Embarked'] = df['Embarked'].fillna("Unknown")
df['Cabin'] = df['Cabin'].fillna("Unknown")

x = df.loc[:, sel_cols].values
y = df.iloc[:, 1].values

# change Age from ['Male', 'Female'] to [0, 1] through sklearn label encoder
x[:, 2] = label_encoder_x.fit_transform(x[:, 2])
x[:, -1] = label_encoder_x.fit_transform(x[:, -1])
x[:, -2] = label_encoder_x.fit_transform(x[:, -2])
x[:, 6] = label_encoder_x.fit_transform(x[:, 6])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

gbm = naive().fit(x_train, y_train)
predictions = gbm.predict(x_test)

# Print accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
confusion_matrix(y_test, predictions)


df2 = pd.read_csv("/Users/apple/Desktop/Vision/Python_Workspace/dataScience/MachineHack/Titanic/data/titanic/train.csv")

# Fill Nan values with average age
df2['Age'] = df2['Age'].fillna((df2['Age'].mean()))

# Fill Nan Categorical values with Unknown age
df2['Embarked'] = df2['Embarked'].fillna("Unknown")
df2['Cabin'] = df2['Cabin'].fillna("Unknown")

labels = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
x2 = df2.loc[:, labels].values
print(x2[0])
# change Age from ['Male', 'Female'] to [0, 1] through sklearn label encoder
x2[:, 2] = label_encoder_x.fit_transform(x2[:, 2])
x2[:, -1] = label_encoder_x.fit_transform(x2[:, -1])
x2[:, -2] = label_encoder_x.fit_transform(x2[:, -2])
x2[:, 6] = label_encoder_x.fit_transform(x2[:, 6])

predictions = gbm.predict(x2)

submission = pd.DataFrame({ 'PassengerId': df2['PassengerId'],
                            'Survived': predictions })
submission.to_csv("submission.csv", index=False)