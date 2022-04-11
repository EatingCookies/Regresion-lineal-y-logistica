import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

#------------------------------------------------------

Python
dataframe = pd.read_excel("./Arboles_Reporte_Medellin.xlsx")
dataframe.head()

#------------------------------------------------------

dataframe.describe()

#-----------------------------------------------------

print(dataframe.groupby('ALTURA TOTAL').size())

#-----------------------------------------------------

dataframe.drop(['ALTURA TOTAL'],1).hist()
plt.show()

#----------------------------------------------------

sb.pairplot(dataframe.dropna(), hue='ALTURA TOTAL',size=2,vars=["ALTURA DE COPA", "DAP"],kind='reg')

#----------------------------------------------------

X = np.array(dataframe.drop(['ALTURA TOTAL'],1))
y = np.array(dataframe['ALTURA TOTAL'])
X.shape

model = linear_model.LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)
print(predictions)[0:5]

model.score(X,y)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))

print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))





