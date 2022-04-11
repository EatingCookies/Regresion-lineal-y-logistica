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







