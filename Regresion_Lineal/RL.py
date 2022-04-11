# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#---------------------------------------------------------

#cargamos los datos de entrada
data = pd.read_excel("./Arboles_Reporte_Medellin.xlsx")
#veamos cuantas dimensiones y registros contiene
data.shape

#---------------------------------------------------------

#son 161 registros con 8 columnas. Veamos los primeros registros
data.head()

#---------------------------------------------------------

# Ahora veamos algunas estadísticas de nuestros datos
data.describe()

#---------------------------------------------------------

# Visualizamos rápidamente las caraterísticas de entrada
data.drop(['NOMBRE COMUN','ESPECIE', 'COBERTURA PIE', 'ID ESTADO', 'ID INTERVENCION'],1).hist()
plt.show()

#---------------------------------------------------------


filtered_data = data[(data['ALTURA TOTAL'] <= 22) & (data['ALTURA DE COPA'] <= 13)]
colores=['orange','blue']
tamanios=[30,60]

f1 = filtered_data['ALTURA TOTAL'].values
f2 = filtered_data['ALTURA DE COPA'].values

asignar=[]
for index, row in filtered_data.iterrows():
    if(row['ALTURA TOTAL']>4.24):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()

#---------------------------------------------------------

# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =filtered_data[["ALTURA TOTAL"]]
X_train = np.array(dataX)
y_train = filtered_data['ALTURA DE COPA'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: ', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: ', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))

#---------------------------------------------------------

