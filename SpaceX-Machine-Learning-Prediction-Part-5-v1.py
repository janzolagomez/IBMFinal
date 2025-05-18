#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 

# # **Space X  Falcon 9 First Stage Landing Prediction**
# 

# ## Hands on Lab: Complete the Machine Learning Prediction lab
# 

# Estimated time needed: **60** minutes
# 

# Space X advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is because Space X can reuse the first stage. Therefore if we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against space X for a rocket launch.   In this lab, you will create a machine learning pipeline  to predict if the first stage will land given the data from the preceding labs.
# 

# ![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/landing_1.gif)
# 

# Several examples of an unsuccessful landing are shown here:
# 

# ![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/crash.gif)
# 

# Most unsuccessful landings are planed. Space X; performs a controlled landing in the oceans.
# 

# ## Objectives
# 

# Perform exploratory  Data Analysis and determine Training Labels
# 
# *   create a column for the class
# *   Standardize the data
# *   Split into training data and test data
# 
# \-Find best Hyperparameter for SVM, Classification Trees and Logistic Regression
# 
# *   Find the method performs best using test data
# 

# ## Import Libraries and Define Auxiliary Functions
# 

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install scikit-learn')


# We will import the following libraries for the lab
# 

# In[3]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# This function is to plot the confusion matrix.
# 

# In[5]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show() 


# ## Load the dataframe
# 

# Load the data
# 

# In[7]:


data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")


# In[8]:


data.head()


# In[11]:


X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv')


# In[13]:


X.head(100)


# ## TASK  1
# 

# Create a NumPy array from the column <code>Class</code> in <code>data</code>, by applying the method <code>to_numpy()</code>  then
# assign it  to the variable <code>Y</code>,make sure the output is a  Pandas series (only one bracket df\['name of  column']).
# 

# In[15]:


# Importar pandas
import pandas as pd

# Cargar el DataFrame
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# Crear la variable Y como una serie de Pandas a partir de la columna 'Class'
Y = pd.Series(data['Class'].to_numpy())

# Verificar el resultado
print(Y.head())


# ## TASK  2
# 

# Standardize the data in <code>X</code> then reassign it to the variable  <code>X</code> using the transform provided below.
# 

# In[55]:


# students get this 
transform = preprocessing.StandardScaler()


# We split the data into training and testing data using the  function  <code>train_test_split</code>.   The training data is divided into validation data, a second set used for training  data; then the models are trained and hyperparameters are selected using the function <code>GridSearchCV</code>.
# 

# ## TASK  3
# 

# Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to  0.2 and random_state to 2. The training data and test data should be assigned to the following labels.
# 

# <code>X_train, X_test, Y_train, Y_test</code>
# 

# In[62]:


# Importar la función train_test_split
from sklearn.model_selection import train_test_split

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Verificar las dimensiones de los conjuntos
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# we can see we only have 18 test samples.
# 

# In[64]:


Y_test.shape


# ## TASK  4
# 

# Create a logistic regression object  then create a  GridSearchCV object  <code>logreg_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[66]:


parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# In[68]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()


# We output the <code>GridSearchCV</code> object for logistic regression. We display the best parameters using the data attribute <code>best_params\_</code> and the accuracy on the validation data using the data attribute <code>best_score\_</code>.
# 

# In[139]:


# Importar las bibliotecas necesarias
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Crear el objeto de regresión logística
logreg = LogisticRegression()

# Definir el diccionario de parámetros
parameters = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Crear el objeto GridSearchCV con cv=10
logreg_cv = GridSearchCV(logreg, parameters, cv=10)

# Ajustar el modelo a los datos de entrenamiento
logreg_cv.fit(X_train, Y_train)

# Imprimir los mejores parámetros y la mejor puntuación
print("Mejores parámetros:", logreg_cv.best_params_)
print("Mejor puntuación (accuracy):", logreg_cv.best_score_)


# ## TASK  5
# 

# Calculate the accuracy on the test data using the method <code>score</code>:
# 

# In[141]:


# Paso para calcular la precisión del modelo de Regresión Logística en los datos de prueba

# Asegúrate de que logreg_cv.fit(X_train, y_train) ya se ejecutó ANTES de este código.
# Asegúrate de que X_test y y_test estén definidos (a partir de train_test_split).

print("Calculando la precisión del modelo de Regresión Logística en los datos de prueba...")

# Calcula la precisión usando el método .score() del objeto GridSearchCV ajustado
accuracy_logreg = logreg_cv.score(X_test, Y_test)

# Imprime la precisión calculada para verificar
print(f"La precisión del modelo de Regresión Logística en los datos de prueba es: {accuracy_logreg:.4f}")

# Ahora la variable accuracy_logreg ya existe y puedes pasar a ejecutar el código de la TAREA 12.


# In[143]:


# Calcular la precisión en los datos de prueba
test_accuracy = logreg_cv.score(X_test, Y_test)

# Imprimir la precisión
print("Precisión en los datos de prueba:", test_accuracy)


# Lets look at the confusion matrix:
# 

# In[149]:


# Calcular las predicciones en los datos de prueba
yhat = logreg_cv.predict(X_test)

# Llamar a la función plot_confusion_matrix con la variable correcta
plot_confusion_matrix(Y_test, yhat)


# Examining the confusion matrix, we see that logistic regression can distinguish between the different classes.  We see that the problem is false positives.
# 
# Overview:
# 
# True Postive - 12 (True label is landed, Predicted label is also landed)
# 
# False Postive - 3 (True label is not landed, Predicted label is landed)
# 

# ## TASK  6
# 

# Create a support vector machine object then  create a  <code>GridSearchCV</code> object  <code>svm_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[35]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()


# In[37]:


# Importar las bibliotecas necesarias
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Crear el objeto de SVM
svm = SVC()

# Definir el diccionario de parámetros
parameters = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Crear el objeto GridSearchCV con cv=10
svm_cv = GridSearchCV(svm, parameters, cv=10)

# Ajustar el modelo a los datos de entrenamiento
svm_cv.fit(X_train, Y_train)

# Imprimir los mejores parámetros y la mejor puntuación
print("Mejores parámetros:", svm_cv.best_params_)
print("Mejor puntuación (accuracy):", svm_cv.best_score_)


# In[39]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# ## TASK  7
# 

# Calculate the accuracy on the test data using the method <code>score</code>:
# 

# In[41]:


# Calcular la precisión en los datos de prueba
test_accuracy = svm_cv.score(X_test, Y_test)

# Imprimir la precisión
print("Precisión en los datos de prueba:", test_accuracy)


# We can plot the confusion matrix
# 

# In[43]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ## TASK  8
# 

# Create a decision tree classifier object then  create a  <code>GridSearchCV</code> object  <code>tree_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[47]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[78]:


# Define el diccionario de parámetros para la búsqueda en cuadrícula
parameters = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [2*n for n in range(1,10)],
              'max_features': [None, 'sqrt'], # 'auto' fue renombrado a 'sqrt' en versiones recientes de scikit-learn, None es otra opción común
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10]}

# Crea un objeto clasificador de árbol de decisión
tree = DecisionTreeClassifier()

# Crea un objeto GridSearchCV con el clasificador, los parámetros y cv=10
tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10)

# Ajusta el objeto GridSearchCV a tus datos de entrenamiento
# Reemplaza 'X_train' y 'y_train' con tus conjuntos de datos de características y etiquetas de entrenamiento reales.
# Asumiendo que ya tienes tus datos de entrenamiento cargados en estas variables.
print("Iniciando la búsqueda de los mejores parámetros para el Árbol de Decisión...")
tree_cv.fit(X_train, Y_train) # <-- Asegúrate de usar tus datos de entrenamiento aquí

print("Búsqueda completada.")

# Opcional: Mostrar los mejores parámetros encontrados y el mejor score
print("Mejores parámetros encontrados:", tree_cv.best_params_)
print("Mejor score (precisión media de validación cruzada):", tree_cv.best_score_)


# In[80]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# ## TASK  9
# 

# Calculate the accuracy of tree_cv on the test data using the method <code>score</code>:
# 

# In[84]:


# Calcula la precisión en el conjunto de prueba
accuracy_tree = tree_cv.score(X_test, Y_test)

# Imprime la precisión
print(f"La precisión del modelo Árbol de Decisión en los datos de prueba es: {accuracy_tree:.4f}")


# We can plot the confusion matrix
# 

# In[86]:


yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ## TASK  10
# 

# Create a k nearest neighbors object then  create a  <code>GridSearchCV</code> object  <code>knn_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[88]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()


# In[92]:


# Define el diccionario de parámetros para la búsqueda en cuadrícula (ya proporcionado)
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]} # p=1 para distancia Manhattan, p=2 para distancia Euclidiana

# Crea un objeto clasificador K-Nearest Neighbors (ya proporcionado)
KNN = KNeighborsClassifier()

# Crea un objeto GridSearchCV con el clasificador, los parámetros y cv=10
print("Creando el objeto GridSearchCV para KNN...")
knn_cv = GridSearchCV(estimator=KNN, param_grid=parameters, cv=10)

# Ajusta el objeto GridSearchCV a tus datos de entrenamiento
# ¡Asegúrate de que X_train y y_train estén definidos a partir de tus pasos de carga y división de datos!
print("Iniciando la búsqueda de los mejores parámetros para KNN...")
knn_cv.fit(X_train, Y_train) # <-- Usa tus datos de entrenamiento aquí

print("Búsqueda para KNN completada.")

# Opcional: Mostrar los mejores parámetros encontrados y el mejor score
print("Mejores parámetros encontrados para KNN:", knn_cv.best_params_)
print("Mejor score (precisión media de validación cruzada) para KNN:", knn_cv.best_score_)

# El mejor modelo KNN ajustado está disponible como knn_cv.best_estimator_


# In[94]:


print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)


# ## TASK  11
# 

# Calculate the accuracy of knn_cv on the test data using the method <code>score</code>:
# 

# In[98]:


# Calcula la precisión del mejor modelo KNN en el conjunto de prueba
accuracy_knn = knn_cv.score(X_test, Y_test)

# Imprime la precisión
print(f"La precisión del modelo K-Nearest Neighbors en los datos de prueba es: {accuracy_knn:.4f}")


# We can plot the confusion matrix
# 

# In[100]:


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ## TASK  12
# 

# Find the method performs best:
# 

# In[151]:


# TAREA 12: Encontrar el método que mejor funciona

# Asegúrate de haber ejecutado las tareas anteriores para calcular la precisión
# en los datos de prueba para cada modelo.
# Necesitas tener las variables con las precisiones, por ejemplo:
# accuracy_logreg = logreg_cv.score(X_test, y_test) # (De una tarea similar a la 9 y 11 pero para Regresión Logística)
# accuracy_tree = tree_cv.score(X_test, y_test)     # (De la TAREA 9)
# accuracy_knn = knn_cv.score(X_test, y_test)       # (De la TAREA 11)

print("Comparando la precisión de los modelos en los datos de prueba:")
print(f"Precisión Regresión Logística: {accuracy_logreg:.4f}")
print(f"Precisión Árbol de Decisión:   {accuracy_tree:.4f}")
print(f"Precisión K-Nearest Neighbors: {accuracy_knn:.4f}")

# Crea un diccionario para comparar fácilmente
model_accuracies = {
    'Regresión Logística': accuracy_logreg,
    'Árbol de Decisión': accuracy_tree,
    'K-Nearest Neighbors': accuracy_knn
}

# Encuentra el modelo con la precisión máxima
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]

print(f"\nEl modelo que mejor funciona en los datos de prueba es: {best_model_name}")
print(f"Con una precisión de: {best_accuracy:.4f}")


# ## Authors
# 

# [Pratiksha Verma](https://www.linkedin.com/in/pratiksha-verma-6487561b1/)
# 

# <!--## Change Log--!>
# 

# <!--| Date (YYYY-MM-DD) | Version | Changed By      | Change Description      |
# | ----------------- | ------- | -------------   | ----------------------- |
# | 2022-11-09        | 1.0     | Pratiksha Verma | Converted initial version to Jupyterlite|--!>
# 

# ### <h3 align="center"> IBM Corporation 2022. All rights reserved. <h3/>
# 
