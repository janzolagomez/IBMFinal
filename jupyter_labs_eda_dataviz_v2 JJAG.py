# -*- coding: utf-8 -*-
"""jupyter-labs-eda-dataviz-v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wesE3dT2222Na3Cwmag6Usw6txA2Sujn

<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
    </a>
</p>

# **SpaceX  Falcon 9 First Stage Landing Prediction**

## Hands-on Lab: Complete the EDA with Visualization

Estimated time needed: **70** minutes

In this assignment, we will predict if the Falcon 9 first stage will land successfully. SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is due to the fact that SpaceX can reuse the first stage.

In this lab, you will perform Exploratory Data Analysis and Feature Engineering.

Falcon 9 first stage will land successfully

![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/landing_1.gif)

Several examples of an unsuccessful landing are shown here:

![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/crash.gif)

Most unsuccessful landings are planned. Space X performs a controlled landing in the oceans.

## Objectives
Perform exploratory Data Analysis and Feature Engineering using `Pandas` and `Matplotlib`

- Exploratory Data Analysis
- Preparing Data  Feature Engineering

----

Install the below libraries
"""

!pip install pandas
!pip install numpy
!pip install seaborn
!pip install matplotlib

"""### Import Libraries and Define Auxiliary Functions

We will import the following libraries the lab
"""

# andas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
#NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns

"""## Exploratory Data Analysis

First, let's read the SpaceX dataset into a Pandas dataframe and print its summary
"""

df=pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

# df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')

df.head(5)

df.columns

"""First, let's try to see how the `FlightNumber` (indicating the continuous launch attempts.) and `Payload` variables would affect the launch outcome.

We can plot out the <code>FlightNumber</code> vs. <code>PayloadMass</code>and overlay the outcome of the launch. We see that as the flight number increases, the first stage is more likely to land successfully. The payload mass is also important; it seems the more massive the payload, the less likely the first stage will return.

"""

sns.catplot(y="PayloadMass", x="FlightNumber", hue="Class", data=df, aspect = 5)
plt.xlabel("Flight Number",fontsize=20)
plt.ylabel("Pay load Mass (kg)",fontsize=20)
plt.show()

"""Next, let's drill down to each site visualize its detailed launch records.

### TASK 1: Visualize the relationship between Flight Number and Launch Site

Use the function <code>catplot</code> to plot <code>FlightNumber</code> vs <code>LaunchSite</code>, set the  parameter <code>x</code>  parameter to <code>FlightNumber</code>,set the  <code>y</code> to <code>Launch Site</code> and set the parameter <code>hue</code> to <code>'class'</code>
"""

# Plot a scatter point chart with x axis to be Flight Number and y axis to be the launch site, and hue to be the class value
sns.catplot(x='FlightNumber', y='LaunchSite', hue='Class', data=df, aspect=3)
plt.xlabel("Número de Vuelo", fontsize=14)
plt.ylabel("Sitio de Lanzamiento", fontsize=14)
plt.title("Relación entre Número de Vuelo y Sitio de Lanzamiento (Clasificado por Éxito/Fallo)", fontsize=16)
plt.grid(axis='x', linestyle='--')
plt.show()

"""Now try to explain the patterns you found in the Flight Number vs. Launch Site scatter point plots.

### TASK 2: Visualize the relationship between Payload and Launch Site

We also want to observe if there is any relationship between launch sites and their payload mass.
"""

# Plot a scatter point chart with x axis to be Pay Load Mass (kg) and y axis to be the launch site, and hue to be the class value
sns.catplot(x='LaunchSite', y='PayloadMass', data=df, kind='box', aspect=2)
plt.xlabel("Sitio de Lanzamiento", fontsize=14)
plt.ylabel("Masa de la Carga Útil (kg)", fontsize=14)
plt.title("Relación entre Masa de la Carga Útil y Sitio de Lanzamiento", fontsize=16)
plt.grid(axis='y', linestyle='--')
plt.show()

"""Now if you observe Payload Vs. Launch Site scatter point chart you will find for the VAFB-SLC  launchsite there are no  rockets  launched for  heavypayload mass(greater than 10000).

### TASK  3: Visualize the relationship between success rate of each orbit type

Next, we want to visually check if there are any relationship between success rate and orbit type.

Let's create a `bar chart` for the sucess rate of each orbit
"""

# HINT use groupby method on Orbit column and get the mean of Class column
# Calcular la tasa de éxito por tipo de órbita
success_rate_orbit = df.groupby('Orbit')['Class'].mean().sort_values(ascending=False)

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
sns.barplot(x=success_rate_orbit.index, y=success_rate_orbit.values, palette='viridis')
plt.xlabel("Tipo de Órbita", fontsize=14)
plt.ylabel("Tasa de Éxito", fontsize=14)
plt.title("Tasa de Éxito por Tipo de Órbita", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)  # La tasa de éxito está entre 0 y 1
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

print("\nTasa de éxito por tipo de órbita:")
print(success_rate_orbit)

"""Analyze the ploted bar chart try to find which orbits have high sucess rate.

### TASK  4: Visualize the relationship between FlightNumber and Orbit type

For each orbit, we want to see if there is any relationship between FlightNumber and Orbit type.
"""

# Plot a scatter point chart with x axis to be FlightNumber and y axis to be the Orbit, and hue to be the class value
plt.figure(figsize=(12, 6))
sns.scatterplot(x='FlightNumber', y='Orbit', hue='Class', data=df, alpha=0.7)
plt.xlabel("Número de Vuelo", fontsize=14)
plt.ylabel("Tipo de Órbita", fontsize=14)
plt.title("Relación entre Número de Vuelo y Tipo de Órbita (Clasificado por Éxito/Fallo)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""You should see that in the LEO orbit the Success appears related to the number of flights; on the other hand, there seems to be no relationship between flight number when in GTO orbit.

### TASK  5: Visualize the relationship between Payload and Orbit type

Similarly, we can plot the Payload vs. Orbit scatter point charts to reveal the relationship between Payload and Orbit type
"""

# Plot a scatter point chart with x axis to be Payload and y axis to be the Orbit, and hue to be the class value
plt.figure(figsize=(12, 6))
sns.scatterplot(x='PayloadMass', y='Orbit', hue='Class', data=df, alpha=0.7)
plt.xlabel("Masa de la Carga Útil (kg)", fontsize=14)
plt.ylabel("Tipo de Órbita", fontsize=14)
plt.title("Relación entre Masa de la Carga Útil y Tipo de Órbita (Clasificado por Éxito/Fallo)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

"""With heavy payloads the successful landing or positive landing rate are more for Polar,LEO and ISS.   

However for GTO we cannot distinguish this well as both positive landing rate and negative landing(unsuccessful mission) are both there here.

### TASK  6: Visualize the launch success yearly trend

You can plot a line chart with x axis to be <code>Year</code> and y axis to be average success rate, to get the average launch success trend.

The function will help you get the year from the date:
"""

# A function to Extract years from the date
year=[]
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year

# Plot a line chart with x axis to be the extracted year and y axis to be the success rate
# Función para extraer el año de la fecha
year = []
def Extract_year(date):
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year

# Aplicar la función para extraer los años
years = Extract_year(df['Date'])
df['Year'] = years

# Calcular la tasa de éxito promedio por año
success_rate_year = df.groupby('Year')['Class'].mean()

# Crear el gráfico de líneas
plt.figure(figsize=(12, 6))
sns.lineplot(x=success_rate_year.index, y=success_rate_year.values, marker='o', color='blue')
plt.xlabel("Año", fontsize=14)
plt.ylabel("Tasa de Éxito Promedio", fontsize=14)
plt.title("Tendencia Anual de la Tasa de Éxito de Lanzamientos", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)  # Ajustar el límite superior para una mejor visualización
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\nTasa de éxito promedio por año:")
print(success_rate_year)

"""You can observe that the success rate since 2013 kept increasing till 2017 (stable in 2014) and after 2015 it started increasing.

## Features Engineering
"""



"""By now, you should obtain some preliminary insights about how each important variable would affect the success rate, we will select the features that will be used in success prediction in the future module.

"""

features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]
features.head()

"""### TASK  7: Create dummy variables to categorical columns

Use the function <code>get_dummies</code> and <code>features</code> dataframe to apply OneHotEncoder to the column <code>Orbits</code>, <code>LaunchSite</code>, <code>LandingPad</code>, and <code>Serial</code>. Assign the value to the variable <code>features_one_hot</code>, display the results using the method head. Your result dataframe must include all features including the encoded ones.
"""

# HINT: Use get_dummies() function on the categorical columns
# Aplicar One-Hot Encoding a las columnas especificadas
features_one_hot = pd.get_dummies(df, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'], dummy_na=True)

# Mostrar las primeras filas del DataFrame resultante
print(features_one_hot.head())

# Imprimir todas las columnas
print(features_one_hot.columns)

"""### TASK  8: Cast all numeric columns to `float64`

Now that our <code>features_one_hot</code> dataframe only contains numbers cast the entire dataframe to variable type <code>float64</code>
"""

import numpy as np

# Identificar las columnas que no son de tipo 'object' (string)
numeric_cols = features_one_hot.select_dtypes(exclude=['object']).columns

# Convertir solo las columnas numéricas a float64
features_one_hot[numeric_cols] = features_one_hot[numeric_cols].astype(np.float64)

# Verificar los tipos de datos del DataFrame resultante
print(features_one_hot.dtypes)

"""We can now export it to a <b>CSV</b> for the next section,but to make the answers consistent, in the next lab we will provide data in a pre-selected date range.

<code>features_one_hot.to_csv('dataset_part_3.csv', index=False)</code>

## Authors

<a href="https://www.linkedin.com/in/joseph-s-50398b136/">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

<a href="https://www.linkedin.com/in/nayefaboutayoun/">Nayef Abou Tayoun</a> is a Data Scientist at IBM and pursuing a Master of Management in Artificial intelligence degree at Queen's University.

## Change Log

| Date (YYYY-MM-DD) | Version | Changed By | Change Description      |
| ----------------- | ------- | ---------- | ----------------------- |
| 2021-10-12        | 1.1     | Lakshmi Holla     | Modified markdown |
| 2020-09-20        | 1.0     | Joseph     | Modified Multiple Areas |
| 2020-11-10       | 1.1    | Nayef      | updating the input data |

Copyright © 2020 IBM Corporation. All rights reserved.
"""