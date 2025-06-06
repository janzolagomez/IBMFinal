# -*- coding: utf-8 -*-
"""jupyter-labs-eda-sql-coursera_sqllite.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1u3qHexDDRbuf1uN6aMj9LzO4-ggOt_h4

<p style="text-align:center">
    <a href="https://skills.network" target="_blank">
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
    </a>
</p>

<h1 align=center><font size = 5>Assignment: SQL Notebook for Peer Assignment</font></h1>

Estimated time needed: **60** minutes.

## Introduction
Using this Python notebook you will:

1.  Understand the Spacex DataSet
2.  Load the dataset  into the corresponding table in a Db2 database
3.  Execute SQL queries to answer assignment questions

## Overview of the DataSet

SpaceX has gained worldwide attention for a series of historic milestones.

It is the only private company ever to return a spacecraft from low-earth orbit, which it first accomplished in December 2010.
SpaceX advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars wheras other providers cost upward of 165 million dollars each, much of the savings is because Space X can reuse the first stage.


Therefore if we can determine if the first stage will land, we can determine the cost of a launch.

This information can be used if an alternate company wants to bid against SpaceX for a rocket launch.

This dataset includes a record for each payload carried during a SpaceX mission into outer space.

### Download the datasets

This assignment requires you to load the spacex dataset.

In many cases the dataset to be analyzed is available as a .CSV (comma separated values) file, perhaps on the internet. Click on the link below to download and save the dataset (.CSV file):

 <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv" target="_blank">Spacex DataSet</a>
"""

!pip install sqlalchemy==1.3.9

"""### Connect to the database

Let us first load the SQL extension and establish a connection with the database

"""

!pip install ipython-sql
!pip install ipython-sql prettytable

# Commented out IPython magic to ensure Python compatibility.
# %load_ext sql

import csv, sqlite3
import prettytable
prettytable.DEFAULT = 'DEFAULT'

con = sqlite3.connect("my_data1.db")
cur = con.cursor()

!pip install -q pandas

# Commented out IPython magic to ensure Python compatibility.
# %sql sqlite:///my_data1.db

import pandas as pd
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_2/data/Spacex.csv")
df.to_sql("SPACEXTBL", con, if_exists='replace', index=False,method="multi")

"""**Note:This below code is added to remove blank rows from table**

"""

# Commented out IPython magic to ensure Python compatibility.
#DROP THE TABLE IF EXISTS

# %sql DROP TABLE IF EXISTS SPACEXTABLE;

# Commented out IPython magic to ensure Python compatibility.
# %sql create table SPACEXTABLE as select * from SPACEXTBL where Date is not null

"""## Tasks

Now write and execute SQL queries to solve the assignment tasks.

**Note: If the column names are in mixed case enclose it in double quotes
   For Example "Landing_Outcome"**

### Task 1




##### Display the names of the unique launch sites  in the space mission

"""

# Ejecutar la consulta SQL para obtener los nombres únicos de los sitios de lanzamiento
cur.execute("SELECT DISTINCT Launch_Site FROM SPACEXTABLE;")

# Obtener todos los resultados de la consulta
unique_launch_sites = cur.fetchall()

# Imprimir los resultados
print("Sitios de lanzamiento únicos:")
for site in unique_launch_sites:
    print(site[0])

"""
### Task 2


#####  Display 5 records where launch sites begin with the string 'CCA'
"""

# Ejecutar la consulta SQL para obtener 5 registros donde el sitio de lanzamiento comienza con 'CCA'
cur.execute("SELECT * FROM SPACEXTABLE WHERE Launch_Site LIKE 'CCA%' LIMIT 5;")

# Obtener los resultados de la consulta
cca_launch_sites = cur.fetchall()

# Obtener los nombres de las columnas para una mejor visualización con Pandas
column_names = [description[0] for description in cur.description]

# Imprimir los resultados usando PrettyTable (si está instalado)
try:
    from prettytable import PrettyTable
    table = PrettyTable(column_names)
    for row in cca_launch_sites:
        table.add_row(row)
    print("Primeros 5 registros con sitios de lanzamiento que comienzan con 'CCA':")
    print(table)
except ImportError:
    print("Primeros 5 registros con sitios de lanzamiento que comienzan con 'CCA':")
    for row in cca_launch_sites:
        print(row)

"""### Task 3




##### Display the total payload mass carried by boosters launched by NASA (CRS)

"""

# Ejecutar la consulta SQL con el nombre de columna correcto
cur.execute("SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Customer = 'NASA (CRS)';")

# Obtener el resultado de la consulta
total_payload_nasa_crs = cur.fetchone()[0]

# Imprimir el resultado
print(f"Masa total de carga útil transportada por boosters lanzados por NASA (CRS): {total_payload_nasa_crs} kg")

"""### Task 4




##### Display average payload mass carried by booster version F9 v1.1

"""

# Ejecutar la consulta SQL para obtener el promedio de PayloadMass para la versión de booster 'F9 v1.1'
cur.execute("SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Booster_Version = 'F9 v1.1';")

# Obtener el resultado de la consulta
average_payload_f9v1_1 = cur.fetchone()[0]

# Imprimir el resultado
print(f"Masa promedio de carga útil transportada por la versión de booster F9 v1.1: {average_payload_f9v1_1:.2f} kg")

"""### Task 5

##### List the date when the first succesful landing outcome in ground pad was acheived.


_Hint:Use min function_

"""

# Ejecutar la consulta SQL para obtener la fecha del primer aterrizaje exitoso en un ground pad
cur.execute("SELECT MIN(Date) FROM SPACEXTABLE WHERE Landing_Outcome = 'Success (ground pad)';")

# Obtener el resultado de la consulta
first_successful_ground_pad = cur.fetchone()[0]

# Imprimir el resultado
print(f"Fecha del primer aterrizaje exitoso en un ground pad: {first_successful_ground_pad}")

"""### Task 6

##### List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000

"""

# Conectar a la base de datos SQLite
con = sqlite3.connect('my_data1.db')
cur = con.cursor()

# Ejecutar la consulta SQL para obtener los nombres de los boosters con aterrizaje exitoso en drone ship
# y masa de carga útil mayor que 4000 y menor que 6000
cur.execute("SELECT Booster_Version FROM SPACEXTABLE WHERE Landing_Outcome = 'Success (drone ship)' AND PAYLOAD_MASS__KG_ > 4000 AND PAYLOAD_MASS__KG_ < 6000;")

# Obtener los resultados de la consulta
successful_drone_ship_boosters = cur.fetchall()

# Imprimir los resultados
print("Boosters con aterrizaje exitoso en drone ship y masa de carga útil entre 4000 y 6000 kg:")
for booster in successful_drone_ship_boosters:
    print(booster[0])

"""### Task 7




##### List the total number of successful and failure mission outcomes

"""

# Conectar a la base de datos SQLite
con = sqlite3.connect('my_data1.db')
cur = con.cursor()

# Ejecutar la consulta SQL para obtener el conteo de resultados exitosos y fallidos de la misión
cur.execute("SELECT Mission_Outcome, COUNT(*) FROM SPACEXTABLE WHERE Mission_Outcome LIKE 'Success%' OR Mission_Outcome LIKE 'Failure%' GROUP BY Mission_Outcome;")

# Obtener los resultados de la consulta
mission_outcomes_count = cur.fetchall()

# Imprimir los resultados
print("Total de resultados exitosos y fallidos de la misión:")
for outcome, count in mission_outcomes_count:
    print(f"{outcome}: {count}")

"""### Task 8



##### List all the booster_versions that have carried the maximum payload mass. Use a subquery.

"""

# Conectar a la base de datos SQLite
con = sqlite3.connect('my_data1.db')
cur = con.cursor()

# Ejecutar la consulta SQL para obtener las booster_versions con la máxima payload mass usando una subconsulta
cur.execute("SELECT Booster_Version FROM SPACEXTABLE WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTABLE);")

# Obtener los resultados de la consulta
boosters_with_max_payload = cur.fetchall()

# Imprimir los resultados
print("Booster versions que han transportado la máxima masa de carga útil:")
for booster in boosters_with_max_payload:
    print(booster[0])

"""### Task 9


##### List the records which will display the month names, failure landing_outcomes in drone ship ,booster versions, launch_site for the months in year 2015.

**Note: SQLLite does not support monthnames. So you need to use  substr(Date, 6,2) as month to get the months and substr(Date,0,5)='2015' for year.**

"""

# Conectar a la base de datos SQLite
con = sqlite3.connect('my_data1.db')
cur = con.cursor()

# Ejecutar la consulta SQL para obtener los registros con las condiciones especificadas para el año 2015
cur.execute("""
SELECT
    CASE
        WHEN SUBSTR(Date, 6, 2) = '01' THEN 'January'
        WHEN SUBSTR(Date, 6, 2) = '02' THEN 'February'
        WHEN SUBSTR(Date, 6, 2) = '03' THEN 'March'
        WHEN SUBSTR(Date, 6, 2) = '04' THEN 'April'
        WHEN SUBSTR(Date, 6, 2) = '05' THEN 'May'
        WHEN SUBSTR(Date, 6, 2) = '06' THEN 'June'
        WHEN SUBSTR(Date, 6, 2) = '07' THEN 'July'
        WHEN SUBSTR(Date, 6, 2) = '08' THEN 'August'
        WHEN SUBSTR(Date, 6, 2) = '09' THEN 'September'
        WHEN SUBSTR(Date, 6, 2) = '10' THEN 'October'
        WHEN SUBSTR(Date, 6, 2) = '11' THEN 'November'
        WHEN SUBSTR(Date, 6, 2) = '12' THEN 'December'
        ELSE NULL
    END AS Month,
    Landing_Outcome,
    Booster_Version,
    Launch_Site
FROM
    SPACEXTABLE
WHERE
    SUBSTR(Date, 1, 4) = '2015' AND Landing_Outcome LIKE 'Failure (drone ship)%';
""")

# Obtener los resultados de la consulta
failure_drone_ship_2015 = cur.fetchall()

# Imprimir los resultados
print("Registros de fallos de aterrizaje en drone ship en 2015:")
for record in failure_drone_ship_2015:
    print(f"Month: {record[0]}, Landing Outcome: {record[1]}, Booster Version: {record[2]}, Launch Site: {record[3]}")

"""### Task 10




##### Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.

"""

# Conectar a la base de datos SQLite
con = sqlite3.connect('my_data1.db')
cur = con.cursor()

# Ejecutar la consulta SQL para obtener el conteo de resultados de aterrizaje dentro del rango de fechas
cur.execute("""
SELECT Landing_Outcome, COUNT(*) AS OutcomeCount
FROM SPACEXTABLE
WHERE Date BETWEEN '2010-06-04' AND '2017-03-20'
GROUP BY Landing_Outcome
ORDER BY OutcomeCount DESC;
""")

# Obtener los resultados de la consulta
landing_outcome_ranking = cur.fetchall()

# Imprimir los resultados
print("Ranking de conteo de resultados de aterrizaje (2010-06-04 a 2017-03-20):")
for outcome, count in landing_outcome_ranking:
    print(f"{outcome}: {count}")

# Cerrar la conexión (buena práctica)
con.close()

"""### Reference Links

* <a href ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20String%20Patterns%20-%20Sorting%20-%20Grouping/instructional-labs.md.html?origin=www.coursera.org">Hands-on Lab : String Patterns, Sorting and Grouping</a>  

*  <a  href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20Built-in%20functions%20/Hands-on_Lab__Built-in_Functions.md.html?origin=www.coursera.org">Hands-on Lab: Built-in functions</a>

*  <a  href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Labs_Coursera_V5/labs/Lab%20-%20Sub-queries%20and%20Nested%20SELECTs%20/instructional-labs.md.html?origin=www.coursera.org">Hands-on Lab : Sub-queries and Nested SELECT Statements</a>

*   <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Module%205/DB0201EN-Week3-1-3-SQLmagic.ipynb">Hands-on Tutorial: Accessing Databases with SQL magic</a>

*  <a href= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DB0201EN-SkillsNetwork/labs/Module%205/DB0201EN-Week3-1-4-Analyzing.ipynb">Hands-on Lab: Analyzing a real World Data Set</a>

## Author(s)

<h4> Lakshmi Holla </h4>

## Other Contributors

<h4> Rav Ahuja </h4>

<!--
## Change log
| Date | Version | Changed by | Change Description |
|------|--------|--------|---------|
| 2024-07-10 | 1.1 |Anita Verma | Changed Version|
| 2021-07-09 | 0.2 |Lakshmi Holla | Changes made in magic sql|
| 2021-05-20 | 0.1 |Lakshmi Holla | Created Initial Version |
-->

## <h3 align="center"> © IBM Corporation 2021. All rights reserved. <h3/>
"""