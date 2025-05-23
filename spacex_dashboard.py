#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Instalar las librerías necesarias
get_ipython().system('pip install pandas dash jupyter-dash')


# In[11]:


# Instalar la librería requests (si no está instalada)
get_ipython().system('pip install requests')

# Importar la librería requests
import requests

# URLs de los archivos
dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
app_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/t4-Vy4iOU19i8y6E3Px_ww/spacex-dash-app.py"

# Descargar el conjunto de datos
try:
    response = requests.get(dataset_url)
    response.raise_for_status()  # Verifica si la descarga fue exitosa
    with open('spacex_launch_dash.csv', 'wb') as file:
        file.write(response.content)
    print("Descarga exitosa: spacex_launch_dash.csv")
except Exception as e:
    print(f"Error al descargar el conjunto de datos: {e}")

# Descargar el archivo base de la aplicación
try:
    response = requests.get(app_url)
    response.raise_for_status()  # Verifica si la descarga fue exitosa
    with open('spacex_dash_app.py', 'wb') as file:
        file.write(response.content)
    print("Descarga exitosa: spacex_dash_app.py")
except Exception as e:
    print(f"Error al descargar el archivo de la aplicación: {e}")

# Verificar que los archivos se descargaron
get_ipython().system('dir')


# In[17]:


# Importar las librerías necesarias
from dash import Dash, dcc, html
import pandas as pd
from IPython.display import display

# Cargar el conjunto de datos
spacex_df = pd.read_csv('spacex_launch_dash.csv')

# Inspeccionar las primeras filas del DataFrame
display(spacex_df.head())

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir el diseño básico (layout) de la aplicación
app.layout = html.Div([
    html.H1('SpaceX Launch Dashboard', style={'textAlign': 'center'})
])

# Ejecutar la aplicación en Jupyter Notebook
app.run(mode='inline', port=8051, debug=False)


# In[19]:


# Importar las librerías necesarias
from dash import Dash, dcc, html
import pandas as pd
from IPython.display import display

# Cargar el conjunto de datos
spacex_df = pd.read_csv('spacex_launch_dash.csv')

# Inspeccionar las primeras filas del DataFrame
display(spacex_df.head())

# Obtener los sitios de lanzamiento únicos
launch_sites = spacex_df['Launch Site'].unique()
dropdown_options = [{'label': 'All Sites', 'value': 'ALL'}] + \
                  [{'label': site, 'value': site} for site in launch_sites]

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir el diseño (layout) con el menú desplegable
app.layout = html.Div([
    html.H1('SpaceX Launch Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',
        placeholder='Selecciona un sitio de lanzamiento',
        searchable=True
    )
])

# Ejecutar la aplicación en Jupyter Notebook
app.run(mode='inline', port=8051, debug=False)


# In[21]:


# Importar las librerías necesarias
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from IPython.display import display

# Cargar el conjunto de datos
spacex_df = pd.read_csv('spacex_launch_dash.csv')

# Inspeccionar las primeras filas del DataFrame
display(spacex_df.head())

# Obtener los sitios de lanzamiento únicos
launch_sites = spacex_df['Launch Site'].unique()
dropdown_options = [{'label': 'All Sites', 'value': 'ALL'}] + \
                  [{'label': site, 'value': site} for site in launch_sites]

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir el diseño (layout) con el menú desplegable y el gráfico de pastel
app.layout = html.Div([
    html.H1('SpaceX Launch Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',
        placeholder='Selecciona un sitio de lanzamiento',
        searchable=True
    ),
    html.Br(),
    dcc.Graph(id='success-pie-chart')
])

# Añadir la función de callback para el gráfico de pastel
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    if entered_site == 'ALL':
        # Para todos los sitios, usar todos los datos
        fig = px.pie(
            spacex_df,
            values='class',
            names='class',
            title='Tasa de éxito de lanzamientos (Todos los sitios)'
        )
        fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
    else:
        # Filtrar por el sitio seleccionado
        filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
        fig = px.pie(
            filtered_df,
            values='class',
            names='class',
            title=f'Tasa de éxito de lanzamientos para {entered_site}'
        )
        fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
    return fig

# Ejecutar la aplicación en Jupyter Notebook
app.run(mode='inline', port=8051, debug=False)


# In[25]:


# Importar las librerías necesarias
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from IPython.display import display

# Cargar el conjunto de datos
try:
    spacex_df = pd.read_csv('spacex_launch_dash.csv')
    display(spacex_df.head())
    print("Columnas del DataFrame:", spacex_df.columns.tolist())
    print("Valores únicos de 'class':", spacex_df['class'].unique())
except Exception as e:
    print(f"Error al cargar el archivo CSV: {e}")

# Obtener los sitios de lanzamiento únicos
launch_sites = spacex_df['Launch Site'].unique()
print("Sitios de lanzamiento:", launch_sites)
dropdown_options = [{'label': 'All Sites', 'value': 'ALL'}] + \
                  [{'label': site, 'value': site} for site in launch_sites]

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir el diseño (layout) con el menú desplegable y el gráfico de pastel
app.layout = html.Div([
    html.H1('SpaceX Launch Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',
        placeholder='Selecciona un sitio de lanzamiento',
        searchable=True
    ),
    html.Br(),
    dcc.Graph(id='success-pie-chart')
])

# Añadir la función de callback para el gráfico de pastel
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    try:
        if entered_site == 'ALL':
            # Contar los valores de 'class' para todos los sitios
            class_counts = spacex_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title='Tasa de éxito de lanzamientos (Todos los sitios)'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        else:
            # Filtrar por el sitio seleccionado
            filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
            if filtered_df.empty:
                print(f"Advertencia: No hay datos para el sitio {entered_site}")
                return px.pie(names=['Sin datos'], values=[1], title=f'Sin datos para {entered_site}')
            class_counts = filtered_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title=f'Tasa de éxito de lanzamientos para {entered_site}'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        return fig
    except Exception as e:
        print(f"Error en la función de callback: {e}")
        return px.pie(names=['Error'], values=[1], title='Error al generar el gráfico')

# Ejecutar la aplicación en Jupyter Notebook
try:
    app.run(mode='inline', port=8051, debug=False)
except Exception as e:
    print(f"Error al ejecutar la aplicación: {e}")


# In[27]:


from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from IPython.display import display

# Cargar el conjunto de datos
spacex_df = pd.read_csv('spacex_launch_dash.csv')
display(spacex_df.head())
print("Columnas del DataFrame:", spacex_df.columns.tolist())
print("Valores únicos de 'class':", spacex_df['class'].unique())

# Obtener los sitios de lanzamiento únicos
launch_sites = spacex_df['Launch Site'].unique()
print("Sitios de lanzamiento:", launch_sites)
dropdown_options = [{'label': 'All Sites', 'value': 'ALL'}] + \
                  [{'label': site, 'value': site} for site in launch_sites]

# Obtener el rango de masa de carga útil
min_payload = spacex_df['Payload Mass (kg)'].min()
max_payload = spacex_df['Payload Mass (kg)'].max()
print("Rango de masa de carga útil:", min_payload, max_payload)

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir el diseño con el menú desplegable, gráfico de pastel y control deslizante
app.layout = html.Div([
    html.H1('SpaceX Launch Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',
        placeholder='Selecciona un sitio de lanzamiento',
        searchable=True
    ),
    html.Br(),
    dcc.Graph(id='success-pie-chart'),
    html.Br(),
    html.Label('Rango de masa de carga útil (kg):'),
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        marks={i: str(i) for i in range(0, 10001, 1000)},
        value=[min_payload, max_payload]
    )
])

# Callback para el gráfico de pastel
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    try:
        if entered_site == 'ALL':
            class_counts = spacex_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title='Tasa de éxito de lanzamientos (Todos los sitios)'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        else:
            filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
            if filtered_df.empty:
                return px.pie(names=['Sin datos'], values=[1], title=f'Sin datos para {entered_site}')
            class_counts = filtered_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title=f'Tasa de éxito de lanzamientos para {entered_site}'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        return fig
    except Exception as e:
        print(f"Error en la función de callback: {e}")
        return px.pie(names=['Error'], values=[1], title='Error al generar el gráfico')

# Ejecutar la aplicación
app.run(mode='inline', port=8051, debug=False)


# In[29]:


from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from IPython.display import display

# Cargar el conjunto de datos
spacex_df = pd.read_csv('spacex_launch_dash.csv')
display(spacex_df.head())
print("Columnas del DataFrame:", spacex_df.columns.tolist())
print("Valores únicos de 'class':", spacex_df['class'].unique())

# Obtener los sitios de lanzamiento únicos
launch_sites = spacex_df['Launch Site'].unique()
print("Sitios de lanzamiento:", launch_sites)
dropdown_options = [{'label': 'All Sites', 'value': 'ALL'}] + \
                  [{'label': site, 'value': site} for site in launch_sites]

# Obtener el rango de masa de carga útil
min_payload = spacex_df['Payload Mass (kg)'].min()
max_payload = spacex_df['Payload Mass (kg)'].max()
print("Rango de masa de carga útil:", min_payload, max_payload)

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir el diseño con todos los componentes
app.layout = html.Div([
    html.H1('SpaceX Launch Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',
        placeholder='Selecciona un sitio de lanzamiento',
        searchable=True
    ),
    html.Br(),
    dcc.Graph(id='success-pie-chart'),
    html.Br(),
    html.Label('Rango de masa de carga útil (kg):'),
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        marks={i: str(i) for i in range(0, 10001, 1000)},
        value=[min_payload, max_payload]
    ),
    html.Br(),
    dcc.Graph(id='success-payload-scatter-chart')
])

# Callback para el gráfico de pastel
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    try:
        if entered_site == 'ALL':
            class_counts = spacex_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title='Tasa de éxito de lanzamientos (Todos los sitios)'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        else:
            filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
            if filtered_df.empty:
                return px.pie(names=['Sin datos'], values=[1], title=f'Sin datos para {entered_site}')
            class_counts = filtered_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title=f'Tasa de éxito de lanzamientos para {entered_site}'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        return fig
    except Exception as e:
        print(f"Error en la función de callback: {e}")
        return px.pie(names=['Error'], values=[1], title='Error al generar el gráfico')

# Callback para el gráfico de dispersión
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def get_scatter_chart(entered_site, payload_range):
    try:
        # Filtrar por rango de masa de carga útil
        filtered_df = spacex_df[
            (spacex_df['Payload Mass (kg)'] >= payload_range[0]) &
            (spacex_df['Payload Mass (kg)'] <= payload_range[1])
        ]
        if entered_site == 'ALL':
            fig = px.scatter(
                filtered_df,
                x='Payload Mass (kg)',
                y='class',
                color='Booster Version Category',
                title='Masa de carga útil vs. Resultado (Todos los sitios)',
                labels={'class': 'Resultado (0=Fallo, 1=Éxito)'}
            )
        else:
            filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
            if filtered_df.empty:
                return px.scatter(title=f'Sin datos para {entered_site}')
            fig = px.scatter(
                filtered_df,
                x='Payload Mass (kg)',
                y='class',
                color='Booster Version Category',
                title=f'Masa de carga útil vs. Resultado para {entered_site}',
                labels={'class': 'Resultado (0=Fallo, 1=Éxito)'}
            )
        return fig
    except Exception as e:
        print(f"Error en la función de callback: {e}")
        return px.scatter(title='Error al generar el gráfico')

# Ejecutar la aplicación
app.run(mode='inline', port=8051, debug=False)


# In[31]:


from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from IPython.display import display

# Cargar el conjunto de datos
spacex_df = pd.read_csv('spacex_launch_dash.csv')
display(spacex_df.head())
print("Columnas del DataFrame:", spacex_df.columns.tolist())
print("Valores únicos de 'class':", spacex_df['class'].unique())

# Obtener los sitios de lanzamiento únicos
launch_sites = spacex_df['Launch Site'].unique()
print("Sitios de lanzamiento:", launch_sites)
dropdown_options = [{'label': 'All Sites', 'value': 'ALL'}] + \
                  [{'label': site, 'value': site} for site in launch_sites]

# Obtener el rango de masa de carga útil
min_payload = spacex_df['Payload Mass (kg)'].min()
max_payload = spacex_df['Payload Mass (kg)'].max()
print("Rango de masa de carga útil:", min_payload, max_payload)

# Inicializar la aplicación Dash
app = Dash(__name__)

# Definir el diseño con todos los componentes
app.layout = html.Div([
    html.H1('SpaceX Launch Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',
        placeholder='Selecciona un sitio de lanzamiento',
        searchable=True
    ),
    html.Br(),
    dcc.Graph(id='success-pie-chart'),
    html.Br(),
    html.Label('Rango de masa de carga útil (kg):'),
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        marks={i: str(i) for i in range(0, 10001, 1000)},
        value=[min_payload, max_payload]
    ),
    html.Br(),
    dcc.Graph(id='success-payload-scatter-chart')
])

# Callback para el gráfico de pastel
@app.callback(
    Output(component_id='success-pie-chart', component_property='figure'),
    Input(component_id='site-dropdown', component_property='value')
)
def get_pie_chart(entered_site):
    try:
        if entered_site == 'ALL':
            class_counts = spacex_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title='Tasa de éxito de lanzamientos (Todos los sitios)'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        else:
            filtered_df = spacex_df[spacex_df['Launch Site'] == entered_site]
            if filtered_df.empty:
                return px.pie(names=['Sin datos'], values=[1], title=f'Sin datos para {entered_site}')
            class_counts = filtered_df['class'].value_counts().reset_index()
            class_counts.columns = ['class', 'count']
            fig = px.pie(
                class_counts,
                values='count',
                names='class',
                title=f'Tasa de éxito de lanzamientos para {entered_site}'
            )
            fig.update_traces(textinfo='percent+label', labels=['Fallo (0)', 'Éxito (1)'])
        return fig
    except Exception as e:
        print(f"Error en la función de callback: {e}")
        return px.pie(names=['Error'], values=[1], title='Error al generar el gráfico')

# Callback para el gráfico de dispersión
@app.callback(
    Output(component_id='success-payload-scatter-chart', component_property='figure'),
    [Input(component_id='site-dropdown', component_property='value'),
     Input(component_id='payload-slider', component_property='value')]
)
def get_scatter_chart(entered_site, payload_range):
    try:
        filtered_df = spacex_df[
            (spacex_df['Payload Mass (kg)'] >= payload_range[0]) &
            (spacex_df['Payload Mass (kg)'] <= payload_range[1])
        ]
        if entered_site == 'ALL':
            fig = px.scatter(
                filtered_df,
                x='Payload Mass (kg)',
                y='class',
                color='Booster Version Category',
                title='Masa de carga útil vs. Resultado (Todos los sitios)',
                labels={'class': 'Resultado (0=Fallo, 1=Éxito)'}
            )
        else:
            filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
            if filtered_df.empty:
                return px.scatter(title=f'Sin datos para {entered_site}')
            fig = px.scatter(
                filtered_df,
                x='Payload Mass (kg)',
                y='class',
                color='Booster Version Category',
                title=f'Masa de carga útil vs. Resultado para {entered_site}',
                labels={'class': 'Resultado (0=Fallo, 1=Éxito)'}
            )
        return fig
    except Exception as e:
        print(f"Error en la función de callback: {e}")
        return px.scatter(title='Error al generar el gráfico')

# Ejecutar la aplicación
app.run(mode='inline', port=8051, debug=False)

