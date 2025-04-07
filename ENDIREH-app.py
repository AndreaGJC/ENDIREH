import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import json
import os

# Configuración streamlit
st.set_page_config(layout='wide',page_icon=":material/groups:")

# Configurar tipo de letra
with open( "style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>',unsafe_allow_html=True)

#-----------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
#-----------------------------------------

df = pd.read_csv("data/raw/TSDem.csv",encoding='latin1').rename(columns=str.lower)

# Dejar variables de interés
vars = ['id_viv','sexo','cve_ent','nom_ent','fac_viv','dominio','edad','p2_9','p2_11','p2_13','p2_15']
df1 = df.loc[df['paren'] == 1,vars].drop_duplicates()

# Cambios de formato
df1['cve_ent'] = df1['cve_ent'].astype(str).str.zfill(2)
df1['nom_ent'] = df1['nom_ent'].str.lower()
df1['dominio'].replace({'U':'urbano','C':'conurbado','R':'rural'},inplace=True)
df1['sexo'].replace({1:'hombre',2:'mujer'},inplace=True)
df1['p2_13'] = np.where(df1['p2_15'].notna(),1,df1['p2_13'])
df1['p2_15'] = pd.Categorical(df1['p2_15'].replace({1:'empleado',2:'obrero',3:'jornalero o peón',4:'cuenta propia',5:'empleador',6:'trabajador sin pago',np.nan:'No definido o no trabaja'}),categories=['empleado', 'obrero', 'jornalero o peón', 'cuenta propia', 'empleador', 'trabajador sin pago','No definido o no trabaja'])
for var in ['p2_9','p2_11','p2_13']:
  df1[var] = pd.Categorical(df1[var].replace({1:'Si',2:'No',np.nan:'No definido'}),categories=['Si', 'No','No definido'])

# Crear nueva jefatura de ambos sexos
df1['obs'] = df1.groupby('id_viv')['id_viv'].transform('size')
df1['new_jefe'] = np.where(df1['obs']>=2,'compartida',df1['sexo'])
df1 = df1.drop('sexo',axis=1).drop_duplicates()

# como new_jefe>2, por la edad obs>=2, dejo al laboralmente activo y mayor de edad
df1 = df1.sort_values(['obs','id_viv','p2_13','edad'],ascending=[False,False,True,False]).drop_duplicates(subset='id_viv',keep='first')
df1['tot_dom'] = df1.groupby(['dominio'])['fac_viv'].transform('sum')
tot_nacion = df1['fac_viv'].sum()


#-----------------------------------------
# 2. INTERFAZ DE USUARIO
#-----------------------------------------

st.header('''Análisis de la jefatura de los hogares en México según el sexo y localización urbana-conurbada-rural''')

#-----------------------------------------
# SECCIÓN: Primera fila
#-----------------------------------------

col = st.columns((4,6),gap='medium')

with col[0]:
  dom_sel = st.selectbox('Selecciona uno de los tres dominios:',df1['dominio'].unique(), label_visibility='visible')
  jef_sel = st.selectbox('Selecciona uno de los tres tipos de jefatura:',df1['new_jefe'].unique(), label_visibility='visible')

# Base filtrada a selección usuario
df1_fil = df1.query('dominio == @dom_sel & new_jefe == @jef_sel').copy()
edad_prom = df1_fil['edad'].mean()

df1_fil['tot_edad'] = df1_fil.groupby(['edad'])['fac_viv'].transform('sum')
df1_fil['tot_esco'] = df1_fil.groupby(['p2_9'])['fac_viv'].transform('sum')
df1_fil['tot_indi'] = df1_fil.groupby(['p2_11'])['fac_viv'].transform('sum')
df1_fil['tot_trab'] = df1_fil.groupby(['p2_13'])['fac_viv'].transform('sum')
df1_fil['tot_ocup'] = df1_fil.groupby(['p2_15'])['fac_viv'].transform('sum')
df1_fil['tot_ent'] = df1_fil.groupby(['cve_ent'])['fac_viv'].transform('sum')
tot_viv = df1_fil['fac_viv'].sum()
tot_dom = df1_fil['tot_dom'].iloc[0]

# Para el grafico icicle
df1_fil['p2_9_with_question'] = '¿Asiste a la escuela?: ' + df1_fil['p2_9'].astype(str)
df1_fil['p2_11_with_question'] = '¿Habla alguna lengua indigena?: ' + df1_fil['p2_11'].astype(str)

with col[1]:
  st.markdown('''En el tratamiento de la información y desarrollo de esta *herramienta*, utilicé **Python**, junto con los paquetes **Streamlit**, **NumPy**, **Pandas** y **Geopandas**. El objetivo es responder ¿Cuál es la proporción de hogares con jefaturas de mujeres y hombres? ¿Este porcentaje, en los hogares con liderazgo femenino, presenta diferencias entre areas urbanas y rurales?''')

  st.markdown('''El **dominio urbano** integra a las viviendas localizadas en ciudades con más de 100 mil habitantes, el **conurbano** a las localizadas en zonas con más de 2,500 habitantes y hasta 99,999 habitantes, y el **rural** a las localizadas en zonas con menos de 2,500 habitantes''')

  st.markdown("<p style='margin-top:0.2cm;line-height:0;text-align:right;color:#ccc;font-size:13px;'>Abril, 2025 - Andrea Guerrero Jiménez</p>",True)


#-----------------------------------------
# SECCIÓN: Segunda fila
#-----------------------------------------

col = st.columns((2,4,4),gap='medium')

# Base col[1]
df1_col1 = df1_fil[['edad','tot_edad']].drop_duplicates()


with col[0]:
  # Base con variables necesarias
  st.subheader('Viviendas de filtrado')
  st.metric(label=f'{dom_sel}-{jef_sel}',border=True,
          value=f"{(tot_viv):,.0f}")
  
  st.divider()
  st.subheader('Viviendas totales')
  st.metric(label='nacional',border=True,
            value=f"{tot_nacion:,.0f}")
  st.metric(label=f'{dom_sel}',border=True,
            value=f"{tot_dom:,.0f}")
  
  st.divider()
  st.subheader('Participación de filtrado en totales')
  st.metric(label='% total nacional',border=True,
        value=f"{(tot_viv/tot_nacion)*100:,.2f}%")
  st.metric(label=f'% total {dom_sel}',border=True,
        value=f"{(tot_viv/tot_dom)*100:,.2f}%")


with col[1]:
  st.subheader('¿Cuántos años cumplidos tiene?')
  st.caption(f'dominio {dom_sel} y jefatura {jef_sel}')
  fig = px.bar(df1_col1,x='edad',y='tot_edad',
               labels={'tot_edad':'Número de viviendas','edad':'Años cumplidos'})
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
  fig.update_traces(
      texttemplate=None,
      hovertemplate='<b>%{x} años</b><br>%{y:,.0f} viviendas<extra></extra>')
  st.plotly_chart(fig)

with col[2]:
  st.subheader('¿Asiste actualmente a la escuela? ¿Habla algún dialecto o lengua indígena?')
  st.caption(f'dominio {dom_sel} y jefatura {jef_sel}')

  fig = px.icicle(df1_fil, path=[px.Constant("Total viviendas"), 'p2_9_with_question','p2_11_with_question'], values='fac_viv')
  fig.update_traces(root_color="lightgrey",texttemplate='<b>%{label}</b><br>%{value}',
                    hovertemplate='<b>%{label}</b><br>%{value} viviendas')
  fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
  st.plotly_chart(fig)


#-----------------------------------------
# SECCIÓN: Tecera fila
#-----------------------------------------

col = st.columns((3,7),gap='medium')

# Base col[0]
df1_col0 = df1_fil[['tot_trab','p2_13']].drop_duplicates()

# Base col[1]
df1_col1 = df1_fil[['tot_ocup','p2_15']].drop_duplicates()

with col[0]:
  st.subheader('¿Trabajó la semana pasada?')
  st.caption(f'dominio {dom_sel} y jefatura {jef_sel}')
  fig = px.pie(df1_col0,values='tot_trab',names='p2_13')
  fig.update_traces(textposition='auto',textinfo=None,
                    texttemplate='<b>%{label}</b><br>%{percent:.2%}',
                    hovertemplate='<b>%{label}</b><br>Porcentaje: %{percent:.2%}<br>Viviendas: %{value}<extra></extra>',
                    rotation=180)
  fig.update_layout(showlegend=False)
  st.plotly_chart(fig)

with col[1]:
  st.subheader('¿En su trabajo o negocio de la semana pasada fue ...?')
  st.caption(f'dominio {dom_sel} y jefatura {jef_sel}')
  fig = px.bar(df1_col1,x='p2_15',y='tot_ocup', color='p2_15', text='tot_ocup',
               labels={'p2_15':'Tipo de ocupación','tot_ocup':'Número de viviendas'},
               category_orders={'p2_15':['empleado', 'obrero', 'jornalero o peón', 'cuenta propia', 'empleador', 'trabajador sin pago']},
               color_discrete_sequence=px.colors.qualitative.T10)
  fig.update_traces(textposition='outside',texttemplate='%{text:,.0f}',
                    hovertemplate='<b>Trabaja como %{x}</b><br>Viviendas: %{y:,.0f}<extra></extra>')
  fig.update_layout(showlegend=False,margin={"r":0,"t":0,"l":0,"b":0})
  st.plotly_chart(fig)


#-----------------------------------------
# SECCIÓN: Cuarta fila
#-----------------------------------------

col = st.columns((2,8),gap='medium')

with col[0]:
  st.subheader('Insights clave')

  st.markdown("Es importante distinguir los hogares donde la jefatura esta encabezada por más de una persona, porque en ellos existe **algún grado de distribución de la autoridad y responsabilidad**.")

  st.markdown('La participación de los hogares con jefatura **masculina, desde el domino urbano pasando por el conurbado y siguiendo al rural, es mayor**. Por el contario, para aquellos con liderazgo femenino o compartido, esta proporción disminuye.')

  st.markdown('En México, durante el año 2021, **por cada hogar encabezado por una mujer hay dos encabezados por el sexo opuesto**.')


# Para la generación del mapa
mapa = df1_fil[['cve_ent','nom_ent','tot_ent']].drop_duplicates()

# Upload the json file
with open('data/raw/MGN/00ent.json') as f:
  mex_states = json.load(f)

# Hacer una función para crear el mapa
def hacer_mapa(base,col,label):
  fig = px.choropleth(base,geojson=mex_states,locations='cve_ent',featureidkey='properties.CVE_ENT',
                    color=col,hover_data='nom_ent',
                    labels=label)
  fig.update_geos(fitbounds="locations", visible=False)
  fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
  fig.update_traces(marker_line_width=0.5, hoverinfo="location+z")
  return fig


with col[1]:

  st.subheader(f'Distribución espacial del total de viviendas')
  st.caption(f'dominio {dom_sel} y jefatura {jef_sel}')
  fig = hacer_mapa(mapa,'tot_ent',
                    {'nom_ent':'Entidad federativa','cve_ent':'Clave entidad','tot_ent':'Viviendas'})
  st.plotly_chart(fig)


#-----------------------------------------
# SECCIÓN: Quinta fila
#-----------------------------------------
st.divider()
st.subheader('¿Qué fuentes de datos utilicé?')
with st.expander(rf"""Haz clic aquí para ver más **información de las fuentes de datos** así como algunas **precisiones en su visualización**:"""):

  st.markdown('''Utilicé los microdatos de uso público del módulo de **características sociodemográficas** de la **Encuesta Nacional sobre la Dinámica de las Relaciones de los Hogares** (ENDIREH) 2021. Ejercicio estadístico realizado por el Instituto Nacional de Estadística y Geográfia (INEGI) en México desde 2003. Además, para desarrollar la cartografía, tomé como capa base el **Marco Geoestadístico Nacional** (MGN) de 2021, también elaborado por INEGI.''')

  st.markdown('''En los hogares con más de un jefe, denominados **con jefatura compartida**, se considera la información del laboralmente activo y con mayor edad.''')
