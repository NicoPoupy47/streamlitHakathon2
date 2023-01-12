#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

#file = open(r'C:/Users/nicol/Downloads/tableH2.csv')
df = pd.read_csv('/Users/philistineserour/Downloads/tableH4.csv')

#df = pd.read_csv(file)


# Image animée

import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_json = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_ev1cfn9h.json")
st_lottie(lottie_json, key = None)

# Regression

X = df[['surface_reelle_bati','nombre_pieces_principales', 'type_local','code_commune', 'code_departement']]
y = df['valeur_fonciere']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, random_state=36)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model_LR = LinearRegression().fit(X_train, y_train)
df['predict_LR'] = model_LR.predict(X)

# Predcition sur streamlit

st.title("Estimation de la valeur d'un bien immobilier")

with st.form(key='Recommandation'):
    surface = st.selectbox("Surface habitable ?", options=df['surface_reelle_bati'].unique())
    piece = st.selectbox('Nombre de pièces ?', options=df['nombre_pieces_principales'].unique())
    bien = st.multiselect("Type de bien ?", options=df['type_local'].unique())
    commune = st.multiselect("Code commune ?", options=df['code_commune'].unique())
    departement = st.multiselect("Département ?", options=df['code_departement'].unique())
    submit = st.form_submit_button(label='Estimation')

if submit:
    # Associer à colonne fac

    #crew_test = int(crew)
    #pax_test = int(pax)

    #filt_model = crash['Aircraft'] == ''.join(model)
    #model_test = crash.loc[filt_model, 'Aircraft_fac'].values[0]


    #filt_departure = crash['departure'] == ''.join(depart)
    #departure_test = crash.loc[filt_model, 'departure_fac'].values[0]

    #filt_operator = crash['Operator'] == ''.join(operator)
    #loc_test = crash.loc[filt_model, 'Operator_fac'].values[0]

    my_data = np.array([surface, piece, bien, commune, departement ]).reshape(1, 5)

    st.success(f"Valeur estimée du bien immobilier : ",icon="✅")
    st.header(f"{''.join(model_LR.predict(my_data))}")

