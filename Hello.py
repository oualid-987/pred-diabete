import streamlit as st
import numpy as np
import pickle

# Charger le modèle pré-entrainé depuis le fichier .sav
loaded_model = pickle.load(open('/workspaces/pred-diabete/trained_model.sav', 'rb'))

# Interface Streamlit
st.title("Prédiction du Diabète")

# Sections pour les caractéristiques
st.header("Informations sur la personne")

# Utilisateur saisit les informations
pregnancies = st.slider("Nombre de grossesses", 0, 10, 1)
glucose = st.slider("Niveau de glucose", 0, 200, 100)
blood_pressure = st.slider("Pression sanguine", 0, 150, 75)
skin_thickness = st.slider("Épaisseur de la peau", 0, 50, 20)
insulin = st.slider("Niveau d'insuline", 0, 300, 150)
bmi = st.slider("Indice de masse corporelle (BMI)", 0.0, 50.0, 25.0)
dpf = st.slider("Fonction de pedigree du diabète", 0.0, 2.0, 1.0)
age = st.slider("Âge de la personne", 20, 80, 40)

# Bouton pour effectuer la prédiction
if st.button("Prédire"):
    # Créer un tableau numpy avec les informations saisies
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)

    # Effectuer la prédiction avec le modèle
    prediction = loaded_model.predict(input_data)

    # Afficher le résultat
    if prediction[0] == 0:
        st.success('La personne n\'est pas diabétique.')
    else:
        st.error('La personne est diabétique.')
