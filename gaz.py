import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns



df = pd.read_csv("Gas_Sensors_Measurements.csv")








st.title('Estimation volume de Gas')

# Tracer la courbe de régression avec seaborn (optionnel)
sns.regplot(x='MQ2', y='Gas_encoded', data=df, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
st.pyplot(plt)

# Titre du tableau de bord
st.header(":green[Présentation des données gas ]")


with st.sidebar:
    menu = st.selectbox('Menu', ["Tableau de bord", "MQ2",	"MQ3",	"MQ5",	"MQ6",	"MQ7",	"MQ8",	"MQ135",	"Gas",	"Gas_encoded","Image_Name_encoded"])

# Affichage en fonction de la sélection
if menu == "Tableau de bord":
    st.write(":green[Bienvenue sur l'application d'analyse descriptive du dataset.]")
    st.write(":blue[Ce tableau de bord montre un exemple de visualisation de données.]")

# Sélection interactive des colonnes
st.header(":red[Sélection Interactive]")
option = st.selectbox(
    'Choisissez une colonne à afficher',
    ("MQ2",	"MQ3",	"MQ5",	"MQ6",	"MQ7",	"MQ8",	"MQ135",	"Gas",	"Gas_encoded","Image_Name_encoded")
)
st.write('Vous avez sélectionné:', option)

# Filtrer les données en fonction de la sélection
#filtered_df = df[[option]]
#st.line_chart(filtered_df)

# Charger le dataset à nouveau
@st.cache_data
def load_data():
    df = pd.read_csv("Gas_Sensors_Measurements.csv")
    return df

df = load_data()

# Menu de navigation
menu = ["Accueil", "Voir les données", "Statistiques descriptives"]
choix = st.sidebar.selectbox("Navigation", menu)

# Affichage en fonction de la sélection
if choix == "Accueil":
    st.write(":green[Bienvenue sur l'application d'analyse descriptive du dataset.]")
elif choix == "Voir les données":
    st.header("Voir les données")
    st.write("Voici un aperçu du dataset :")
    st.write(df.head())
elif choix == "Statistiques descriptives":
    st.header("Statistiques descriptives")

    # Sélection des colonnes numériques
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    col = st.selectbox("Sélectionnez une colonne pour voir les statistiques descriptives", numeric_cols)

    # Afficher les statistiques descriptives
    st.write(df[col].describe())

    # Afficher la distribution de la colonne sélectionnée
    st.subheader(f"Distribution de {col}")

    
# Prédiction

st.title(":blue[volume de Gaz]")

# Entrer les valeurs par l'utilisateur
st.sidebar.subheader("Entrez les valeurs")
tpe = st.sidebar.slider("MQ2", min_value=1.0, max_value=900.0, value=6.0)
oxg = st.sidebar.slider("MQ3", min_value=1.0, max_value=900.0, value=6.0)
pls = st.sidebar.slider("MQ5", min_value=1.0, max_value=900.0, value=6.0)
glm = st.sidebar.slider("MQ6", min_value=1.0, max_value=900.0, value=6.0)
tsn = st.sidebar.slider("MQ7", min_value=1.0, max_value=900.0, value=6.0)
mq8 = st.sidebar.slider("MQ8", min_value=1.0, max_value=900.0, value=6.0)
mq135 = st.sidebar.slider("MQ135", min_value=1.0, max_value=900.0, value=6.0)
gaz = st.sidebar.slider("Gas", min_value=1.0, max_value=900.0, value=6.0)
GE = st.sidebar.slider("Gas_encoded", min_value=1.0, max_value=900.0, value=6.0)
sensor_10 = st.sidebar.slider("Sensor 10", min_value=1.0, max_value=900.0, value=6.0)
sensor_11 = st.sidebar.slider("Sensor 11", min_value=1.0, max_value=900.0, value=6.0)

# Mise à jour de input_data
input_data = np.array([[tpe, oxg, pls, glm, tsn, mq8, mq135, gaz, GE, sensor_10, sensor_11]])

# Faire la prédiction
prediction = model.predict(input_data)[0]
st.write(f"La prédiction pour les valeurs données est: {prediction}")

# Footer
st.sidebar.text("© 2024 Mamadou Mbow * Machine Learning")


