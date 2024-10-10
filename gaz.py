import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection  import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns



df = pd.read_csv("Gas_Sensors_Measurements.csv")


# Initialiser le LabelEncoder
encoder = LabelEncoder()
# Encodage de la colonne 'Gas'
df['Gas_encoded'] = encoder.fit_transform(df['Gas'])
# Encodage de la colonne 'Corresponding Image Name'
df['Image_Name_encoded'] = encoder.fit_transform(df['Corresponding Image Name'])


#encoder = LabelEncoder()

# Initialiser le LabelEncoder
encoder = LabelEncoder()
# Encodage de la colonne 'Gas'
df['Gas'] = encoder.fit_transform(df['Gas'])


# Initialiser le LabelEncoder
encoder = LabelEncoder()
# Encodage de la colonne 'Corresponding Image Name'
df['Image_Name_encoded'] = encoder.fit_transform(df['Corresponding Image Name'])

# Initialiser le LabelEncoder
label_encoder = LabelEncoder()
# Appliquer l'encodage sur la colonne 'Category'
df['Corresponding Image Name'] = label_encoder.fit_transform(df['Corresponding Image Name'])

# Utilisez drop() avec des parenthèses
x = df.drop(columns=['Corresponding Image Name'])

y = df['Corresponding Image Name']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2 , random_state=42)


scaler = MinMaxScaler()

# Sélectionner uniquement les colonnes numériques pour la normalisation
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

# Appliquer la normalisation sur les colonnes numériques
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])


model = LinearRegression()
model.fit(x_test , y_test)
# Prédictions pour la ligne de régression
predictions = model.predict(x)

# Calculer les métriques
mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)









st.title('Estimation volume de Gas')

# Tracer la courbe de régression avec seaborn (optionnel)
plt.figure(figsize=(10, 6))
sns.regplot(x='MQ2', y='Gas_encoded', data=df, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
plt.title('Courbe Linéaire entre MQ2 et Measurement avec Seaborn')
plt.xlabel('MQ2')
plt.ylabel('Measurement')
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


